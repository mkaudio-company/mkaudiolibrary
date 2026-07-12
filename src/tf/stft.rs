//! Windowing, Short-Time Fourier Transform, and multi-resolution STFT.

use std::f32::consts::PI;

use super::complex::Complex64;
use super::fft::{fft, ifft};

/// Analysis window shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// No windowing (all-ones); sharpest spectral leakage.
    Rectangular,
    /// Raised-cosine window that tapers to zero at both edges.
    Hann,
    /// Raised-cosine window with a small nonzero edge value; lower first
    /// sidelobe than Hann at the cost of slower sidelobe rolloff.
    Hamming,
    /// Three-term cosine window with very low sidelobes at the cost of a
    /// wider main lobe than Hann/Hamming.
    Blackman,
}

/// Generate a window of the given length and shape.
pub fn window(kind: WindowFunction, len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }

    let n = (len - 1) as f32;
    (0..len)
        .map(|i| {
            let x = i as f32 / n;
            match kind {
                WindowFunction::Rectangular => 1.0,
                WindowFunction::Hann => 0.5 - 0.5 * (2.0 * PI * x).cos(),
                WindowFunction::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
                WindowFunction::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos()
                }
            }
        })
        .collect()
}

/// STFT configuration: FFT size, hop size (in samples), and analysis window shape.
#[derive(Debug, Clone, Copy)]
pub struct StftConfig {
    /// FFT size in samples (also the analysis window length).
    pub fft_size: usize,
    /// Hop size between successive frames, in samples.
    pub hop_size: usize,
    /// Analysis window shape applied before each frame's FFT.
    pub window: WindowFunction,
}

impl StftConfig {
    /// Construct an STFT configuration from its FFT size, hop size, and window shape.
    pub fn new(fft_size: usize, hop_size: usize, window: WindowFunction) -> Self {
        Self {
            fft_size,
            hop_size,
            window,
        }
    }
}

/// Short-Time Fourier Transform: splits `signal` into overlapping,
/// windowed frames and FFTs each one.
///
/// Returns one row per frame, each row holding `fft_size` complex bins
/// (the full spectrum, not just the non-negative-frequency half - useful
/// since [`istft`] needs it for exact reconstruction).
pub fn stft(signal: &[f32], config: &StftConfig) -> Vec<Vec<Complex64>> {
    if config.fft_size == 0 || config.hop_size == 0 || signal.len() < config.fft_size {
        return Vec::new();
    }

    let win = window(config.window, config.fft_size);
    let num_frames = (signal.len() - config.fft_size) / config.hop_size + 1;
    let mut scratch = vec![0.0f32; config.fft_size];

    (0..num_frames)
        .map(|frame| {
            let start = frame * config.hop_size;
            crate::simd::mul_elementwise(
                &mut scratch,
                &signal[start..start + config.fft_size],
                &win,
            );
            let windowed: Vec<Complex64> =
                scratch.iter().map(|&x| Complex64::new(x, 0.0)).collect();
            fft(&windowed)
        })
        .collect()
}

/// Inverse STFT via overlap-add. Approximately reconstructs the original
/// signal when frames are unmodified and the window/hop combination
/// satisfies the constant-overlap-add condition (e.g. Hann window at 50%
/// hop); for other combinations (or after modifying magnitudes/phases) the
/// result is the best overlap-add reconstruction, not an exact inverse.
pub fn istft(frames: &[Vec<Complex64>], config: &StftConfig) -> Vec<f32> {
    if frames.is_empty() || config.fft_size == 0 {
        return Vec::new();
    }

    let win = window(config.window, config.fft_size);
    let output_len = (frames.len() - 1) * config.hop_size + config.fft_size;
    let mut output = vec![0.0; output_len];
    let mut window_sum = vec![0.0; output_len];

    for (frame_index, frame) in frames.iter().enumerate() {
        let time_domain = ifft(frame);
        let start = frame_index * config.hop_size;
        for i in 0..config.fft_size {
            output[start + i] += time_domain[i].re * win[i];
            window_sum[start + i] += win[i] * win[i];
        }
    }

    for i in 0..output_len {
        if window_sum[i] > 1e-12 {
            output[i] /= window_sum[i];
        }
    }
    output
}

/// One [`stft`] result per configuration - lets a caller analyze the same
/// signal at multiple time/frequency resolutions at once (e.g. short
/// windows for transient detail, long windows for tonal/harmonic detail),
/// which is the usual motivation for a "multi-resolution STFT".
pub fn mr_stft(signal: &[f32], configs: &[StftConfig]) -> Vec<Vec<Vec<Complex64>>> {
    configs.iter().map(|config| stft(signal, config)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints_are_zero() {
        let w = window(WindowFunction::Hann, 8);
        assert!(w[0].abs() < 1e-12);
        assert!(w[7].abs() < 1e-12);
    }

    #[test]
    fn istft_reconstructs_signal_with_hann_50_percent_overlap() {
        let signal: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.05).sin()).collect();
        let config = StftConfig::new(256, 128, WindowFunction::Hann);

        let frames = stft(&signal, &config);
        let reconstructed = istft(&frames, &config);

        // Compare over the region fully covered by overlap-add (skip the
        // first/last window's worth of samples, where edge effects live).
        let skip = config.fft_size;
        for i in skip..(signal.len() - skip).min(reconstructed.len()) {
            assert!(
                (signal[i] - reconstructed[i]).abs() < 1e-4,
                "sample {}: {} vs {}",
                i,
                signal[i],
                reconstructed[i]
            );
        }
    }

    #[test]
    fn mr_stft_returns_one_result_per_config() {
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let configs = [
            StftConfig::new(128, 64, WindowFunction::Hann),
            StftConfig::new(512, 256, WindowFunction::Hann),
        ];
        let results = mr_stft(&signal, &configs);
        assert_eq!(results.len(), 2);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());
    }
}
