//! Constant-Q Transform.
//!
//! Direct time-domain correlation against per-bin, Hann-windowed complex
//! exponential kernels sized so each bin has the same Q (center frequency
//! / bandwidth) - the defining property of a CQT, and why (unlike an STFT)
//! low bins get long, high-frequency-resolution kernels and high bins get
//! short, high-time-resolution ones. This is the straightforward
//! "time-domain CQT"; Brown & Puckette's sparse-FFT-kernel method is faster
//! for large kernel counts but substantially more involved to implement
//! and verify correct.

use std::f32::consts::PI;

use super::complex::Complex64;
use super::stft::{WindowFunction, window};

/// Constant-Q transform configuration.
#[derive(Debug, Clone, Copy)]
pub struct CqtConfig {
    /// Sample rate of the input signal in Hz.
    pub sample_rate: f32,
    /// Center frequency of bin 0, in Hz.
    pub min_freq: f32,
    /// Total number of output frequency bins.
    pub num_bins: usize,
    /// Bins per octave (sets the constant Q / kernel length per bin).
    pub bins_per_octave: usize,
    /// Hop size between analysis frames, in samples.
    pub hop_size: usize,
}

/// Run the CQT, returning one row per output frame (spaced `hop_size`
/// samples apart), each `num_bins` complex values wide (bin 0 = `min_freq`).
pub fn cqt(signal: &[f32], config: &CqtConfig) -> Vec<Vec<Complex64>> {
    if config.num_bins == 0
        || config.hop_size == 0
        || config.bins_per_octave == 0
        || signal.is_empty()
    {
        return Vec::new();
    }

    let q = 1.0 / (2f32.powf(1.0 / config.bins_per_octave as f32) - 1.0);

    let kernels: Vec<(Vec<f32>, Vec<f32>)> = (0..config.num_bins)
        .map(|bin| {
            let freq = config.min_freq * 2f32.powf(bin as f32 / config.bins_per_octave as f32);
            let kernel_len =
                ((q * config.sample_rate / freq).round() as usize).clamp(1, signal.len().max(1));
            let win = window(WindowFunction::Hann, kernel_len);
            let norm = 1.0 / kernel_len as f32;

            let mut re = Vec::with_capacity(kernel_len);
            let mut im = Vec::with_capacity(kernel_len);
            for (n, &w) in win.iter().enumerate() {
                let angle = -2.0 * PI * freq * n as f32 / config.sample_rate;
                re.push(w * angle.cos() * norm);
                im.push(w * angle.sin() * norm);
            }
            (re, im)
        })
        .collect();

    let num_frames = (signal.len() - 1) / config.hop_size + 1;

    (0..num_frames)
        .map(|frame| {
            let center = frame * config.hop_size;
            kernels
                .iter()
                .map(|(re, im)| {
                    let half = re.len() / 2;
                    let lo = center.saturating_sub(half);
                    let hi = (lo + re.len()).min(signal.len());
                    if hi <= lo {
                        return Complex64::new(0.0, 0.0);
                    }

                    let segment = &signal[lo..hi];
                    let len = segment.len();
                    Complex64::new(
                        crate::simd::dot(segment, &re[..len]),
                        crate::simd::dot(segment, &im[..len]),
                    )
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cqt_output_shape_matches_config() {
        let signal: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.1).sin()).collect();
        let config = CqtConfig {
            sample_rate: 8000.0,
            min_freq: 55.0,
            num_bins: 24,
            bins_per_octave: 12,
            hop_size: 512,
        };
        let result = cqt(&signal, &config);
        assert!(!result.is_empty());
        for row in &result {
            assert_eq!(row.len(), 24);
        }
    }

    #[test]
    fn cqt_responds_more_strongly_at_matching_bin() {
        let sample_rate = 8000.0;
        let tone_freq = 220.0; // matches bin index 0 below (min_freq = 220Hz)
        let signal: Vec<f32> = (0..8000)
            .map(|i| (2.0 * PI * tone_freq * i as f32 / sample_rate).sin())
            .collect();

        let config = CqtConfig {
            sample_rate,
            min_freq: 220.0,
            num_bins: 12,
            bins_per_octave: 12,
            hop_size: 1024,
        };
        let result = cqt(&signal, &config);
        let mid_frame = &result[result.len() / 2];

        let matching_bin_energy = mid_frame[0].norm();
        let far_bin_energy = mid_frame[11].norm(); // an octave away
        assert!(
            matching_bin_energy > far_bin_energy,
            "matching-bin energy {} should exceed far-bin energy {}",
            matching_bin_energy,
            far_bin_energy
        );
    }
}
