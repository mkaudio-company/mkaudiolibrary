//! Continuous Wavelet Transform (complex Morlet wavelet).

use std::f32::consts::PI;

use super::complex::Complex64;

/// Direct time-domain CWT with a complex Morlet wavelet.
///
/// For each scale, correlates the signal against a Gaussian-windowed
/// complex exponential kernel (real and imaginary parts convolved
/// separately via SIMD-accelerated dot products, see [`crate::simd::dot`]).
/// `O(n * scales * kernel_width)` - kernel width grows with scale, so very
/// large scales get proportionally more expensive; this trades some
/// efficiency (an FFT-based convolution would be `O(n log n)` per scale)
/// for a much simpler, easier-to-verify implementation.
///
/// # Arguments
/// * `signal` - input samples
/// * `scales` - wavelet scales to evaluate (larger scale = lower center frequency)
/// * `w0` - Morlet wavelet's center angular frequency parameter (6.0 is a common default)
/// * `sample_rate` - sample rate of `signal`, in Hz
///
/// # Returns
/// One row per scale, each the same length as `signal`.
pub fn cwt(signal: &[f32], scales: &[f32], w0: f32, sample_rate: f32) -> Vec<Vec<Complex64>> {
    let n = signal.len();

    scales
        .iter()
        .map(|&scale| {
            let scale = scale.max(1e-9);
            let half_width = ((4.0 * scale * sample_rate).ceil() as isize).max(1);
            let kernel_len = (2 * half_width + 1) as usize;

            let norm = PI.powf(-0.25) / scale.sqrt();
            let mut kernel_re = Vec::with_capacity(kernel_len);
            let mut kernel_im = Vec::with_capacity(kernel_len);
            for k in -half_width..=half_width {
                let t = k as f32 / sample_rate / scale;
                let envelope = norm * (-t * t / 2.0).exp();
                let c = Complex64::from_polar(envelope, w0 * t);
                kernel_re.push(c.re);
                kernel_im.push(c.im);
            }

            (0..n)
                .map(|center| {
                    let lo = (center as isize - half_width).max(0) as usize;
                    let hi = ((center as isize + half_width + 1).max(0) as usize).min(n);
                    if hi <= lo {
                        return Complex64::new(0.0, 0.0);
                    }

                    // Offset into the kernel where it starts overlapping the signal
                    // (nonzero only near the signal's edges, where the kernel would
                    // otherwise run off the start).
                    let k_start = (lo as isize - (center as isize - half_width)).max(0) as usize;
                    let segment = &signal[lo..hi];
                    let len = segment.len();

                    Complex64::new(
                        crate::simd::dot(segment, &kernel_re[k_start..k_start + len]),
                        crate::simd::dot(segment, &kernel_im[k_start..k_start + len]),
                    )
                })
                .collect()
        })
        .collect()
}

/// Convenience: geometrically-spaced scales covering `[min_freq, max_freq]`
/// (converted via the Morlet wavelet's approximate center-frequency
/// relationship `f_c = w0 / (2*pi*scale)`), `voices_per_octave` scales per
/// octave - the usual way to lay out a CWT's scale axis for visualization.
pub fn log_scales(min_freq: f32, max_freq: f32, voices_per_octave: usize, w0: f32) -> Vec<f32> {
    if min_freq <= 0.0 || max_freq <= min_freq || voices_per_octave == 0 {
        return Vec::new();
    }

    let octaves = (max_freq / min_freq).log2();
    let num_scales = (octaves * voices_per_octave as f32).ceil() as usize;

    (0..=num_scales)
        .map(|i| {
            let freq = min_freq * 2f32.powf(i as f32 / voices_per_octave as f32);
            w0 / (2.0 * PI * freq)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cwt_output_shape_matches_scales_and_signal_length() {
        let signal: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let scales = [0.01, 0.02, 0.04];
        let result = cwt(&signal, &scales, 6.0, 8000.0);
        assert_eq!(result.len(), scales.len());
        for row in &result {
            assert_eq!(row.len(), signal.len());
        }
    }

    #[test]
    fn cwt_responds_more_strongly_at_matching_scale() {
        // A pure tone should produce larger-magnitude coefficients at the
        // scale whose Morlet center frequency matches the tone than at a
        // scale an octave away.
        let sample_rate = 8000.0;
        let tone_freq = 200.0;
        let signal: Vec<f32> = (0..4000)
            .map(|i| (2.0 * PI * tone_freq * i as f32 / sample_rate).sin())
            .collect();

        let w0 = 6.0;
        let matching_scale = w0 / (2.0 * PI * tone_freq);
        let off_scale = w0 / (2.0 * PI * (tone_freq * 4.0));

        let result = cwt(&signal, &[matching_scale, off_scale], w0, sample_rate);
        let mid = signal.len() / 2;
        let matching_energy = result[0][mid].norm();
        let off_energy = result[1][mid].norm();

        assert!(
            matching_energy > off_energy,
            "matching-scale energy {} should exceed off-scale energy {}",
            matching_energy,
            off_energy
        );
    }
}
