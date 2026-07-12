//! Zero-stuff + FIR-lowpass oversampling for reducing aliasing around
//! nonlinear processing (saturation, clipping, etc).
//!
//! This is the same zero-stuff/filter/decimate technique [`crate::sim`]
//! uses internally for its own components, built here on [`super::fir`] so
//! it shares the crate's SIMD-accelerated convolution hot loop.

use crate::dsp::fir::design_lowpass;

/// Upsample -> process -> downsample wrapper around any per-sample closure.
///
/// Reduces aliasing from nonlinear processing (e.g. [`super::Saturation`])
/// by running the nonlinearity at a higher internal sample rate, where
/// harmonics generated above the original Nyquist frequency fold back into
/// the (still supersonic) oversampled Nyquist band instead of the audible
/// range, then filtering and decimating back down.
///
/// # Example
///
/// ```ignore
/// use mkaudiolibrary::dsp::{Oversampler, Saturation};
///
/// let mut os = Oversampler::new(4, 44100.0);
/// let sat = Saturation::new(10.0, 10.0, 1.0, 1.0, 0.0, false);
/// let wet = os.process(1.0, |x| sat.process(x));
/// ```
pub struct Oversampler {
    factor: usize,
    upsample_taps: usize,
    up_history: Vec<f32>,
    up_pos: usize,
    up_coeffs: Vec<f32>,
    down_history: Vec<f32>,
    down_pos: usize,
    down_coeffs: Vec<f32>,
}

impl Oversampler {
    /// Create an oversampler for the given integer factor (e.g. 2, 4, 8)
    /// at the given (pre-oversampling) sample rate.
    pub fn new(factor: usize, sample_rate: f32) -> Self {
        assert!(factor >= 1, "oversampling factor must be >= 1");

        let taps = if factor <= 1 {
            1
        } else {
            33usize.min(8 * factor + 1) | 1
        };
        let cutoff = sample_rate * 0.5 / factor as f32 * 0.9;
        let coeffs = if factor <= 1 {
            vec![1.0]
        } else {
            design_lowpass(taps, sample_rate * factor as f32, cutoff)
        };

        Self {
            factor,
            upsample_taps: coeffs.len(),
            up_history: vec![0.0; coeffs.len()],
            up_pos: 0,
            up_coeffs: coeffs.clone(),
            down_history: vec![0.0; coeffs.len()],
            down_pos: 0,
            down_coeffs: coeffs,
        }
    }

    /// The oversampling factor.
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// Push one sample through the FIR history buffer and return the
    /// filtered result (shared by the upsample and downsample stages,
    /// which each keep independent state).
    #[inline]
    fn fir_step(history: &mut [f32], pos: &mut usize, coeffs: &[f32], x: f32) -> f32 {
        let len = coeffs.len();
        history[*pos] = x;

        let mut sum = 0.0;
        let mut idx = *pos;
        for &c in coeffs {
            sum += c * history[idx];
            idx = if idx == 0 { len - 1 } else { idx - 1 };
        }

        *pos = (*pos + 1) % len;
        sum
    }

    /// Process a single input sample: upsample by `factor`, run `f` at the
    /// oversampled rate, downsample back to the original rate.
    #[inline]
    pub fn process<F: FnMut(f32) -> f32>(&mut self, input: f32, mut f: F) -> f32 {
        if self.factor <= 1 {
            return f(input);
        }

        let factor = self.factor as f32;
        let mut last_down = 0.0;
        for i in 0..self.factor {
            // Zero-stuff: only the first sub-sample of each group carries
            // the input value (scaled by `factor` to preserve energy
            // through the lowpass' passband attenuation).
            let stuffed = if i == 0 { input * factor } else { 0.0 };
            let up = Self::fir_step(
                &mut self.up_history,
                &mut self.up_pos,
                &self.up_coeffs,
                stuffed,
            );
            let processed = f(up);
            last_down = Self::fir_step(
                &mut self.down_history,
                &mut self.down_pos,
                &self.down_coeffs,
                processed,
            );
        }
        last_down
    }

    /// Process a buffer through the given per-sample nonlinearity at the
    /// oversampled rate.
    pub fn run<F: FnMut(f32) -> f32>(&mut self, input: &[f32], output: &mut [f32], f: F) {
        let len = input.len().min(output.len());

        let mut f = f;
        for index in 0..len {
            output[index] = self.process(input[index], &mut f);
        }
    }

    /// Reset internal filter state.
    pub fn reset(&mut self) {
        self.up_history.fill(0.0);
        self.up_pos = 0;
        self.down_history.fill(0.0);
        self.down_pos = 0;
    }

    /// Number of taps used by the upsample/downsample lowpass filters.
    pub fn taps(&self) -> usize {
        self.upsample_taps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bypass_factor_one() {
        let mut os = Oversampler::new(1, 44100.0);
        for i in 0..64 {
            let x = (i as f32) * 0.01;
            assert_eq!(os.process(x, |v| v * 2.0), x * 2.0);
        }
    }

    #[test]
    fn test_oversampling_no_nan() {
        let mut os = Oversampler::new(4, 44100.0);
        for i in 0..1000 {
            let x = (i as f32 * 0.05).sin();
            let y = os.process(x, |v| v.tanh());
            assert!(y.is_finite(), "oversampled output not finite at {i}");
        }
    }

    #[test]
    fn test_oversampling_dc_passthrough() {
        let mut os = Oversampler::new(4, 44100.0);
        let mut last = 0.0;
        for _ in 0..500 {
            last = os.process(1.0, |v| v);
        }
        assert!(
            (last - 1.0).abs() < 0.05,
            "DC should pass through oversampling near-unity, got {last}"
        );
    }
}
