//! IIR filtering: RBJ biquad sections and Butterworth cascades.

use no_denormals::*;
use std::f32::consts::PI;

/// Biquad filter response type, using the RBJ Audio EQ Cookbook formulas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BiquadType {
    /// 2nd-order lowpass (12dB/octave rolloff above `freq`).
    LowPass,
    /// 2nd-order highpass (12dB/octave rolloff below `freq`).
    HighPass,
    /// Constant skirt gain bandpass centered at `freq`, width set by `q`.
    BandPass,
    /// Narrow band-reject (notch) filter centered at `freq`, width set by `q`.
    Notch,
    /// Passes all frequencies unattenuated but shifts phase around `freq`.
    AllPass,
    /// Parametric peaking EQ; `gain_db` sets boost/cut at the center frequency.
    Peak,
    /// Shelf below `freq`; `gain_db` sets boost/cut.
    LowShelf,
    /// Shelf above `freq`; `gain_db` sets boost/cut.
    HighShelf,
}

/// Second-order IIR section (biquad), Direct Form II Transposed.
///
/// Direct Form II Transposed is used because it only needs two state
/// variables and is well-behaved numerically under coefficient modulation,
/// making it safe to retune (e.g. sweep a filter cutoff) between blocks.
///
/// # Transfer Function
///
/// `H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)`
///
/// Coefficients are computed from the RBJ Audio EQ Cookbook formulas for
/// each [`BiquadType`].
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Design a biquad section.
    ///
    /// # Arguments
    /// * `kind` - Filter response type
    /// * `sample_rate` - Sample rate in Hz
    /// * `freq` - Center/cutoff frequency in Hz
    /// * `q` - Quality factor (bandwidth for `BandPass`/`Notch`, resonance
    ///   for `LowPass`/`HighPass`, slope shaping for shelves). Must be > 0.
    /// * `gain_db` - Boost/cut in dB, used only by `Peak`/`LowShelf`/`HighShelf`.
    pub fn new(kind: BiquadType, sample_rate: f32, freq: f32, q: f32, gain_db: f32) -> Self {
        let q = q.max(1e-4);
        let omega = 2.0 * PI * freq.max(1.0).min(sample_rate * 0.499) / sample_rate;
        let sin_w = omega.sin();
        let cos_w = omega.cos();
        let alpha = sin_w / (2.0 * q);
        let a = 10.0f32.powf(gain_db / 40.0);

        let (b0, b1, b2, a0, a1, a2) = match kind {
            BiquadType::LowPass => {
                let b1 = 1.0 - cos_w;
                let b0 = b1 * 0.5;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighPass => {
                let b0 = (1.0 + cos_w) * 0.5;
                let b1 = -(1.0 + cos_w);
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::BandPass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::AllPass => {
                let b0 = 1.0 - alpha;
                let b1 = -2.0 * cos_w;
                let b2 = 1.0 + alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Peak => {
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_w;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha / a;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::LowShelf => {
                let sqrt_a = a.sqrt();
                let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w + two_sqrt_a_alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w - two_sqrt_a_alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w + two_sqrt_a_alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighShelf => {
                let sqrt_a = a.sqrt();
                let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w + two_sqrt_a_alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w - two_sqrt_a_alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w + two_sqrt_a_alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Design a biquad directly from normalized coefficients
    /// (`a0` is implicitly 1; pass already-normalized `b0..a2`).
    ///
    /// Useful for building [`IirFilter`] sections with a custom design
    /// (e.g. a specific Butterworth/Chebyshev stage) not covered by
    /// [`Biquad::new`].
    pub fn from_coefficients(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Reset internal filter state (clears any stored history).
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Process a single sample (Direct Form II Transposed).
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }

    /// Process a buffer of samples.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            no_denormals(|| {
                for index in 0..input.len().min(output.len()) {
                    output[index] = self.process(input[index]);
                }
            });
        }
    }
}

/// Cascade of [`Biquad`] sections for higher-order IIR filtering.
///
/// A single biquad gives a 2nd-order (12dB/octave) rolloff; cascading `N`
/// sections multiplies their responses for `2N`-order filtering with a
/// steeper rolloff. [`IirFilter::butterworth`] builds a maximally-flat
/// Butterworth cascade at a given even order; sections can also be pushed
/// individually for custom designs (e.g. mixing a shelf with a peak).
pub struct IirFilter {
    sections: Vec<Biquad>,
}

impl IirFilter {
    /// Create an empty cascade (no sections; `process` is a no-op passthrough).
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
        }
    }

    /// Append a section to the cascade (processed after all existing sections).
    pub fn push(&mut self, section: Biquad) {
        self.sections.push(section);
    }

    /// Number of biquad sections in the cascade.
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Whether the cascade has no sections.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Build a maximally-flat Butterworth lowpass/highpass cascade.
    ///
    /// `order` is the total filter order and must be even (each biquad
    /// section contributes 2nd order); each section's Q is set per the
    /// standard Butterworth pole distribution
    /// `Q_k = 1 / (2 * cos((2k+1)*pi / (2*order)))`.
    ///
    /// # Panics
    /// Panics if `order` is zero or odd.
    pub fn butterworth(kind: BiquadType, order: usize, sample_rate: f32, cutoff: f32) -> Self {
        assert!(
            order > 0 && order.is_multiple_of(2),
            "Butterworth cascade order must be a positive even number, got {order}"
        );
        assert!(
            matches!(kind, BiquadType::LowPass | BiquadType::HighPass),
            "Butterworth cascade only supports LowPass/HighPass"
        );

        let n_sections = order / 2;
        let mut sections = Vec::with_capacity(n_sections);
        for k in 0..n_sections {
            let q = 1.0 / (2.0 * (((2 * k + 1) as f32) * PI / (2.0 * order as f32)).cos());
            sections.push(Biquad::new(kind, sample_rate, cutoff, q, 0.0));
        }

        Self { sections }
    }

    /// Reset all sections' internal state.
    pub fn reset(&mut self) {
        for s in &mut self.sections {
            s.reset();
        }
    }

    /// Process a single sample through every section in series.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let mut x = input;
        for s in &mut self.sections {
            x = s.process(x);
        }
        x
    }

    /// Process a buffer of samples.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            no_denormals(|| {
                for index in 0..input.len().min(output.len()) {
                    output[index] = self.process(input[index]);
                }
            });
        }
    }
}

impl Default for IirFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowpass_attenuates_high_frequency() {
        let sr = 48000.0;
        let mut lp = Biquad::new(BiquadType::LowPass, sr, 500.0, 0.707, 0.0);

        // Settle, then measure RMS response to a low and a high tone.
        let measure = |filter: &mut Biquad, freq: f32| -> f32 {
            filter.reset();
            let n = (sr / freq * 20.0) as usize;
            let mut sum_sq = 0.0;
            let mut count = 0;
            for i in 0..n {
                let t = i as f32 / sr;
                let x = (2.0 * PI * freq * t).sin();
                let y = filter.process(x);
                if i > n / 2 {
                    sum_sq += y * y;
                    count += 1;
                }
            }
            (sum_sq / count as f32).sqrt()
        };

        let low_rms = measure(&mut lp, 50.0);
        let high_rms = measure(&mut lp, 8000.0);

        assert!(
            high_rms < low_rms * 0.2,
            "lowpass should attenuate 8kHz much more than 50Hz: low={low_rms}, high={high_rms}"
        );
    }

    #[test]
    fn test_highpass_attenuates_low_frequency() {
        let sr = 48000.0;
        let mut hp = Biquad::new(BiquadType::HighPass, sr, 2000.0, 0.707, 0.0);

        let measure = |filter: &mut Biquad, freq: f32| -> f32 {
            filter.reset();
            let n = (sr / freq * 20.0).min(sr) as usize;
            let mut sum_sq = 0.0;
            let mut count = 0;
            for i in 0..n {
                let t = i as f32 / sr;
                let x = (2.0 * PI * freq * t).sin();
                let y = filter.process(x);
                if i > n / 2 {
                    sum_sq += y * y;
                    count += 1;
                }
            }
            (sum_sq / count as f32).sqrt()
        };

        let low_rms = measure(&mut hp, 50.0);
        let high_rms = measure(&mut hp, 8000.0);

        assert!(
            low_rms < high_rms * 0.2,
            "highpass should attenuate 50Hz much more than 8kHz: low={low_rms}, high={high_rms}"
        );
    }

    #[test]
    fn test_butterworth_cascade_no_nan() {
        let mut filter = IirFilter::butterworth(BiquadType::LowPass, 8, 44100.0, 1000.0);
        assert_eq!(filter.len(), 4);

        for i in 0..1000 {
            let x = ((i as f32) * 0.01).sin();
            let y = filter.process(x);
            assert!(y.is_finite(), "cascade output not finite at sample {i}");
        }
    }

    #[test]
    #[should_panic]
    fn test_butterworth_odd_order_panics() {
        let _ = IirFilter::butterworth(BiquadType::LowPass, 3, 44100.0, 1000.0);
    }
}
