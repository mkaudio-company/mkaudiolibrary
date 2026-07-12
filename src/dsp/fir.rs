//! FIR filtering: windowed-sinc coefficient design plus a streaming applicator.

use std::f32::consts::PI;

use crate::dsp::Convolution;

/// Blackman window value for tap `n` of `order` total taps.
///
/// Chosen over Hamming/Hann for its lower sidelobes (~-58dB vs ~-43dB for
/// Hamming), at the cost of a slightly wider transition band -- a good
/// default for audio EQ/crossover use where stopband leakage tends to be
/// more audible than a marginally softer knee.
#[inline]
fn blackman(n: usize, order: usize) -> f32 {
    if order <= 1 {
        return 1.0;
    }
    let denom = (order - 1) as f32;
    0.42 - 0.5 * (2.0 * PI * n as f32 / denom).cos() + 0.08 * (4.0 * PI * n as f32 / denom).cos()
}

/// Design a windowed-sinc lowpass kernel.
///
/// `order` is the number of taps (should be odd for a symmetric, exactly
/// linear-phase kernel). `cutoff` and `sample_rate` are in Hz. The kernel
/// is normalized to unity DC gain.
pub fn design_lowpass(order: usize, sample_rate: f32, cutoff: f32) -> Vec<f32> {
    assert!(order > 0, "FIR order must be positive");
    let fc = (cutoff / sample_rate).clamp(1e-6, 0.5);
    let half = (order - 1) as f32 / 2.0;

    let mut coeffs = vec![0.0f32; order];
    for (n, c) in coeffs.iter_mut().enumerate() {
        let x = n as f32 - half;
        let sinc = if x.abs() < 1e-6 {
            2.0 * fc
        } else {
            (2.0 * PI * fc * x).sin() / (PI * x)
        };
        *c = sinc * blackman(n, order);
    }

    let sum: f32 = coeffs.iter().sum();
    if sum.abs() > 1e-9 {
        for c in &mut coeffs {
            *c /= sum;
        }
    }
    coeffs
}

/// Design a windowed-sinc highpass kernel via spectral inversion of a
/// lowpass kernel of the same order.
///
/// `order` must be odd (spectral inversion needs a single center tap).
pub fn design_highpass(order: usize, sample_rate: f32, cutoff: f32) -> Vec<f32> {
    assert!(
        order % 2 == 1,
        "FIR highpass order must be odd, got {order}"
    );
    let mut coeffs = design_lowpass(order, sample_rate, cutoff);
    for c in &mut coeffs {
        *c = -*c;
    }
    coeffs[order / 2] += 1.0;
    coeffs
}

/// Design a windowed-sinc bandpass kernel as the difference of two lowpass
/// kernels (`design_lowpass(high) - design_lowpass(low)`).
///
/// `order` must be odd. `low` and `high` are the passband edges in Hz.
pub fn design_bandpass(order: usize, sample_rate: f32, low: f32, high: f32) -> Vec<f32> {
    assert!(
        order % 2 == 1,
        "FIR bandpass order must be odd, got {order}"
    );
    assert!(low < high, "bandpass low edge must be below high edge");
    let lp_high = design_lowpass(order, sample_rate, high);
    let lp_low = design_lowpass(order, sample_rate, low);
    lp_high
        .iter()
        .zip(lp_low.iter())
        .map(|(h, l)| h - l)
        .collect()
}

/// Design a windowed-sinc band-stop (notch) kernel via spectral inversion
/// of a bandpass kernel of the same order.
///
/// `order` must be odd.
pub fn design_bandstop(order: usize, sample_rate: f32, low: f32, high: f32) -> Vec<f32> {
    assert!(
        order % 2 == 1,
        "FIR bandstop order must be odd, got {order}"
    );
    let mut coeffs = design_bandpass(order, sample_rate, low, high);
    for c in &mut coeffs {
        *c = -*c;
    }
    coeffs[order / 2] += 1.0;
    coeffs
}

/// Streaming FIR filter with a windowed-sinc design front-end.
///
/// Internally this is a thin wrapper over [`Convolution`] (so it inherits
/// the same SIMD-accelerated dot-product hot loop), with named constructors
/// for the common windowed-sinc filter shapes.
pub struct FirFilter {
    conv: Convolution,
}

impl FirFilter {
    /// Wrap an arbitrary FIR kernel (e.g. from [`design_lowpass`] or a
    /// custom-designed impulse response).
    pub fn new(coeffs: &[f32]) -> Self {
        Self {
            conv: Convolution::new(coeffs),
        }
    }

    /// Design-and-build a lowpass FIR filter. See [`design_lowpass`].
    pub fn lowpass(order: usize, sample_rate: f32, cutoff: f32) -> Self {
        Self::new(&design_lowpass(order, sample_rate, cutoff))
    }

    /// Design-and-build a highpass FIR filter. See [`design_highpass`].
    pub fn highpass(order: usize, sample_rate: f32, cutoff: f32) -> Self {
        Self::new(&design_highpass(order, sample_rate, cutoff))
    }

    /// Design-and-build a bandpass FIR filter. See [`design_bandpass`].
    pub fn bandpass(order: usize, sample_rate: f32, low: f32, high: f32) -> Self {
        Self::new(&design_bandpass(order, sample_rate, low, high))
    }

    /// Design-and-build a band-stop FIR filter. See [`design_bandstop`].
    pub fn bandstop(order: usize, sample_rate: f32, low: f32, high: f32) -> Self {
        Self::new(&design_bandstop(order, sample_rate, low, high))
    }

    /// Number of taps in the filter kernel.
    pub fn order(&self) -> usize {
        self.conv.kernel_len()
    }

    /// Process a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        self.conv.process(input)
    }

    /// Process a buffer of samples.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        self.conv.run(input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_response(filter: &mut FirFilter, sample_rate: f32, freq: f32) -> f32 {
        let n = ((sample_rate / freq) as usize * 20).max(filter.order() * 4);
        let mut sum_sq = 0.0;
        let mut count = 0;
        for i in 0..n {
            let t = i as f32 / sample_rate;
            let x = (2.0 * PI * freq * t).sin();
            let y = filter.process(x);
            if i > n / 2 {
                sum_sq += y * y;
                count += 1;
            }
        }
        (sum_sq / count as f32).sqrt()
    }

    #[test]
    fn test_lowpass_attenuates_high_frequency() {
        let sr = 48000.0;
        let mut filter = FirFilter::lowpass(101, sr, 1000.0);

        let low_rms = rms_response(&mut filter, sr, 100.0);
        let high_rms = rms_response(&mut filter, sr, 10000.0);

        assert!(
            high_rms < low_rms * 0.1,
            "FIR lowpass should attenuate 10kHz much more than 100Hz: low={low_rms}, high={high_rms}"
        );
    }

    #[test]
    fn test_highpass_attenuates_low_frequency() {
        let sr = 48000.0;
        let mut filter = FirFilter::highpass(101, sr, 4000.0);

        let low_rms = rms_response(&mut filter, sr, 100.0);
        let high_rms = rms_response(&mut filter, sr, 12000.0);

        assert!(
            low_rms < high_rms * 0.1,
            "FIR highpass should attenuate 100Hz much more than 12kHz: low={low_rms}, high={high_rms}"
        );
    }

    #[test]
    fn test_bandpass_passes_only_band() {
        let sr = 48000.0;
        let mut filter = FirFilter::bandpass(151, sr, 1000.0, 3000.0);

        let in_band = rms_response(&mut filter, sr, 2000.0);
        let below = rms_response(&mut filter, sr, 200.0);
        let above = rms_response(&mut filter, sr, 15000.0);

        assert!(
            in_band > below * 5.0,
            "in-band should pass more than sub-band"
        );
        assert!(
            in_band > above * 5.0,
            "in-band should pass more than super-band"
        );
    }
}
