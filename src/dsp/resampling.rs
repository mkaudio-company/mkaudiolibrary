//! Whole-buffer sample-rate conversion via frequency-domain zero-padding/
//! truncation, built on [`crate::tf::fft`].
//!
//! This is the "FFT resample" technique (as used by e.g. `scipy.signal.resample`):
//! take the real FFT of the input, truncate or zero-pad the spectrum to the
//! bin count implied by the target length, rescale for the length change,
//! and inverse-transform. Downsampling truncates away the content above the
//! new Nyquist frequency, which is simultaneously the resampling and the
//! anti-aliasing step. This assumes the input is one full period of a
//! periodic (or transient-free, e.g. windowed) signal -- like any FFT-based
//! method it will ring at hard discontinuities at the buffer edges -- so it
//! suits offline/analysis resampling of whole buffers rather than
//! per-sample real-time streaming. For real-time-safe streaming rate
//! conversion, build a zero-stuff/[`super::fir`]-lowpass/decimate chain
//! directly (see [`super::Oversampler`] for the integer-factor case).
//!
use crate::tf::complex::Complex64;
use crate::tf::fft::{irfft, rfft};

/// Resample `input` from `in_rate` Hz to `out_rate` Hz, returning a new
/// buffer of `round(input.len() * out_rate / in_rate)` samples.
///
/// Returns an empty vector if `input` is empty or either rate is non-positive.
pub fn resample(input: &[f32], in_rate: f32, out_rate: f32) -> Vec<f32> {
    if input.is_empty() || in_rate <= 0.0 || out_rate <= 0.0 {
        return Vec::new();
    }
    if (in_rate - out_rate).abs() < 1e-6 {
        return input.to_vec();
    }

    let n_in = input.len();
    let n_out = ((n_in as f32) * out_rate / in_rate).round() as usize;
    if n_out == 0 {
        return Vec::new();
    }

    let spectrum = rfft(input);
    let out_half = n_out / 2 + 1;
    let copy_len = out_half.min(spectrum.len());

    // Forward FFT here is unnormalized and `irfft` divides by `n_out`
    // internally, so rescale by `n_out / n_in` to preserve amplitude
    // across the length change (matches scipy.signal.resample's convention).
    let scale = n_out as f32 / n_in as f32;
    let mut out_spectrum = vec![Complex64::new(0.0, 0.0); out_half];
    for (dst, src) in out_spectrum[..copy_len]
        .iter_mut()
        .zip(spectrum[..copy_len].iter())
    {
        *dst = src.scale(scale);
    }

    irfft(&out_spectrum, n_out)
}

/// [`resample`], writing into a caller-provided output slice.
///
/// `output` must already be sized for the resampled length (see
/// [`resample_output_len`]); only `min(output.len(), resampled.len())`
/// samples are copied.
pub fn resample_into(input: &[f32], in_rate: f32, output: &mut [f32], out_rate: f32) {
    let resampled = resample(input, in_rate, out_rate);
    let len = resampled.len().min(output.len());
    output[..len].copy_from_slice(&resampled[..len]);
}

/// The output length [`resample`] will produce for a given input length and
/// rate pair, useful for sizing a destination buffer up front.
pub fn resample_output_len(input_len: usize, in_rate: f32, out_rate: f32) -> usize {
    if input_len == 0 || in_rate <= 0.0 || out_rate <= 0.0 {
        return 0;
    }
    ((input_len as f32) * out_rate / in_rate).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_resample_identity() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).sin()).collect();
        let output = resample(&input, 44100.0, 44100.0);
        assert_eq!(output, input);
    }

    #[test]
    fn test_resample_output_length() {
        let input = vec![0.0; 1000];
        let output = resample(&input, 44100.0, 48000.0);
        let expected_len = resample_output_len(1000, 44100.0, 48000.0);
        assert_eq!(output.len(), expected_len);
    }

    #[test]
    fn test_resample_preserves_tone_frequency() {
        let sr_in = 8000.0;
        let sr_out = 16000.0;
        let freq = 200.0;
        let n_in = 800; // exactly 20 cycles, avoids edge discontinuity ringing
        let input: Vec<f32> = (0..n_in)
            .map(|i| (2.0 * PI * freq * i as f32 / sr_in).sin())
            .collect();

        let output = resample(&input, sr_in, sr_out);
        assert_eq!(output.len(), 1600);

        // Count zero crossings as a coarse frequency check: a 200Hz tone
        // over 100ms (1600 samples @ 16kHz) should cross zero ~40 times.
        let crossings = output.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(
            (36..=44).contains(&crossings),
            "expected ~40 zero crossings for a 200Hz tone, got {crossings}"
        );
    }
}
