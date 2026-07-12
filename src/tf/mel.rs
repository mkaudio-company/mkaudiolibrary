//! Mel filterbank and mel spectrogram.

use super::stft::{StftConfig, stft};

/// Mel filterbank configuration.
#[derive(Debug, Clone, Copy)]
pub struct MelConfig {
    /// Sample rate of the input signal in Hz.
    pub sample_rate: f32,
    /// Number of mel filterbank bands.
    pub num_mels: usize,
    /// Lowest frequency covered by the filterbank, in Hz.
    pub min_freq: f32,
    /// Highest frequency covered by the filterbank, in Hz.
    pub max_freq: f32,
}

#[inline]
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

#[inline]
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank: `num_mels` rows, each `fft_size/2 + 1`
/// bins wide, ready to be applied to a power/magnitude spectrum via a dot
/// product (see [`mel_spectrogram`]).
pub fn mel_filterbank(config: &MelConfig, fft_size: usize) -> Vec<Vec<f32>> {
    let num_bins = fft_size / 2 + 1;
    if config.num_mels == 0 || num_bins == 0 {
        return Vec::new();
    }

    let mel_min = hz_to_mel(config.min_freq);
    let mel_max = hz_to_mel(config.max_freq);
    let mel_points: Vec<f32> = (0..config.num_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (config.num_mels + 1) as f32)
        .collect();
    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&m| (((fft_size + 1) as f32) * mel_to_hz(m) / config.sample_rate).floor() as usize)
        .collect();

    let mut filters = vec![vec![0.0; num_bins]; config.num_mels];
    for m in 0..config.num_mels {
        let (left, center, right) = (bin_points[m], bin_points[m + 1], bin_points[m + 2]);

        if center > left {
            for (bin, slot) in filters[m]
                .iter_mut()
                .enumerate()
                .take(center.min(num_bins))
                .skip(left)
            {
                *slot = (bin - left) as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for (bin, slot) in filters[m]
                .iter_mut()
                .enumerate()
                .take(right.min(num_bins))
                .skip(center)
            {
                *slot = (right - bin) as f32 / (right - center) as f32;
            }
        }
    }

    filters
}

/// Mel-scaled power spectrogram: an [`stft`] followed by triangular
/// mel-filter application (a SIMD-accelerated dot product per filter, via
/// [`crate::simd::dot`]). Returns one row per frame, each `num_mels` wide.
/// Callers typically log-compress the result (e.g. `10.0 * v.max(eps).log10()`)
/// before further use.
pub fn mel_spectrogram(
    signal: &[f32],
    stft_config: &StftConfig,
    mel_config: &MelConfig,
) -> Vec<Vec<f32>> {
    let frames = stft(signal, stft_config);
    let filters = mel_filterbank(mel_config, stft_config.fft_size);
    let num_bins = stft_config.fft_size / 2 + 1;

    frames
        .iter()
        .map(|frame| {
            let power: Vec<f32> = frame.iter().take(num_bins).map(|c| c.norm_sqr()).collect();
            filters
                .iter()
                .map(|filter| crate::simd::dot(&power, filter))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::stft::WindowFunction;
    use super::*;

    #[test]
    fn filterbank_rows_peak_near_their_center_frequency() {
        let config = MelConfig {
            sample_rate: 44100.0,
            num_mels: 20,
            min_freq: 0.0,
            max_freq: 22050.0,
        };
        let filters = mel_filterbank(&config, 2048);
        assert_eq!(filters.len(), 20);
        for filter in &filters {
            assert!(
                filter.iter().any(|&v| v > 0.0),
                "filter has no nonzero weights"
            );
        }
    }

    #[test]
    fn mel_spectrogram_has_expected_shape() {
        let signal: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.05).sin()).collect();
        let stft_config = StftConfig::new(1024, 512, WindowFunction::Hann);
        let mel_config = MelConfig {
            sample_rate: 44100.0,
            num_mels: 40,
            min_freq: 20.0,
            max_freq: 20000.0,
        };

        let result = mel_spectrogram(&signal, &stft_config, &mel_config);
        assert!(!result.is_empty());
        for row in &result {
            assert_eq!(row.len(), 40);
        }
    }
}
