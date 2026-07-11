//! Time-frequency analysis: DFT, FFT, DCT, STFT (and multi-resolution
//! STFT), CWT, CQT, and mel spectrograms.
//!
//! Hot inner loops (per-bin/per-filter dot products) go through
//! [`crate::simd::dot`], so they get the same AVX2/FMA/SSE2 (`x86_64`) or
//! NEON (`aarch64`) acceleration as the rest of the crate when the `simd`
//! feature is enabled, with a scalar fallback otherwise.
//!
//! For Cohen's class of *bilinear* time-frequency distributions (Wigner-Ville,
//! Choi-Williams, Rihaczek, cone-shape, ...) see the sibling `bilinear_tf`
//! crate instead - this module covers the standard *linear* transforms.
//!
//! ```ignore
//! use mkaudiolibrary::tf::{fft, stft, mel, complex::Complex64};
//!
//! // FFT of a real signal (any length, not just powers of two)
//! let spectrum = fft::rfft(&signal);
//!
//! // Mel spectrogram
//! let stft_config = stft::StftConfig::new(1024, 256, stft::WindowFunction::Hann);
//! let mel_config = mel::MelConfig { sample_rate: 44100.0, num_mels: 80, min_freq: 20.0, max_freq: 20000.0 };
//! let frames = mel::mel_spectrogram(&signal, &stft_config, &mel_config);
//! ```

pub mod complex;
pub mod cqt;
pub mod cwt;
pub mod dct;
pub mod fft;
pub mod mel;
pub mod stft;

pub use complex::Complex64;
