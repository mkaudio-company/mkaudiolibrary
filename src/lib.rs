//! # mkaudiolibrary
//!
//! A Rust library for real-time audio signal processing, featuring analog modeling
//! through numeric functions and circuit simulation via Modified Nodal Analysis (MNA).
//!
//! ## Features
//!
//! - **Thread-safe buffers** - Concurrent access with `RwLock`-based locking
//! - **Analog modeling** - Asymmetric saturation for tube/tape-style harmonics
//! - **Circuit simulation** - Real-time MNA solver for reactive circuits
//! - **DSP primitives** - Convolution, compression, limiting, and delay, with
//!   pre-allocated scratch buffers so steady-state processing never allocates
//! - **SIMD** - AVX2+FMA/SSE2 (`x86_64`) or NEON (`aarch64`) hot loops (optional `simd` feature)
//! - **Time-frequency analysis** - DFT, FFT, DCT, STFT, CWT, CQT, mel spectrograms (see [`tf`])
//! - **Audio file I/O** - WAV, BWF, and AIFF format support with Buffer integration
//! - **Plugin hosting** - Load and run MKAP, VST3, and AUv2 (macOS) plugins through one trait (see [`host`])
//! - **MKAP plugin system** - Native format for building your own modular processing chains
//! - **Real-time streaming** - RTAudio-style API with real CoreAudio/WASAPI/ALSA backends (optional `realtime` feature)
//!
//! ## Quick Start
//!
//! ```ignore
//! use mkaudiolibrary::audiofile::{AudioFile, FileFormat};
//! use mkaudiolibrary::dsp::Compression;
//!
//! // Load an audio file
//! let mut audio = AudioFile::default();
//! audio.load("input.wav");
//!
//! // Convert to thread-safe buffers for processing
//! let buffers = audio.to_buffers();
//!
//! // Apply compression
//! let mut comp = Compression::new(audio.sample_rate() as f64);
//! comp.threshold = -12.0;
//! comp.ratio = 4.0;
//! for buffer in &buffers {
//!     let output = mkaudiolibrary::buffer::Buffer::new(buffer.len());
//!     comp.run(buffer, &output);
//!     // ... use processed output
//! }
//!
//! // Save result
//! audio.save("output.wav", FileFormat::Wav);
//! ```
//!
//! ## Modules
//!
//! - [`buffer`] - Thread-safe audio buffers (`Buffer`, `PushBuffer`, `CircularBuffer`)
//! - [`dsp`] - Digital signal processing components
//! - [`simd`] - SIMD-accelerated primitives used by `dsp` and `tf`'s hot loops
//! - [`tf`] - Time-frequency analysis (DFT, FFT, DCT, STFT, CWT, CQT, mel spectrograms)
//! - [`audiofile`] - WAV/BWF/AIFF file loading and saving
//! - [`processor`] - MKAP plugin format and dynamic loading
//! - [`host`] - Unified plugin hosting for MKAP, VST3 (`vst3` feature), and AUv2 (`au` feature)
//! - [`realtime`] - Real-time audio streaming I/O (requires `realtime` feature)
//!
//! ## Thread Safety
//!
//! All buffer types use `Arc<RwLock<...>>` internally, enabling:
//! - Multiple concurrent readers
//! - Exclusive writer access
//! - Safe sharing across threads via `Clone`
//!
//! ```ignore
//! use mkaudiolibrary::buffer::Buffer;
//! use std::thread;
//!
//! let buffer = Buffer::<f64>::new(1024);
//! let buffer_clone = buffer.clone();  // Shares underlying data
//!
//! thread::spawn(move || {
//!     let mut guard = buffer_clone.write();
//!     guard[0] = 1.0;
//! });
//! ```
//!
//! ## DSP Processing Examples
//!
//! ### Saturation (Analog Modeling)
//!
//! ```ignore
//! use mkaudiolibrary::dsp::Saturation;
//! use mkaudiolibrary::buffer::Buffer;
//!
//! let sat = Saturation::new(10.0, 10.0, 1.0, 1.0, 0.0, false);
//! let input = Buffer::from_slice(&[0.0, 0.5, 1.0, -0.5, -1.0]);
//! let output = Buffer::new(5);
//! sat.run(&input, &output);
//! ```
//!
//! ### Circuit Simulation
//!
//! ```ignore
//! use mkaudiolibrary::dsp::{Circuit, Resistor, Capacitor};
//!
//! // RC lowpass filter: R=1kΩ, C=1µF, fc ≈ 159Hz
//! let mut circuit = Circuit::new(44100.0, 2);
//! circuit.add_component(Box::new(Resistor::new(1, 2, 1000.0)));
//! circuit.add_component(Box::new(Capacitor::new(2, 0, 1e-6)));
//! circuit.preprocess(10.0);
//!
//! let output = circuit.process(1.0, 2);  // Input 1V, probe node 2
//! ```
//!
//! ### Dynamics Processing
//!
//! ```ignore
//! use mkaudiolibrary::dsp::{Compression, Limit};
//!
//! let mut compressor = Compression::new(44100.0);
//! compressor.threshold = -20.0;  // dB
//! compressor.ratio = 4.0;        // 4:1
//!
//! let mut limiter = Limit::new(44100.0);
//! limiter.ceiling = -0.1;        // dB
//! ```
//!
//! ## License
//!
//! This library is dual-licensed:
//!
//! - **GPL-3.0** for open source projects
//! - **Commercial license** available for closed source usage
//!
//! For commercial licensing inquiries, contact: minjaekim@mkaudio.company

/// Thread-safe audio buffers for real-time concurrent processing.
///
/// Provides `Buffer`, `PushBuffer`, and `CircularBuffer` types with
/// `RwLock`-based locking for safe multi-threaded access.
pub mod buffer;

/// Digital signal processing components for real-time audio.
///
/// Includes convolution, saturation, circuit simulation, compression,
/// limiting, and delay effects.
pub mod dsp;

/// MKAU plugin format for modular audio processing chains.
///
/// Provides the `Processor` trait and dynamic plugin loading.
pub mod processor;

/// Audio file loading and saving for WAV and AIFF formats.
///
/// Supports 8/16/24/32-bit audio with normalized f64 sample representation.
pub mod audiofile;

/// Real-time audio streaming I/O inspired by RTAudio.
///
/// Provides cross-platform audio input/output with a callback-based API.
/// Enable with the `realtime` feature flag.
#[cfg(feature = "realtime")]
pub mod realtime;

#[cfg(all(target_os = "macos", any(feature = "realtime", feature = "au")))]
mod macos_util;

/// SIMD-accelerated primitives for hot per-sample DSP loops.
///
/// Falls back to scalar loops unless the `simd` feature is enabled.
pub mod simd;

/// Plugin hosting for MKAP, VST3, and AUv2 formats.
///
/// Provides a unified `HostedPlugin` trait and scanning API on top of the
/// native MKAP loader (always available), VST3 hosting (`vst3` feature),
/// and AUv2 hosting on macOS (`au` feature).
pub mod host;

/// Time-frequency analysis: DFT, FFT, DCT, STFT/multi-resolution STFT,
/// CWT, CQT, and mel spectrograms.
pub mod tf;
