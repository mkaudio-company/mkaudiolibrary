//! # mkaudiolibrary
//!
//! A Rust library for real-time audio signal processing, featuring analog modeling
//! through numeric functions and circuit simulation via Modified Nodal Analysis (MNA).
//!
//! Every audio-sample-carrying API in this crate - [`dsp`], [`processor`],
//! [`audiofile`], [`host`], [`realtime`] - operates on plain `f32` (matching
//! VST3/AU/MKAP plugin hosting's native sample format, and [`sim`]'s own
//! `f32` circuit models), passed as plain `&[f32]`/`&mut [f32]` slices or
//! the unlocked [`buffer::Buffer<f32>`] wrapper. None of this crate's own
//! processors share buffers across threads internally, so nothing here
//! pays for locking that has no reader on the other end; if you do need to
//! hand a buffer to another thread (e.g. an audio thread feeding a UI
//! meter), wrap it yourself with whatever synchronization actually fits.
//!
//! ## Features
//!
//! - **Analog modeling** - asymmetric log-curve saturation, and (via the
//!   `sim` feature) physically-modeled vacuum tube saturation
//! - **Circuit simulation** - real-time MNA solver for reactive circuits
//!   ([`dsp::Circuit`]), plus a full tube/diode/transistor + Wave Digital
//!   Filter circuit modeling toolkit (see [`sim`], `sim` feature)
//! - **DSP primitives** - convolution, IIR (biquad/Butterworth) and FIR
//!   (windowed-sinc) filtering, compression/limiting/gating, delay, integer
//!   oversampling, and FFT-based sample-rate conversion - all with
//!   pre-allocated scratch buffers so steady-state processing never allocates
//! - **SIMD** - AVX2+FMA/SSE2 (`x86_64`) or NEON (`aarch64`) hot loops (optional `simd` feature)
//! - **Time-frequency analysis** - DFT, FFT, DCT, STFT, CWT, CQT, mel spectrograms (see [`tf`])
//! - **Audio file I/O** - WAV, BWF, and AIFF format support with Buffer integration
//! - **Plugin hosting** - load and run MKAP, VST3, and AUv2 (macOS) plugins through one trait (see [`host`])
//! - **MKAP plugin system** - native format for building your own modular processing chains
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
//! // Convert to buffers for processing
//! let mut buffers = audio.to_buffers();
//!
//! // Apply compression
//! let mut comp = Compression::new(audio.sample_rate());
//! comp.threshold = -12.0;
//! comp.ratio = 4.0;
//! for buffer in &mut buffers {
//!     let mut output = mkaudiolibrary::buffer::Buffer::new(buffer.len());
//!     comp.run(buffer, &mut output);
//!     // ... use processed output
//! }
//!
//! // Save result
//! audio.save("output.wav", FileFormat::Wav);
//! ```
//!
//! ## Modules
//!
//! - [`buffer`] - plain (unlocked) audio sample containers (`Buffer`, `PushBuffer`, `CircularBuffer`)
//! - [`dsp`] - digital signal processing components
//! - [`simd`] - SIMD-accelerated primitives used by `dsp` and `tf`'s hot loops
//! - [`tf`] - time-frequency analysis (DFT, FFT, DCT, STFT, CWT, CQT, mel spectrograms)
//! - [`sim`] - analog circuit simulation: tubes, diodes, transistors, WDF networks (`sim` feature)
//! - [`audiofile`] - WAV/BWF/AIFF file loading and saving
//! - [`processor`] - MKAP plugin format and dynamic loading
//! - [`host`] - unified plugin hosting for MKAP, VST3 (`vst3` feature), and AUv2 (`au` feature)
//! - [`realtime`] - real-time audio streaming I/O (requires `realtime` feature)
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
//! let mut output = Buffer::new(5);
//! sat.run(&input, &mut output);
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
//! use mkaudiolibrary::dsp::{Compression, Limit, Gate};
//!
//! let mut compressor = Compression::new(44100.0);
//! compressor.threshold = -20.0;  // dB
//! compressor.ratio = 4.0;        // 4:1
//!
//! let mut limiter = Limit::new(44100.0);
//! limiter.ceiling = -0.1;        // dB
//!
//! let mut gate = Gate::new(44100.0);
//! gate.threshold = -40.0;        // dB
//! ```
//!
//! ### IIR/FIR Filtering
//!
//! ```ignore
//! use mkaudiolibrary::dsp::iir::{Biquad, BiquadType};
//! use mkaudiolibrary::dsp::fir::FirFilter;
//!
//! let mut lowpass = Biquad::new(BiquadType::LowPass, 44100.0, 1000.0, 0.707, 0.0);
//! let y = lowpass.process(0.5);
//!
//! let mut fir_lp = FirFilter::lowpass(101, 44100.0, 1000.0);
//! let y2 = fir_lp.process(0.5);
//! ```
//!
//! ## License
//!
//! MIT License.

/// Plain (unlocked) audio sample containers for real-time processing.
///
/// Provides `Buffer`, `PushBuffer`, and `CircularBuffer` types with no
/// internal locking - see the module docs for why.
pub mod buffer;

/// Digital signal processing components for real-time audio.
///
/// Includes convolution, IIR/FIR filtering, saturation, circuit simulation,
/// compression/limiting/gating, delay, oversampling, and resampling.
pub mod dsp;

/// MKAU plugin format for modular audio processing chains.
///
/// Provides the `Processor` trait and dynamic plugin loading.
pub mod processor;

/// Audio file loading and saving for WAV and AIFF formats.
///
/// Supports 8/16/24/32-bit audio with normalized f32 sample representation.
pub mod audiofile;

/// Real-time audio streaming I/O inspired by RTAudio.
///
/// Provides cross-platform audio input/output with a callback-based API.
/// Enable with the `realtime` feature flag.
#[cfg(feature = "realtime")]
pub mod realtime;

#[cfg(all(target_os = "macos", any(feature = "realtime", feature = "au")))]
mod macos_util;

/// SIMD-accelerated primitives for hot per-sample DSP/TF loops.
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

/// Real-time analog circuit simulation (tubes, diodes, transistors, WDF
/// networks), merged in from [libmksim](https://github.com/mkaudio-company/libmksim).
///
/// Enable with the `sim` feature flag; `sim-avx2`/`sim-avx512`/`sim-neon`
/// additionally enable SIMD backends for its internal math.
#[cfg(feature = "sim")]
pub mod sim;
