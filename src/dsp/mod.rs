//! Digital signal processing components for real-time audio.
//!
//! This module provides a collection of DSP primitives including:
//!
//! - **Utility functions** - dB/ratio conversion
//! - **Convolution** ([`Convolution`]) - FIR filtering with an arbitrary impulse response
//! - **Filtering** ([`iir`], [`fir`]) - biquad/Butterworth IIR cascades and
//!   windowed-sinc FIR design
//! - **Saturation** ([`Saturation`], [`TubeSaturation`]) - a cheap asymmetric
//!   log-curve waveshaper, and a physically-modeled vacuum tube alternative
//! - **Circuit simulation** ([`Circuit`]) - real-time transient analysis using
//!   Modified Nodal Analysis (MNA)
//! - **Dynamics** ([`Compression`], [`Limit`], [`Gate`]) - compression,
//!   limiting, and noise gating with envelope detection
//! - **Time-based** ([`Delay`]) - delay with feedback and wet/dry mix
//! - **Sample-rate conversion** ([`Oversampler`], [`resampling`]) - integer
//!   oversampling around nonlinear stages, and whole-buffer FFT resampling
//! - **Parameter smoothing** ([`parameter::Parameter`]) - one-pole smoothed
//!   `f32` control values, used internally by [`crate::sim`]'s components
//!   and available for any other `f32`-domain smoothing needs
//!
//! All processors operate on plain `f32` slices (`&[f32]` in, `&mut [f32]`
//! out) - the same sample format VST3/AU/MKAP plugin hosting uses - and are
//! designed for real-time audio processing: each processor owns its state
//! outright with no internal locking, so use `&mut Processor` per
//! audio-processing thread rather than sharing one instance across threads.
//! Hot per-sample-block loops (envelope-to-gain application, convolution/FIR
//! dot products) go through [`crate::simd`], picking up AVX2+FMA/SSE2
//! (`x86_64`) or NEON (`aarch64`) acceleration when the `simd` feature is
//! enabled, with a scalar fallback otherwise.
//!
//! ## Example: Processing with Saturation
//!
//! ```ignore
//! use mkaudiolibrary::dsp::Saturation;
//!
//! let sat = Saturation::new(10.0, 10.0, 1.0, 1.0, 0.0, false);
//! let input = [0.0, 0.5, 1.0, -0.5, -1.0];
//! let mut output = [0.0; 5];
//!
//! sat.run(&input, &mut output);
//! ```
//!
//! ## Example: Circuit Simulation
//!
//! ```ignore
//! use mkaudiolibrary::dsp::{Circuit, Resistor, Capacitor};
//!
//! // Create a simple RC lowpass filter
//! let mut circuit = Circuit::new(44100.0, 2);
//! circuit.add_component(Box::new(Resistor::new(1, 2, 1000.0)));   // 1kΩ
//! circuit.add_component(Box::new(Capacitor::new(2, 0, 1e-6)));    // 1µF
//! circuit.preprocess(10.0);
//!
//! let output = circuit.process(1.0, 2);  // Input 1V, probe node 2
//! ```
//!
//! ## Example: IIR/FIR filtering
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

mod circuit;
mod convolution;
mod delay;
mod dynamics;
pub mod fir;
pub mod iir;
mod oversampling;
/// One-pole smoothed `f32` parameter ([`parameter::Parameter`]), used
/// internally by [`crate::sim`]'s components.
pub mod parameter;
pub mod resampling;
mod saturation;

pub use circuit::{Capacitor, Circuit, Component, Inductor, Resistor};
pub use convolution::Convolution;
pub use delay::Delay;
pub use dynamics::{Compression, Gate, Limit};
pub use oversampling::Oversampler;
pub use resampling::resample;
pub use saturation::Saturation;

#[cfg(feature = "sim")]
pub use saturation::TubeSaturation;

// ==========================================
// Utility Functions
// ==========================================

/// Convert a linear ratio to decibels.
///
/// Formula: `dB = 20 * log10(ratio)`
#[inline]
pub fn ratio_to_db(ratio: f32) -> f32 {
    20.0 * ratio.log10()
}

/// Convert decibels to a linear ratio.
///
/// Formula: `ratio = 10^(dB / 20)`
#[inline]
pub fn db_to_ratio(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}
