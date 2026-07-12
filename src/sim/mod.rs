//! Real-time SIMD-optimized analog circuit simulation, merged in from
//! [libmksim](https://github.com/mkaudio-company/libmksim).
//!
//! `sim` models vacuum tubes, RLC circuits, op-amps, diodes, transistors,
//! potentiometers, switches, and passive filters using Wave Digital Filters
//! for linear networks and local Newton-Raphson solvers for nonlinear devices.
//!
//! [`dsp::TubeSaturation`](crate::dsp::TubeSaturation) builds on this module
//! to provide a physically-modeled alternative to [`dsp::Saturation`](crate::dsp::Saturation).
//!
//! ## Quick Start
//!
//! This module exposes individual circuit components and the WDF/solver
//! infrastructure they're built from -- there is no prebuilt "amp engine"
//! facade; compose the pieces you need directly:
//!
//! ```ignore
//! use mkaudiolibrary::sim::components::tubes::{TriodeStage, PARAMS_12AX7};
//! use mkaudiolibrary::sim::components::CircuitComponent;
//!
//! let mut triode = TriodeStage::new(PARAMS_12AX7.clone());
//! triode.prepare(44100.0);
//!
//! let input = [0.5f32; 64];
//! let mut output = [0.0f32; 64];
//! triode.process_block(&input, &mut output);
//! ```
//!
//! ## Architecture
//!
//! This module separates analog circuits into two domains:
//!
//! - **Linear passive networks** are modeled with [`wdf`] (Wave Digital Filters) --
//!   series/parallel adaptor trees that evaluate in O(1) per component with
//!   guaranteed passive stability.
//!
//! - **Nonlinear devices** (tubes, diodes, transistors) use local
//!   Newton-Raphson solvers ([`core::solver::NewtonSolver`]), typically
//!   converging in 2-4 iterations.
//!
//! All math is written generically over the [`crate::simd::generic::SimdFloat`]
//! trait, enabling transparent SIMD acceleration on AVX2, AVX-512, and ARM
//! NEON (via the `sim-avx2`, `sim-avx512`, and `sim-neon` features
//! respectively). That trait and its backends live under [`crate::simd`]
//! alongside the crate's other (`f32`) SIMD code, not under this module --
//! see [`crate::simd`]'s docs for why they're independent of each other.
//! Likewise, the `f32` parameter-smoothing type these components use for
//! click-free control changes lives at [`crate::dsp::parameter::Parameter`].
//!
//! ## Modules
//!
//! | Module          | Purpose                                            |
//! |-----------------|----------------------------------------------------|
//! | [`core`]        | Newton solver, circuit graph, buffer management    |
//! | [`components`]  | All circuit component models                       |
//! | [`wdf`]         | Wave Digital Filter framework                      |
//!
//! ## Differences from upstream libmksim
//!
//! This is a source merge, not a vendored dependency: internal `crate::`
//! paths were rewritten to `crate::sim::` so the module lives inside
//! `mkaudiolibrary` rather than as a standalone crate. The `api::ffi` C ABI
//! layer was dropped since `mkaudiolibrary` is rlib-only and exposes its own
//! plugin surface via [`crate::processor`] and [`crate::host`]. The prebuilt
//! `stages`/`api::processor` amp-engine facade (`PreampStage`, `ToneStack`,
//! `PowerAmpStage`, `TubeDspEngine`) was dropped too -- this module now
//! covers only the reusable circuit-modeling primitives; assembling them
//! into a full instrument/amp signal chain is left to the caller (or to
//! [`crate::dsp::TubeSaturation`] for the single-stage case). Its former
//! `simd` and `dsp` submodules (the `SimdFloat` trait system and the
//! `Parameter` smoothing type) were merged into the crate's top-level
//! [`crate::simd`] and [`crate::dsp`] modules respectively, since both are
//! independently useful outside of circuit simulation and there's no
//! reason for them to live nested under `sim`. Its `f32`
//! `DcBlocker`/`Oversampler`/`AntiAliasFilter` utilities were dropped
//! entirely along with the amp-stage facade, since nothing in `sim` used
//! them once that was gone - use [`crate::dsp::iir::Biquad`],
//! [`crate::dsp::Oversampler`], and [`crate::dsp::fir`] instead for that
//! functionality in `f32`.

#![allow(clippy::excessive_precision)]

pub mod components;
pub mod core;
pub mod wdf;
