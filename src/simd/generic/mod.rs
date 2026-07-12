//! SIMD abstraction layer and fast math functions.
//!
//! All numerical computation in this module is written generically over the
//! [`crate::simd::generic::SimdFloat`] trait. The default
//! [`crate::simd::generic::ScalarFloat`] backend (WIDTH=1) works everywhere;
//! platform backends are activated by [`crate::sim`]'s compile-time-selected
//! opt-in features, or unconditionally by [`crate::simd`]'s own
//! runtime-detected `simd` feature (see
//! [`crate::simd::generic::dispatch::detect_backend`]):
//!
//! | Feature       | Type       | WIDTH | Platform  |
//! |---------------|------------|-------|-----------|
//! | (none)        | `ScalarFloat` | 1  | all       |
//! | `simd`        | `F32x4Sse` | 4     | x86_64 (baseline floor)|
//! | `sim-avx2`    | `F32x8`    | 8     | x86_64    |
//! | `sim-avx512`  | `F32x16`   | 16    | x86_64    |
//! | `sim-neon`    | `F32x4`    | 4     | aarch64   |
//!
//! # Fast Math
//!
//! The following functions are provided with < 1e-5 relative error:
//!
//! - [`crate::simd::generic::simd_exp`] -- exponential via range reduction + degree-5 polynomial
//! - [`crate::simd::generic::simd_log`] -- natural log via atanh series expansion
//! - [`crate::simd::generic::simd_log1pexp`] -- numerically stable softplus
//! - [`crate::simd::generic::simd_sigmoid`] -- logistic sigmoid without overflow
//! - [`crate::simd::generic::simd_recip`], [`crate::simd::generic::simd_rsqrt`] -- Newton-refined reciprocal/inverse sqrt

// `exp`'s Cody-Waite range-reduction constants are intentionally given to
// more decimal digits than `f32` can represent exactly (for readability
// against reference values), hence this allowance for the whole module.
#![allow(clippy::excessive_precision)]

/// SIMD exponential (`simd_exp`).
pub mod exp;
/// SIMD natural log and softplus (`simd_log`, `simd_log1pexp`).
pub mod log;
/// SIMD reciprocal and inverse square root (`simd_recip`, `simd_rsqrt`).
pub mod reciprocal;
/// The always-available `WIDTH=1` fallback backend.
pub mod scalar;
/// SIMD logistic sigmoid (`simd_sigmoid`).
pub mod sigmoid;
/// The [`crate::simd::generic::SimdFloat`] abstraction trait itself.
pub mod traits;

/// AVX2 `F32x8` backend, `x86_64` only. Compiled whenever either
/// [`crate::sim`]'s compile-time-selected `sim-avx2` backend or
/// [`crate::simd`]'s runtime-dispatched `simd` feature needs it - the two
/// layers share this same type rather than each hand-rolling their own.
#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "sim-avx2")))]
pub mod avx2;

/// AVX-512 `F32x16` backend, `x86_64` only. Compiled whenever either
/// [`crate::sim`]'s compile-time-selected `sim-avx512` backend or
/// [`crate::simd`]'s runtime-dispatched `simd` feature needs it, mirroring
/// [`crate::simd::generic::avx2`]'s dual use.
#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "sim-avx512")))]
pub mod avx512;

/// ARM NEON `F32x4` backend, `aarch64` only. Compiled whenever either
/// [`crate::sim`]'s compile-time-selected `sim-neon` backend or
/// [`crate::simd`]'s runtime-dispatched `simd` feature needs it, mirroring
/// [`crate::simd::generic::avx2`]'s dual use.
#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "sim-neon")))]
pub mod neon;

/// Baseline SSE2 `F32x4Sse` backend, `x86_64` only - see
/// [`crate::simd::generic::sse2`]'s module docs for why it needs no opt-in
/// feature of its own.
#[cfg(target_arch = "x86_64")]
pub mod sse2;

/// Width-agnostic `dot`/`mul_elementwise`/`mix_scalar` shared by every
/// [`crate::simd::generic::SimdFloat`] backend - see
/// [`crate::simd::generic::ops`]'s module docs.
pub mod ops;

/// Runtime backend selection ([`crate::simd::generic::dispatch::detect_backend`]).
pub mod dispatch;

pub use dispatch::SimdBackend;
pub use scalar::ScalarFloat;
pub use traits::SimdFloat;

pub use exp::simd_exp;
pub use log::{simd_log, simd_log1pexp};
pub use reciprocal::{simd_recip, simd_rsqrt};
pub use sigmoid::simd_sigmoid;

/// Select the best available SIMD backend at compile time.
///
/// This is [`crate::sim`]'s own compile-time choice (which `sim-*` feature
/// was enabled), distinct from
/// [`crate::simd::generic::dispatch::detect_backend`]'s runtime CPU check -
/// see that function's docs for why the two differ. Returns a string
/// identifying the active backend, from the same vocabulary as
/// [`SimdBackend::as_str`].
pub fn active_backend() -> &'static str {
    #[cfg(all(target_arch = "x86_64", feature = "sim-avx512"))]
    {
        return SimdBackend::Avx512.as_str();
    }

    // `not(sim-avx512)` keeps this mutually exclusive with the branch above
    // so `--all-features` (which enables every `sim-*` feature at once)
    // doesn't produce two unconditional `return`s in a row.
    #[cfg(all(
        target_arch = "x86_64",
        feature = "sim-avx2",
        not(feature = "sim-avx512")
    ))]
    {
        return SimdBackend::Avx2.as_str();
    }

    #[cfg(all(target_arch = "aarch64", feature = "sim-neon"))]
    {
        return SimdBackend::Neon.as_str();
    }

    #[allow(unreachable_code)]
    SimdBackend::Scalar.as_str()
}
