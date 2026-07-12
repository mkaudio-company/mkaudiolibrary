//! SIMD-accelerated primitives for hot DSP/TF loops.
//!
//! Enabled via the `simd` Cargo feature. On `x86_64` this dispatches at
//! runtime to AVX-512, then AVX2+FMA, then baseline SSE2 (guaranteed
//! present on every x86_64 target), whichever the CPU actually supports.
//! On `aarch64` this uses NEON directly, since NEON is mandatory in the
//! AArch64 specification and needs no runtime detection. With the `simd`
//! feature disabled, or on an architecture with no vectorized path here,
//! every primitive falls back to the plain WIDTH=1 scalar backend.
//!
//! All functions operate on `f32` (this crate's native audio sample type)
//! and on the shared length of their input slices; any trailing elements
//! beyond a full SIMD-width chunk are handled by a scalar remainder loop so
//! callers never need to pad buffers to a particular width.
//!
//! ## One implementation, every backend
//!
//! There's exactly one implementation of `dot`/`mul_elementwise`/
//! `mix_scalar`, written generically over [`crate::simd::generic::SimdFloat`]
//! in [`crate::simd::generic::ops`], and exactly one place that decides
//! which backend is active for this module:
//! [`crate::simd::generic::dispatch::detect_backend`]. This module's
//! `x86_64`/`aarch64` files are thin dispatch wrappers - they call
//! `detect_backend()` (or, on `aarch64`, skip straight to NEON, since
//! there's nothing to detect), then monomorphize
//! [`crate::simd::generic::ops`] against whichever concrete lane type
//! ([`crate::simd::generic::avx512::F32x16`],
//! [`crate::simd::generic::avx2::F32x8`],
//! [`crate::simd::generic::sse2::F32x4Sse`],
//! [`crate::simd::generic::neon::F32x4`], or
//! [`crate::simd::generic::ScalarFloat`]) that backend uses. [`crate::sim`]'s
//! circuit simulation shares the exact same lane types and the same
//! [`crate::simd::generic::SimdFloat`] trait for its own math (exp/log/
//! sigmoid/etc, in [`crate::simd::generic::exp`],
//! [`crate::simd::generic::log`], [`crate::simd::generic::sigmoid`]) - it
//! just picks one backend unconditionally at
//! compile time via the `sim-avx2`/`sim-avx512`/`sim-neon` features
//! (assuming the build targets hardware known to support it) instead of
//! detecting at runtime, since it doesn't need this module's
//! single-portable-binary guarantee.

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod x86_64;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64;

/// `f32`-lane, trait-based SIMD abstraction ([`generic::SimdFloat`]) and
/// fast math (exp/log/sigmoid/reciprocal), used by [`crate::sim`].
pub mod generic;

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        x86_64::dot(a, b)
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        aarch64::dot(a, b)
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        generic::ops::dot::<generic::ScalarFloat>(a, b)
    }
}

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[inline]
pub fn mul_elementwise(dst: &mut [f32], a: &[f32], b: &[f32]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        x86_64::mul_elementwise(dst, a, b)
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        aarch64::mul_elementwise(dst, a, b)
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        generic::ops::mul_elementwise::<generic::ScalarFloat>(dst, a, b)
    }
}

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[inline]
pub fn mix_scalar(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        x86_64::mix_scalar(dst, dry, wet, mix)
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        aarch64::mix_scalar(dst, dry, wet, mix)
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        generic::ops::mix_scalar::<generic::ScalarFloat>(dst, dry, wet, mix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        generic::ops::dot::<generic::ScalarFloat>(a, b)
    }
    fn scalar_mul_elementwise(dst: &mut [f32], a: &[f32], b: &[f32]) {
        generic::ops::mul_elementwise::<generic::ScalarFloat>(dst, a, b)
    }
    fn scalar_mix_scalar(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
        generic::ops::mix_scalar::<generic::ScalarFloat>(dst, dry, wet, mix)
    }

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).sin()).collect();

        let expected = scalar_dot(&a, &b);

        assert!((dot(&a, &b) - expected).abs() < 1e-4);
    }

    #[test]
    fn mul_elementwise_matches_scalar() {
        let a: Vec<f32> = (0..23).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..23).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        let mut dst = vec![0.0f32; 23];
        let mut expected = vec![0.0f32; 23];

        mul_elementwise(&mut dst, &a, &b);
        scalar_mul_elementwise(&mut expected, &a, &b);

        for i in 0..23 {
            assert!((dst[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn mix_scalar_matches_manual() {
        let dry: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let wet: Vec<f32> = (0..19).map(|i| -(i as f32)).collect();
        let mut dst = vec![0.0f32; 19];
        let mut expected = vec![0.0f32; 19];

        mix_scalar(&mut dst, &dry, &wet, 0.25);
        scalar_mix_scalar(&mut expected, &dry, &wet, 0.25);

        for i in 0..19 {
            let expected = expected[i];
            assert!((dst[i] - expected).abs() < 1e-6);
        }
    }
}
