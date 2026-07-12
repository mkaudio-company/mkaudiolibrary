//! SIMD-accelerated primitives for hot DSP/TF loops.
//!
//! Enabled via the `simd` Cargo feature. On `x86_64` this dispatches at
//! runtime to AVX2+FMA when available, falling back to baseline SSE2
//! (guaranteed present on every x86_64 target). On `aarch64` this uses NEON
//! directly, since NEON is mandatory in the AArch64 specification and needs
//! no runtime detection. Every primitive also has a plain scalar
//! implementation used when the `simd` feature is disabled or the target
//! architecture has no vectorized path here.
//!
//! All functions operate on `f32` (this crate's native audio sample type)
//! and on the shared length of their input slices; any trailing elements
//! beyond a full SIMD-width chunk are handled by a scalar remainder loop so
//! callers never need to pad buffers to a particular width.
//!
//! ## Relationship to [`generic`]
//!
//! The arithmetic isn't duplicated per width or per caller: [`generic`]
//! holds the actual SIMD lane types ([`generic::avx2::F32x8`],
//! [`generic::sse2::F32x4Sse`], [`generic::neon::F32x4`],
//! [`generic::ScalarFloat`]) behind one trait ([`generic::SimdFloat`]) and
//! one set of width-agnostic implementations ([`generic::ops`]). This
//! module's `x86_64`/`aarch64` backends are thin runtime-dispatch wrappers
//! that pick a concrete lane type and call into [`generic::ops`];
//! [`crate::sim`]'s circuit simulation instead picks its lane type once at
//! compile time via the `sim-avx2`/`sim-avx512`/`sim-neon` features and
//! calls the same trait directly for its own math (exp/log/sigmoid/etc, in
//! [`generic::exp`], [`generic::log`], [`generic::sigmoid`]). The two
//! callers guard the lane types differently, matching their different
//! deployment models: this module's AVX2/SSE2 dispatch is runtime-detected
//! (`is_x86_feature_detected!`) behind a matching `#[target_feature]` call
//! site, so a single binary is portable across whatever `x86_64` CPU it
//! actually runs on; `sim-avx2`/`sim-avx512`/`sim-neon` instead pick one
//! backend unconditionally at compile time, on the assumption that whoever
//! enables that feature is building for hardware known to support it.

mod scalar;

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
        scalar::dot(a, b)
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
        scalar::mul_elementwise(dst, a, b)
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
        scalar::mix_scalar(dst, dry, wet, mix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).sin()).collect();

        let expected = scalar::dot(&a, &b);

        assert!((dot(&a, &b) - expected).abs() < 1e-4);
    }

    #[test]
    fn mul_elementwise_matches_scalar() {
        let a: Vec<f32> = (0..23).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..23).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        let mut dst = vec![0.0f32; 23];
        let mut expected = vec![0.0f32; 23];

        mul_elementwise(&mut dst, &a, &b);
        scalar::mul_elementwise(&mut expected, &a, &b);

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
        scalar::mix_scalar(&mut expected, &dry, &wet, 0.25);

        for i in 0..19 {
            let expected = expected[i];
            assert!((dst[i] - expected).abs() < 1e-6);
        }
    }
}
