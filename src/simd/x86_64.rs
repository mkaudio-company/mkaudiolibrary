//! `x86_64` backend: AVX2+FMA when available at runtime, falling back to
//! baseline SSE2 (guaranteed present on every `x86_64` target).
//!
//! Both widths are thin, runtime-detected dispatch wrappers around
//! [`crate::simd::generic::ops`]'s width-agnostic implementations,
//! monomorphized for [`crate::simd::generic::avx2::F32x8`] and
//! [`crate::simd::generic::sse2::F32x4Sse`] respectively - the arithmetic
//! itself isn't duplicated per width.

use super::generic::ops;
use super::generic::{avx2::F32x8, sse2::F32x4Sse};

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { dot_avx2_fma(a, b) }
    } else {
        unsafe { dot_sse2(a, b) }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    ops::dot::<F32x8>(a, b)
}

#[target_feature(enable = "sse2")]
unsafe fn dot_sse2(a: &[f32], b: &[f32]) -> f32 {
    ops::dot::<F32x4Sse>(a, b)
}

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[inline]
pub fn mul_elementwise(dst: &mut [f32], a: &[f32], b: &[f32]) {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { mul_avx2(dst, a, b) }
    } else {
        unsafe { mul_sse2(dst, a, b) }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_avx2(dst: &mut [f32], a: &[f32], b: &[f32]) {
    ops::mul_elementwise::<F32x8>(dst, a, b)
}

#[target_feature(enable = "sse2")]
unsafe fn mul_sse2(dst: &mut [f32], a: &[f32], b: &[f32]) {
    ops::mul_elementwise::<F32x4Sse>(dst, a, b)
}

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[inline]
pub fn mix_scalar(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { mix_avx2_fma(dst, dry, wet, mix) }
    } else {
        unsafe { mix_sse2(dst, dry, wet, mix) }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn mix_avx2_fma(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    ops::mix_scalar::<F32x8>(dst, dry, wet, mix)
}

#[target_feature(enable = "sse2")]
unsafe fn mix_sse2(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    ops::mix_scalar::<F32x4Sse>(dst, dry, wet, mix)
}
