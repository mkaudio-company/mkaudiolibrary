//! `x86_64` backend: dispatches at runtime to AVX-512, then AVX2+FMA, then
//! baseline SSE2 (guaranteed present on every `x86_64` target) - whichever
//! is the best backend [`crate::simd::generic::dispatch::detect_backend`]
//! finds available on the running CPU.
//!
//! Every width is a thin dispatch wrapper around
//! [`crate::simd::generic::ops`]'s width-agnostic implementations,
//! monomorphized for [`crate::simd::generic::avx512::F32x16`],
//! [`crate::simd::generic::avx2::F32x8`], or
//! [`crate::simd::generic::sse2::F32x4Sse`] respectively - the arithmetic
//! itself isn't duplicated per width.

use super::generic;
use super::generic::dispatch::{SimdBackend, detect_backend};
use super::generic::{avx2::F32x8, avx512::F32x16, sse2::F32x4Sse};

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    match detect_backend() {
        SimdBackend::Avx512 => unsafe { dot_avx512(a, b) },
        SimdBackend::Avx2 => unsafe { dot_avx2_fma(a, b) },
        _ => unsafe { dot_sse2(a, b) },
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    generic::ops::dot::<F32x16>(a, b)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    generic::ops::dot::<F32x8>(a, b)
}

#[target_feature(enable = "sse2")]
unsafe fn dot_sse2(a: &[f32], b: &[f32]) -> f32 {
    generic::ops::dot::<F32x4Sse>(a, b)
}

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[inline]
pub fn mul_elementwise(dst: &mut [f32], a: &[f32], b: &[f32]) {
    match detect_backend() {
        SimdBackend::Avx512 => unsafe { mul_avx512(dst, a, b) },
        SimdBackend::Avx2 => unsafe { mul_avx2(dst, a, b) },
        _ => unsafe { mul_sse2(dst, a, b) },
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn mul_avx512(dst: &mut [f32], a: &[f32], b: &[f32]) {
    generic::ops::mul_elementwise::<F32x16>(dst, a, b)
}

#[target_feature(enable = "avx2")]
unsafe fn mul_avx2(dst: &mut [f32], a: &[f32], b: &[f32]) {
    generic::ops::mul_elementwise::<F32x8>(dst, a, b)
}

#[target_feature(enable = "sse2")]
unsafe fn mul_sse2(dst: &mut [f32], a: &[f32], b: &[f32]) {
    generic::ops::mul_elementwise::<F32x4Sse>(dst, a, b)
}

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[inline]
pub fn mix_scalar(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    match detect_backend() {
        SimdBackend::Avx512 => unsafe { mix_avx512(dst, dry, wet, mix) },
        SimdBackend::Avx2 => unsafe { mix_avx2_fma(dst, dry, wet, mix) },
        _ => unsafe { mix_sse2(dst, dry, wet, mix) },
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn mix_avx512(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    generic::ops::mix_scalar::<F32x16>(dst, dry, wet, mix)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn mix_avx2_fma(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    generic::ops::mix_scalar::<F32x8>(dst, dry, wet, mix)
}

#[target_feature(enable = "sse2")]
unsafe fn mix_sse2(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    generic::ops::mix_scalar::<F32x4Sse>(dst, dry, wet, mix)
}
