//! `aarch64` backend: ARM NEON. No runtime detection needed since NEON is
//! mandatory in the AArch64 specification.
//!
//! Each function is a thin wrapper around
//! [`crate::simd::generic::ops`]'s width-agnostic implementation,
//! monomorphized for [`crate::simd::generic::neon::F32x4`] - the
//! arithmetic itself lives in one place, shared with the `x86_64` backend.

use super::generic::neon::F32x4;
use super::generic::ops;

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    ops::dot::<F32x4>(a, b)
}

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[inline]
pub fn mul_elementwise(dst: &mut [f32], a: &[f32], b: &[f32]) {
    ops::mul_elementwise::<F32x4>(dst, a, b)
}

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[inline]
pub fn mix_scalar(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    ops::mix_scalar::<F32x4>(dst, dry, wet, mix)
}
