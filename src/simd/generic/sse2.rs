//! `x86_64` SSE2 SIMD backend — 4-wide f32 using `__m128`.
//!
//! SSE2 is guaranteed present on every `x86_64` target (it's part of the
//! baseline ABI), so this needs no runtime feature detection and no opt-in
//! Cargo feature - it's the always-available floor beneath
//! [`crate::simd::generic::avx2`].

use core::arch::x86_64::*;

use super::traits::SimdFloat;

/// 4-wide f32 SIMD type using baseline SSE2.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x4Sse(__m128);

impl SimdFloat for F32x4Sse {
    const WIDTH: usize = 4;

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        F32x4Sse(unsafe { _mm_loadu_ps(ptr) })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        unsafe { _mm_storeu_ps(ptr, self.0) };
    }

    #[inline(always)]
    fn splat(v: f32) -> Self {
        F32x4Sse(unsafe { _mm_set1_ps(v) })
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        F32x4Sse(unsafe { _mm_add_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F32x4Sse(unsafe { _mm_sub_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F32x4Sse(unsafe { _mm_mul_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        F32x4Sse(unsafe { _mm_div_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        // No FMA3 on baseline SSE2 - multiply then add as two instructions.
        F32x4Sse(unsafe { _mm_add_ps(_mm_mul_ps(self.0, b.0), c.0) })
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        F32x4Sse(unsafe { _mm_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        F32x4Sse(unsafe { _mm_min_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let mask = unsafe { _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF_u32 as i32)) };
        F32x4Sse(unsafe { _mm_and_ps(self.0, mask) })
    }

    #[inline(always)]
    fn neg(self) -> Self {
        let zero = unsafe { _mm_setzero_ps() };
        F32x4Sse(unsafe { _mm_sub_ps(zero, self.0) })
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        F32x4Sse(unsafe { _mm_cmpge_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn blend(mask: Self, a: Self, b: Self) -> Self {
        // SSE2 has no blendv - build it from AND/ANDNOT/OR against the mask.
        F32x4Sse(unsafe { _mm_or_ps(_mm_and_ps(mask.0, a.0), _mm_andnot_ps(mask.0, b.0)) })
    }

    #[inline(always)]
    fn first(self) -> f32 {
        unsafe { _mm_cvtss_f32(self.0) }
    }
}
