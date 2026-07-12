//! AVX-512 SIMD backend — 16-wide f32 using `__m512`.
//!
//! Requires the `avx512f` target feature at the call site (guarded by
//! `#[target_feature]` + `is_x86_feature_detected!` in `crate::simd::x86_64`
//! - an internal module - or by the `sim-avx512` opt-in feature assuming
//! target hardware support in [`crate::sim`]).

use core::arch::x86_64::*;

use super::traits::SimdFloat;

/// 16-wide f32 SIMD type using AVX-512F.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x16(__m512);

impl SimdFloat for F32x16 {
    const WIDTH: usize = 16;

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        F32x16(unsafe { _mm512_loadu_ps(ptr) })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        unsafe { _mm512_storeu_ps(ptr, self.0) };
    }

    #[inline(always)]
    fn splat(v: f32) -> Self {
        F32x16(unsafe { _mm512_set1_ps(v) })
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        F32x16(unsafe { _mm512_add_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F32x16(unsafe { _mm512_sub_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F32x16(unsafe { _mm512_mul_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        F32x16(unsafe { _mm512_div_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        F32x16(unsafe { _mm512_fmadd_ps(self.0, b.0, c.0) })
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        F32x16(unsafe { _mm512_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        F32x16(unsafe { _mm512_min_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let mask = unsafe { _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFF_u32 as i32)) };
        F32x16(unsafe { _mm512_and_ps(self.0, mask) })
    }

    #[inline(always)]
    fn neg(self) -> Self {
        let zero = unsafe { _mm512_setzero_ps() };
        F32x16(unsafe { _mm512_sub_ps(zero, self.0) })
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        // AVX-512 compares produce a 16-bit mask register, not a vector -
        // broadcast it back into a full lane-wise all-1s/all-0s vector so
        // this matches every other backend's `SimdFloat::cmp_ge` contract.
        let mask = unsafe { _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GE_OQ) };
        let all_ones = unsafe { _mm512_castsi512_ps(_mm512_set1_epi32(-1)) };
        F32x16(unsafe { _mm512_maskz_mov_ps(mask, all_ones) })
    }

    #[inline(always)]
    fn blend(mask: Self, a: Self, b: Self) -> Self {
        // Inverse of `cmp_ge`: collapse the lane-wise vector mask back into
        // a mask register for `_mm512_mask_blend_ps`.
        let k = unsafe { _mm512_movepi32_mask(_mm512_castps_si512(mask.0)) };
        F32x16(unsafe { _mm512_mask_blend_ps(k, b.0, a.0) })
    }

    #[inline(always)]
    fn first(self) -> f32 {
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(self.0)) }
    }
}
