//! AVX2 SIMD backend — 8-wide f32 using __m256.
//!
//! Enabled with `--features avx2` on x86_64 targets.
//! All functions use `#[target_feature(enable = "avx2,fma")]`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::traits::SimdFloat;

/// 8-wide f32 SIMD type using AVX2.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x8(#[cfg(target_arch = "x86_64")] __m256);

#[cfg(target_arch = "x86_64")]
impl SimdFloat for F32x8 {
    const WIDTH: usize = 8;

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        F32x8(unsafe { _mm256_loadu_ps(ptr) })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        unsafe { _mm256_storeu_ps(ptr, self.0) };
    }

    #[inline(always)]
    fn splat(v: f32) -> Self {
        F32x8(unsafe { _mm256_set1_ps(v) })
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        F32x8(unsafe { _mm256_add_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F32x8(unsafe { _mm256_sub_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F32x8(unsafe { _mm256_mul_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        F32x8(unsafe { _mm256_div_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        F32x8(unsafe { _mm256_fmadd_ps(self.0, b.0, c.0) })
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        F32x8(unsafe { _mm256_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        F32x8(unsafe { _mm256_min_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // Clear sign bit
        let mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF_u32 as i32)) };
        F32x8(unsafe { _mm256_and_ps(self.0, mask) })
    }

    #[inline(always)]
    fn neg(self) -> Self {
        let zero = unsafe { _mm256_setzero_ps() };
        F32x8(unsafe { _mm256_sub_ps(zero, self.0) })
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        F32x8(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_GE_OQ) })
    }

    #[inline(always)]
    fn blend(mask: Self, a: Self, b: Self) -> Self {
        F32x8(unsafe { _mm256_blendv_ps(b.0, a.0, mask.0) })
    }

    #[inline(always)]
    fn first(self) -> f32 {
        unsafe { _mm256_cvtss_f32(self.0) }
    }
}
