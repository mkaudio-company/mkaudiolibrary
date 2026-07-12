//! ARM NEON SIMD backend — 4-wide f32 using float32x4_t.
//!
//! Enabled with `--features neon` on aarch64 targets.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use super::traits::SimdFloat;

/// 4-wide f32 SIMD type using ARM NEON.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x4(#[cfg(target_arch = "aarch64")] float32x4_t);

#[cfg(target_arch = "aarch64")]
impl SimdFloat for F32x4 {
    const WIDTH: usize = 4;

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        F32x4(unsafe { vld1q_f32(ptr) })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        unsafe { vst1q_f32(ptr, self.0) };
    }

    #[inline(always)]
    fn splat(v: f32) -> Self {
        F32x4(unsafe { vdupq_n_f32(v) })
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        F32x4(unsafe { vaddq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F32x4(unsafe { vsubq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F32x4(unsafe { vmulq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        F32x4(unsafe { vdivq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        F32x4(unsafe { vfmaq_f32(c.0, self.0, b.0) })
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        F32x4(unsafe { vmaxq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        F32x4(unsafe { vminq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        F32x4(unsafe { vabsq_f32(self.0) })
    }

    #[inline(always)]
    fn neg(self) -> Self {
        F32x4(unsafe { vnegq_f32(self.0) })
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        let mask = unsafe { vcgeq_f32(self.0, rhs.0) };
        F32x4(unsafe { vreinterpretq_f32_u32(mask) })
    }

    #[inline(always)]
    fn blend(mask: Self, a: Self, b: Self) -> Self {
        let mask_u32 = unsafe { vreinterpretq_u32_f32(mask.0) };
        F32x4(unsafe { vbslq_f32(mask_u32, a.0, b.0) })
    }

    #[inline(always)]
    fn first(self) -> f32 {
        unsafe { vgetq_lane_f32(self.0, 0) }
    }
}
