use super::traits::SimdFloat;

/// Scalar (WIDTH=1) backend — the universal fallback.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct ScalarFloat(pub f32);

impl SimdFloat for ScalarFloat {
    const WIDTH: usize = 1;

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        ScalarFloat(unsafe { *ptr })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        unsafe { *ptr = self.0 };
    }

    #[inline(always)]
    fn splat(v: f32) -> Self {
        ScalarFloat(v)
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        ScalarFloat(self.0 + rhs.0)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        ScalarFloat(self.0 - rhs.0)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        ScalarFloat(self.0 * rhs.0)
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        ScalarFloat(self.0 / rhs.0)
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        ScalarFloat(self.0.mul_add(b.0, c.0))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        ScalarFloat(if self.0 >= rhs.0 { self.0 } else { rhs.0 })
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        ScalarFloat(if self.0 <= rhs.0 { self.0 } else { rhs.0 })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        ScalarFloat(self.0.abs())
    }

    #[inline(always)]
    fn neg(self) -> Self {
        ScalarFloat(-self.0)
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            ScalarFloat(f32::from_bits(0xFFFF_FFFF))
        } else {
            ScalarFloat(0.0)
        }
    }

    #[inline(always)]
    fn blend(mask: Self, a: Self, b: Self) -> Self {
        if mask.0.to_bits() != 0 { a } else { b }
    }

    #[inline(always)]
    fn first(self) -> f32 {
        self.0
    }
}
