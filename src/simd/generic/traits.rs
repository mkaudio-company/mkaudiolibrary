/// Core SIMD abstraction trait.
///
/// All math in the library is written generically over this trait.
/// The scalar backend (`ScalarFloat`) implements it with `WIDTH=1`;
/// platform-specific backends (AVX2, AVX512, NEON) are added later.
pub trait SimdFloat: Copy + Clone + Sized {
    /// Number of f32 lanes in this SIMD type.
    const WIDTH: usize;

    /// Load WIDTH f32 values from `ptr` (unaligned-safe).
    ///
    /// # Safety
    /// `ptr` must be valid for `WIDTH` f32 reads. No particular alignment
    /// is required - backends use unaligned load instructions so callers
    /// can point directly into ordinary `&[f32]` slices.
    unsafe fn load(ptr: *const f32) -> Self;

    /// Store WIDTH f32 values to `ptr` (unaligned-safe).
    ///
    /// # Safety
    /// `ptr` must be valid for `WIDTH` f32 writes. No particular alignment
    /// is required - backends use unaligned store instructions so callers
    /// can point directly into ordinary `&mut [f32]` slices.
    unsafe fn store(self, ptr: *mut f32);

    /// Broadcast a single f32 to all lanes.
    fn splat(v: f32) -> Self;

    /// Lane-wise addition.
    fn add(self, rhs: Self) -> Self;

    /// Lane-wise subtraction.
    fn sub(self, rhs: Self) -> Self;

    /// Lane-wise multiplication.
    fn mul(self, rhs: Self) -> Self;

    /// Lane-wise division.
    fn div(self, rhs: Self) -> Self;

    /// Fused multiply-add: `self * b + c`.
    fn fma(self, b: Self, c: Self) -> Self;

    /// Lane-wise maximum.
    fn max(self, rhs: Self) -> Self;

    /// Lane-wise minimum.
    fn min(self, rhs: Self) -> Self;

    /// Lane-wise absolute value.
    fn abs(self) -> Self;

    /// Lane-wise negation.
    fn neg(self) -> Self;

    /// Lane-wise comparison: `self >= rhs`. Returns mask where true lanes are all-1 bits.
    fn cmp_ge(self, rhs: Self) -> Self;

    /// Bitwise blend: for each lane, if `mask` bit is set select `a`, else select `b`.
    fn blend(mask: Self, a: Self, b: Self) -> Self;

    /// Return the value of the first lane (useful for scalar extraction).
    fn first(self) -> f32;

    // ── Convenience constants ──

    /// All lanes set to `0.0`.
    fn zero() -> Self {
        Self::splat(0.0)
    }

    /// All lanes set to `1.0`.
    fn one() -> Self {
        Self::splat(1.0)
    }
}
