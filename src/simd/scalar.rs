//! Portable scalar fallback, used when the `simd` feature is disabled or
//! the target architecture has no vectorized backend here. Also used
//! (unconditionally) by this module's own tests as the reference
//! implementation to check the SIMD backends against - which is the only
//! use these functions have on a configuration where a vectorized backend
//! is active, hence the crate-visible `dead_code` allowance below for
//! non-test builds.
#![cfg_attr(not(test), allow(dead_code))]

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[inline]
pub fn mul_elementwise(dst: &mut [f32], a: &[f32], b: &[f32]) {
    let len = dst.len().min(a.len()).min(b.len());
    for i in 0..len {
        dst[i] = a[i] * b[i];
    }
}

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[inline]
pub fn mix_scalar(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    let len = dst.len().min(dry.len()).min(wet.len());
    for i in 0..len {
        dst[i] = dry[i] * (1.0 - mix) + wet[i] * mix;
    }
}
