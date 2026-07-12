//! Width-agnostic `dot`/`mul_elementwise`/`mix_scalar`, written once against
//! [`SimdFloat`] and monomorphized per backend. This is the single
//! implementation behind both [`crate::simd`]'s runtime-dispatched
//! `x86_64`/`aarch64` entry points and any [`crate::sim`] code that wants
//! the same primitives at its own compile-time-selected width.

use super::traits::SimdFloat;

/// Largest lane count among the backends in this module (AVX2 = 8); sized
/// generously so the horizontal-sum scratch buffer in [`dot`] never
/// overflows regardless of which `S` is monomorphized.
const MAX_WIDTH: usize = 16;

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[inline(always)]
pub fn dot<S: SimdFloat>(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / S::WIDTH;

    let mut acc = S::zero();
    for i in 0..chunks {
        unsafe {
            let va = S::load(a.as_ptr().add(i * S::WIDTH));
            let vb = S::load(b.as_ptr().add(i * S::WIDTH));
            acc = va.fma(vb, acc);
        }
    }

    let mut lanes = [0.0f32; MAX_WIDTH];
    unsafe { acc.store(lanes.as_mut_ptr()) };
    let mut sum: f32 = lanes[..S::WIDTH].iter().sum();

    for i in (chunks * S::WIDTH)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[inline(always)]
pub fn mul_elementwise<S: SimdFloat>(dst: &mut [f32], a: &[f32], b: &[f32]) {
    let len = dst.len().min(a.len()).min(b.len());
    let chunks = len / S::WIDTH;

    for i in 0..chunks {
        unsafe {
            let va = S::load(a.as_ptr().add(i * S::WIDTH));
            let vb = S::load(b.as_ptr().add(i * S::WIDTH));
            va.mul(vb).store(dst.as_mut_ptr().add(i * S::WIDTH));
        }
    }
    for i in (chunks * S::WIDTH)..len {
        dst[i] = a[i] * b[i];
    }
}

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[inline(always)]
pub fn mix_scalar<S: SimdFloat>(dst: &mut [f32], dry: &[f32], wet: &[f32], mix: f32) {
    let len = dst.len().min(dry.len()).min(wet.len());
    let chunks = len / S::WIDTH;
    let vmix = S::splat(mix);
    let vinv = S::splat(1.0 - mix);

    for i in 0..chunks {
        unsafe {
            let vdry = S::load(dry.as_ptr().add(i * S::WIDTH));
            let vwet = S::load(wet.as_ptr().add(i * S::WIDTH));
            let dry_term = vdry.mul(vinv);
            vwet.fma(vmix, dry_term)
                .store(dst.as_mut_ptr().add(i * S::WIDTH));
        }
    }
    for i in (chunks * S::WIDTH)..len {
        dst[i] = dry[i] * (1.0 - mix) + wet[i] * mix;
    }
}
