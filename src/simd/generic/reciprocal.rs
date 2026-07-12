use super::traits::SimdFloat;

/// Fast SIMD reciprocal: `1/x` with one Newton-Raphson refinement step.
///
/// For the scalar backend this is just division; SIMD backends will use
/// hardware approximate reciprocal + refinement.
#[inline]
pub fn simd_recip<S: SimdFloat>(x: S) -> S {
    // Initial estimate
    let r = S::one().div(x);

    // Newton refinement: r' = r * (2 - x * r)
    let two = S::splat(2.0);
    let xr = x.mul(r);
    r.mul(two.sub(xr))
}

/// Fast SIMD reciprocal square root: `1/sqrt(x)` with Newton-Raphson refinement.
#[inline]
pub fn simd_rsqrt<S: SimdFloat>(x: S) -> S {
    // For scalar, use 1/sqrt directly as initial estimate
    // SIMD backends will use hardware rsqrt + refinement
    let mut buf = [0.0f32; 16];
    unsafe {
        x.store(buf.as_mut_ptr());
    }

    let mut result = [0.0f32; 16];
    for i in 0..S::WIDTH {
        // Fast inverse square root (initial estimate)
        let val = buf[i];
        if val > 0.0 {
            result[i] = 1.0 / val.sqrt();
        } else {
            result[i] = 0.0;
        }
    }

    let r = unsafe { S::load(result.as_ptr()) };

    // Newton refinement: r' = r * (3 - x * r^2) / 2
    let three = S::splat(3.0);
    let half = S::splat(0.5);
    let xr2 = x.mul(r).mul(r);
    r.mul(three.sub(xr2)).mul(half)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::generic::ScalarFloat;

    #[test]
    fn test_recip_accuracy() {
        for &x in &[0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 0.001] {
            let result = simd_recip(ScalarFloat(x)).0;
            let expected = 1.0 / x;
            let rel_err = ((result - expected) / expected).abs();
            assert!(
                rel_err < 1e-6,
                "recip({x}): got {result}, expected {expected}, rel_err {rel_err}"
            );
        }
    }

    #[test]
    fn test_rsqrt_accuracy() {
        for &x in &[0.1, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0] {
            let result = simd_rsqrt(ScalarFloat(x)).0;
            let expected = 1.0 / x.sqrt();
            let rel_err = ((result - expected) / expected).abs();
            assert!(
                rel_err < 1e-5,
                "rsqrt({x}): got {result}, expected {expected}, rel_err {rel_err}"
            );
        }
    }
}
