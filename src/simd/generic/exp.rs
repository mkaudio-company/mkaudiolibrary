use super::traits::SimdFloat;

/// Fast SIMD exponential: `exp(x)` using range reduction + degree-5 polynomial.
///
/// Algorithm:
/// 1. Clamp input to [-87.3, 88.7] to avoid overflow/underflow
/// 2. Range reduction: n = round(x / ln(2)), r = x - n*ln(2)
/// 3. Polynomial approximation of exp(r) on [-0.5*ln(2), 0.5*ln(2)]
/// 4. Reconstruct: exp(x) = 2^n * exp(r)
///
/// Max error < 1e-6 relative to `f32::exp()` over [-87, 88].
#[inline]
pub fn simd_exp<S: SimdFloat>(x: S) -> S {
    // Clamp to safe range
    let x = x.max(S::splat(-87.3)).min(S::splat(88.7));

    // Range reduction: n = round(x / ln2)
    let log2e = S::splat(core::f32::consts::LOG2_E); // 1/ln(2)
    let nf = simd_round::<S>(x.mul(log2e));

    // r = x - n * ln2 (Cody-Waite two-step for precision)
    let ln2_hi = S::splat(0.693145751953125);
    let ln2_lo = S::splat(1.428606765330187e-6);
    let r = x.sub(nf.mul(ln2_hi)).sub(nf.mul(ln2_lo));

    // Degree-5 polynomial for exp(r) on [-ln2/2, ln2/2]
    let c0 = S::splat(1.0);
    let c1 = S::splat(1.0);
    let c2 = S::splat(0.5000001);
    let c3 = S::splat(0.1666666);
    let c4 = S::splat(0.0416681);
    let c5 = S::splat(0.0083341);

    // Horner: p = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    let p = c5.fma(r, c4);
    let p = p.fma(r, c3);
    let p = p.fma(r, c2);
    let p = p.fma(r, c1);
    let p = p.fma(r, c0);

    // Reconstruct: exp(x) = 2^n * p via IEEE 754 bit manipulation
    pow2_mul::<S>(p, nf)
}

/// Round to nearest integer (as float).
#[inline]
fn simd_round<S: SimdFloat>(x: S) -> S {
    // Add/subtract 2^23 to force rounding, then reverse
    let magic = S::splat(12582912.0); // 2^23 + 2^22 — rounds to nearest even
    let sign = x.cmp_ge(S::zero());
    let pos = x.add(magic).sub(magic);
    let neg = x.sub(magic).add(magic);
    S::blend(sign, pos, neg)
}

/// Compute `x * 2^n` where n is an integer-valued float.
///
/// Uses IEEE 754 bit manipulation through memory to construct 2^n,
/// then multiplies. Works for n in [-126, 127].
#[inline]
fn pow2_mul<S: SimdFloat>(x: S, nf: S) -> S {
    // Store n values, convert to integer, build 2^n via bit manipulation
    let mut n_buf = [0.0f32; 16];
    unsafe {
        nf.store(n_buf.as_mut_ptr());
    }

    let mut pow2_buf = [0.0f32; 16];
    for i in 0..S::WIDTH {
        let n = n_buf[i] as i32;
        // IEEE 754: 2^n has bits = (n + 127) << 23
        let n_clamped = n.clamp(-126, 127);
        let bits = ((n_clamped + 127) as u32) << 23;
        pow2_buf[i] = f32::from_bits(bits);
    }

    let pow2n = unsafe { S::load(pow2_buf.as_ptr()) };
    x.mul(pow2n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::generic::ScalarFloat;

    #[test]
    fn test_exp_accuracy() {
        let n = 1_000_000;
        let mut max_rel_err: f32 = 0.0;

        for i in 0..n {
            let x = -10.0 + 20.0 * (i as f32) / (n as f32);
            let expected = x.exp();
            let result = simd_exp(ScalarFloat(x)).0;

            let abs_err = (result - expected).abs();
            let rel_err = if expected.abs() > 1e-10 {
                abs_err / expected.abs()
            } else {
                abs_err
            };

            max_rel_err = max_rel_err.max(rel_err);
        }

        assert!(
            max_rel_err < 1e-5,
            "exp max relative error {max_rel_err} exceeds 1e-5"
        );
    }

    #[test]
    fn test_exp_edge_cases() {
        // Zero
        let r = simd_exp(ScalarFloat(0.0)).0;
        assert!((r - 1.0).abs() < 1e-6, "exp(0) = {r}");

        // Large positive (should clamp, not overflow)
        let r = simd_exp(ScalarFloat(100.0)).0;
        assert!(r.is_finite(), "exp(100) should be finite");

        // Large negative (should approach 0, not underflow to NaN)
        let r = simd_exp(ScalarFloat(-100.0)).0;
        assert!(r.is_finite() && r >= 0.0, "exp(-100) = {r}");

        // Moderate values
        let r = simd_exp(ScalarFloat(1.0)).0;
        assert!((r - core::f32::consts::E).abs() < 1e-5, "exp(1) = {r}");
    }

    #[test]
    fn test_exp_no_nan() {
        for &x in &[-88.0, -50.0, -1.0, 0.0, 1.0, 50.0, 88.0] {
            let r = simd_exp(ScalarFloat(x)).0;
            assert!(!r.is_nan(), "exp({x}) produced NaN");
            assert!(!r.is_infinite() || x > 88.0, "exp({x}) = {r}");
        }
    }
}
