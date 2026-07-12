use super::exp::simd_exp;
use super::traits::SimdFloat;

/// Fast SIMD natural logarithm: `ln(x)`.
///
/// Algorithm:
/// 1. Extract exponent and mantissa from IEEE 754 representation
/// 2. Normalize mantissa to [sqrt(2)/2, sqrt(2)] for better polynomial convergence
/// 3. Polynomial approximation of `ln(1+f)` where f = mantissa - 1
/// 4. Reconstruct: `ln(x) = ln(m) + e * ln(2)`
///
/// Max error < 1e-5 relative to `f32::ln()` for x > 0.
#[inline]
pub fn simd_log<S: SimdFloat>(x: S) -> S {
    // Protect against non-positive input
    let tiny = S::splat(1.175494e-38);
    let x = x.max(tiny);

    // Extract exponent and mantissa via memory round-trip
    let mut buf = [0.0f32; 16];
    unsafe {
        x.store(buf.as_mut_ptr());
    }

    let mut mantissa_buf = [0.0f32; 16];
    let mut exponent_buf = [0.0f32; 16];

    let sqrt2 = core::f32::consts::SQRT_2;

    for i in 0..S::WIDTH {
        let bits = buf[i].to_bits();
        let mut e = ((bits >> 23) & 0xFF) as i32 - 127;
        let m_bits = (bits & 0x007F_FFFF) | 0x3F80_0000;
        let mut m = f32::from_bits(m_bits); // m in [1.0, 2.0)

        // Normalize to [sqrt(2)/2, sqrt(2)] ≈ [0.707, 1.414]
        // If m >= sqrt(2), divide by 2 and increment exponent
        if m >= sqrt2 {
            m *= 0.5;
            e += 1;
        }

        mantissa_buf[i] = m;
        exponent_buf[i] = e as f32;
    }

    let m = unsafe { S::load(mantissa_buf.as_ptr()) };
    let e = unsafe { S::load(exponent_buf.as_ptr()) };

    // f = m - 1, now f is in approximately [-0.293, 0.414]
    let f = m.sub(S::one());

    // Use rational approximation: ln(1+f) ≈ f * P(f) / Q(f)
    // Or use a higher degree polynomial with better coefficients.
    //
    // Alternative approach: use ln(m) = 2*atanh((m-1)/(m+1)) which converges faster.
    // Let s = (m-1)/(m+1), then ln(m) = 2*(s + s^3/3 + s^5/5 + s^7/7 + ...)
    let two = S::splat(2.0);
    let s = f.div(m.add(S::one())); // s = (m-1)/(m+1)
    let s2 = s.mul(s);

    // ln(m) = 2*s*(1 + s^2/3 + s^4/5 + s^6/7 + s^8/9)
    let c1 = S::one();
    let c3 = S::splat(1.0 / 3.0);
    let c5 = S::splat(1.0 / 5.0);
    let c7 = S::splat(1.0 / 7.0);
    let c9 = S::splat(1.0 / 9.0);
    let c11 = S::splat(1.0 / 11.0);

    let q = c11.fma(s2, c9);
    let q = q.fma(s2, c7);
    let q = q.fma(s2, c5);
    let q = q.fma(s2, c3);
    let q = q.fma(s2, c1);
    let p = two.mul(s).mul(q);

    // Reconstruct: ln(x) = ln(m) + e * ln(2)
    let ln2 = S::splat(core::f32::consts::LN_2);
    e.fma(ln2, p)
}

/// Stable computation of `ln(1 + exp(x))` — the softplus function.
///
/// Uses the identity: `ln(1 + exp(x)) = max(x, 0) + ln(1 + exp(-|x|))`
/// This avoids overflow for large positive x and underflow for large negative x.
#[inline]
pub fn simd_log1pexp<S: SimdFloat>(x: S) -> S {
    let abs_x = x.abs();
    let max_x_0 = x.max(S::zero());

    let neg_abs_x = abs_x.neg();
    let exp_neg = simd_exp(neg_abs_x);
    let one_plus = exp_neg.add(S::one());
    let log_part = simd_log(one_plus);

    max_x_0.add(log_part)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::generic::ScalarFloat;

    #[test]
    fn test_log_accuracy() {
        let n = 1_000_000;
        let mut max_rel_err: f32 = 0.0;

        for i in 1..n {
            let x = 0.001 + 20.0 * (i as f32) / (n as f32);
            let expected = x.ln();
            let result = simd_log(ScalarFloat(x)).0;

            let abs_err = (result - expected).abs();
            let rel_err = if expected.abs() > 1e-6 {
                abs_err / expected.abs()
            } else {
                abs_err
            };

            max_rel_err = max_rel_err.max(rel_err);
        }

        assert!(
            max_rel_err < 1e-5,
            "log max relative error {max_rel_err} exceeds 1e-5"
        );
    }

    #[test]
    fn test_log_edge_cases() {
        // ln(1) = 0
        let r = simd_log(ScalarFloat(1.0)).0;
        assert!(r.abs() < 1e-6, "ln(1) = {r}");

        // ln(e) ≈ 1
        let r = simd_log(ScalarFloat(core::f32::consts::E)).0;
        assert!((r - 1.0).abs() < 1e-5, "ln(e) = {r}");

        // Very small positive
        let r = simd_log(ScalarFloat(1e-30)).0;
        assert!(r.is_finite(), "ln(1e-30) should be finite");

        // Zero / negative should not produce NaN (clamped to tiny)
        let r = simd_log(ScalarFloat(0.0)).0;
        assert!(r.is_finite(), "ln(0) should be finite (clamped)");
    }

    #[test]
    fn test_log1pexp_accuracy() {
        let n = 100_000;
        let mut max_abs_err: f32 = 0.0;

        for i in 0..n {
            let x = -10.0 + 20.0 * (i as f32) / (n as f32);
            let expected = (1.0 + x.exp()).ln();
            let result = simd_log1pexp(ScalarFloat(x)).0;

            let abs_err = (result - expected).abs();
            max_abs_err = max_abs_err.max(abs_err);
        }

        assert!(
            max_abs_err < 1e-4,
            "log1pexp max abs error {max_abs_err} exceeds 1e-4"
        );
    }

    #[test]
    fn test_log1pexp_large_values() {
        // Large positive: ln(1+exp(50)) ≈ 50
        let r = simd_log1pexp(ScalarFloat(50.0)).0;
        assert!((r - 50.0).abs() < 0.01, "log1pexp(50) = {r}");

        // Large negative: ln(1+exp(-50)) ≈ 0
        let r = simd_log1pexp(ScalarFloat(-50.0)).0;
        assert!(r.abs() < 0.001 && r >= 0.0, "log1pexp(-50) = {r}");

        // Zero: ln(1+exp(0)) = ln(2)
        let r = simd_log1pexp(ScalarFloat(0.0)).0;
        assert!(
            (r - core::f32::consts::LN_2).abs() < 1e-4,
            "log1pexp(0) = {r}"
        );
    }
}
