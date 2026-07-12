use super::exp::simd_exp;
use super::traits::SimdFloat;

/// Fast SIMD sigmoid: `exp(x) / (1 + exp(x))`.
///
/// Numerically stable via: `sigmoid(x) = 1 / (1 + exp(-x))` for positive x,
/// `sigmoid(x) = exp(x) / (1 + exp(x))` for negative x.
#[inline]
pub fn simd_sigmoid<S: SimdFloat>(x: S) -> S {
    // Use: sigmoid(x) = 0.5 + 0.5 * tanh(x/2)
    // Or direct: exp(x) / (1 + exp(x))
    //
    // For stability, we compute:
    //   if x >= 0: 1 / (1 + exp(-x))
    //   if x <  0: exp(x) / (1 + exp(x))
    //
    // Generic SIMD version using blend:
    let neg_abs_x = x.abs().neg(); // always <= 0
    let exp_neg = simd_exp(neg_abs_x); // always in (0, 1]
    let denom = S::one().add(exp_neg);
    let sig_pos = S::one().div(denom); // for x >= 0
    let sig_neg = exp_neg.div(denom); // for x < 0

    let mask = x.cmp_ge(S::zero());
    S::blend(mask, sig_pos, sig_neg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::generic::ScalarFloat;

    #[test]
    fn test_sigmoid_accuracy() {
        let n = 1_000_000;
        let mut max_abs_err: f32 = 0.0;

        for i in 0..n {
            let x = -10.0 + 20.0 * (i as f32) / (n as f32);
            let expected = 1.0 / (1.0 + (-x).exp());
            let result = simd_sigmoid(ScalarFloat(x)).0;

            let abs_err = (result - expected).abs();
            max_abs_err = max_abs_err.max(abs_err);
        }

        assert!(
            max_abs_err < 1e-5,
            "sigmoid max abs error {max_abs_err} exceeds 1e-5"
        );
    }

    #[test]
    fn test_sigmoid_properties() {
        // sigmoid(0) = 0.5
        let r = simd_sigmoid(ScalarFloat(0.0)).0;
        assert!((r - 0.5).abs() < 1e-6, "sigmoid(0) = {r}");

        // sigmoid(-x) = 1 - sigmoid(x)
        for &x in &[1.0, 2.0, 5.0, -3.0] {
            let s_pos = simd_sigmoid(ScalarFloat(x)).0;
            let s_neg = simd_sigmoid(ScalarFloat(-x)).0;
            assert!(
                (s_pos + s_neg - 1.0).abs() < 1e-5,
                "sigmoid({x}) + sigmoid({}) = {}",
                -x,
                s_pos + s_neg
            );
        }

        // Large positive → 1
        let r = simd_sigmoid(ScalarFloat(50.0)).0;
        assert!((r - 1.0).abs() < 1e-5, "sigmoid(50) = {r}");

        // Large negative → 0
        let r = simd_sigmoid(ScalarFloat(-50.0)).0;
        assert!(r.abs() < 1e-5, "sigmoid(-50) = {r}");
    }
}
