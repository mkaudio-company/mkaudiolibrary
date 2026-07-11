//! Discrete Cosine Transform (type II / III, the "the" DCT and its inverse).
//!
//! Naive `O(n^2)` direct-sum implementation, SIMD-accelerated per-output
//! dot products (via [`crate::simd::dot`]) - the same approach as
//! [`crate::tf::fft::dft`].

use std::f64::consts::PI;

/// DCT-II. `idct2(dct2(x))` recovers `x` exactly (up to floating point).
pub fn dct2(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut output = vec![0.0; n];
    let mut table = vec![0.0; n];

    for (k, out) in output.iter_mut().enumerate() {
        for (t, slot) in table.iter_mut().enumerate() {
            *slot = (PI / n as f64 * (t as f64 + 0.5) * k as f64).cos();
        }
        *out = crate::simd::dot(input, &table);
    }

    output
}

/// DCT-III, scaled to be the exact inverse of [`dct2`]. Unnormalized DCT-III
/// (`X[0]/2 + sum_{k>=1} X[k]cos(...)`) is the inverse of unnormalized
/// DCT-II up to a factor of `N/2` (verified directly for N=2 - the naive
/// "divide by N" scaling is off by exactly 2x), so this scales by `2/N`.
pub fn idct2(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut output = vec![0.0; n];
    let mut table = vec![0.0; n];

    for (t, out) in output.iter_mut().enumerate() {
        table[0] = 0.5;
        for (k, slot) in table.iter_mut().enumerate().skip(1) {
            *slot = (PI / n as f64 * (t as f64 + 0.5) * k as f64).cos();
        }
        *out = crate::simd::dot(input, &table) * 2.0 / n as f64;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct2_of_dct2_is_identity() {
        let input: Vec<f64> = (0..40).map(|i| (i as f64 * 0.21).sin() + 0.3).collect();
        let roundtrip = idct2(&dct2(&input));
        for (a, b) in input.iter().zip(roundtrip.iter()) {
            assert!((a - b).abs() < 1e-9, "{} vs {}", a, b);
        }
    }
}
