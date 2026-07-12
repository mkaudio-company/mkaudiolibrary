//! Discrete/Fast Fourier Transform.
//!
//! - [`dft`] - naive `O(n^2)` transform, SIMD-accelerated per-bin dot
//!   products (via [`crate::simd::dot`]); useful as a reference and for
//!   small/prime-length inputs.
//! - [`fft`]/[`ifft`] - `O(n log n)`, iterative radix-2 Cooley-Tukey for
//!   power-of-2 lengths, falling back to Bluestein's algorithm (chirp-z,
//!   itself built on the radix-2 core) for arbitrary lengths.
//! - [`rfft`]/[`irfft`] - real-input/real-output specializations that skip
//!   computing the conjugate-symmetric half of the spectrum.

use std::f32::consts::PI;

use super::complex::Complex64;

/// Naive `O(n^2)` DFT. Correct for any length; prefer [`fft`] for anything
/// beyond a few hundred samples.
pub fn dft(input: &[f32]) -> Vec<Complex64> {
    let n = input.len();
    let mut output = Vec::with_capacity(n);
    let mut cos_table = vec![0.0; n];
    let mut sin_table = vec![0.0; n];

    for k in 0..n {
        for t in 0..n {
            let angle = -2.0 * PI * ((k * t) % n) as f32 / n as f32;
            cos_table[t] = angle.cos();
            sin_table[t] = angle.sin();
        }
        output.push(Complex64::new(
            crate::simd::dot(input, &cos_table),
            crate::simd::dot(input, &sin_table),
        ));
    }

    output
}

fn reverse_bits(mut x: usize, bits: u32) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place iterative radix-2 Cooley-Tukey FFT. `input.len()` must be a power of 2.
fn fft_radix2(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    let mut a = input.to_vec();
    if n <= 1 {
        return a;
    }

    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = reverse_bits(i, bits);
        if j > i {
            a.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let angle_step = -2.0 * PI / len as f32;
        let w_len = Complex64::from_polar(1.0, angle_step);
        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..len / 2 {
                let u = a[i + j];
                let v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w = w * w_len;
            }
            i += len;
        }
        len <<= 1;
    }

    a
}

/// Bluestein's algorithm (chirp-z transform): reduces an arbitrary-length
/// DFT to a power-of-2 convolution, computed via [`fft_radix2`].
fn bluestein_fft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let m = (2 * n - 1).next_power_of_two();

    let mut chirp = vec![Complex64::new(0.0, 0.0); n];
    for (k, c) in chirp.iter_mut().enumerate() {
        let kk = (k as u128 * k as u128) % (2 * n as u128);
        let angle = -PI * kk as f32 / n as f32;
        *c = Complex64::from_polar(1.0, angle);
    }

    let mut a = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..n {
        a[k] = input[k] * chirp[k];
    }

    let mut b = vec![Complex64::new(0.0, 0.0); m];
    b[0] = chirp[0].conj();
    for k in 1..n {
        b[k] = chirp[k].conj();
        b[m - k] = chirp[k].conj();
    }

    let fa = fft_radix2(&a);
    let fb = fft_radix2(&b);
    let fc: Vec<Complex64> = fa.iter().zip(fb.iter()).map(|(&x, &y)| x * y).collect();
    let conv = ifft(&fc);

    (0..n).map(|k| conv[k] * chirp[k]).collect()
}

/// Forward FFT: `O(n log n)` for power-of-2 lengths, `O(n log n)` (via
/// Bluestein's algorithm) for any other length.
pub fn fft(input: &[Complex64]) -> Vec<Complex64> {
    if input.is_empty() {
        return Vec::new();
    }
    if input.len().is_power_of_two() {
        fft_radix2(input)
    } else {
        bluestein_fft(input)
    }
}

/// Inverse FFT.
pub fn ifft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let conjugated: Vec<Complex64> = input.iter().map(|c| c.conj()).collect();
    let scale = 1.0 / n as f32;
    fft(&conjugated)
        .iter()
        .map(|c| c.conj().scale(scale))
        .collect()
}

/// Real-input FFT: returns only the first `n/2 + 1` bins (the rest are the
/// conjugate mirror of these, per the symmetry of a real-valued signal's spectrum).
pub fn rfft(input: &[f32]) -> Vec<Complex64> {
    let complex_input: Vec<Complex64> = input.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let full = fft(&complex_input);
    let half = input.len() / 2 + 1;
    full.into_iter().take(half).collect()
}

/// Inverse of [`rfft`]: reconstructs the full conjugate-symmetric spectrum
/// from its first `output_len/2 + 1` bins, then takes the real part of the
/// inverse FFT.
pub fn irfft(spectrum: &[Complex64], output_len: usize) -> Vec<f32> {
    if output_len == 0 {
        return Vec::new();
    }

    let mut full = vec![Complex64::new(0.0, 0.0); output_len];
    let half = (output_len / 2 + 1).min(spectrum.len());
    full[..half].copy_from_slice(&spectrum[..half]);
    for k in (output_len / 2 + 1)..output_len {
        full[k] = full[output_len - k].conj();
    }

    ifft(&full).iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_complex(a: &[Complex64], b: &[Complex64], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x.re - y.re).abs() < tol,
                "re mismatch: {} vs {}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < tol,
                "im mismatch: {} vs {}",
                x.im,
                y.im
            );
        }
    }

    #[test]
    fn fft_matches_dft_power_of_two() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7).sin()).collect();
        let expected = dft(&input);
        let complex_input: Vec<Complex64> = input.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let actual = fft(&complex_input);
        approx_eq_complex(&expected, &actual, 1e-4);
    }

    #[test]
    fn fft_matches_dft_arbitrary_length() {
        // 13 is prime, forcing the Bluestein path.
        let input: Vec<f32> = (0..13).map(|i| (i as f32 * 1.3).cos()).collect();
        let expected = dft(&input);
        let complex_input: Vec<Complex64> = input.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let actual = fft(&complex_input);
        approx_eq_complex(&expected, &actual, 1e-4);
    }

    #[test]
    fn ifft_of_fft_is_identity() {
        let input: Vec<Complex64> = (0..37)
            .map(|i| Complex64::new((i as f32).sin(), (i as f32 * 0.5).cos()))
            .collect();
        let roundtrip = ifft(&fft(&input));
        approx_eq_complex(&input, &roundtrip, 1e-4);
    }

    #[test]
    fn irfft_of_rfft_is_identity() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin() * 0.5).collect();
        let spectrum = rfft(&input);
        let roundtrip = irfft(&spectrum, input.len());
        for (a, b) in input.iter().zip(roundtrip.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
