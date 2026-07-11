//! SIMD-accelerated primitives for hot DSP loops.
//!
//! Enabled via the `simd` Cargo feature. On `x86_64` this dispatches at
//! runtime to AVX2+FMA when available, falling back to baseline SSE2
//! (guaranteed present on every x86_64 target). On `aarch64` this uses NEON
//! directly, since double-precision NEON is mandatory in the AArch64
//! specification and needs no runtime detection. Every primitive also has a
//! plain scalar implementation used when the `simd` feature is disabled or
//! the target architecture has no vectorized path here.
//!
//! All functions operate on the shared length of their input slices; any
//! trailing elements beyond a full SIMD-width chunk are handled by a scalar
//! remainder loop so callers never need to pad buffers to a particular width.

// ==========================================
// Dot product - used by FIR convolution
// ==========================================

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);

    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { dot_avx2_fma(a, b) }
    } else {
        unsafe { dot_sse2(a, b) }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    unsafe {
        let len = a.len();
        let chunks = len / 4;

        let mut acc = _mm256_setzero_pd();
        for i in 0..chunks {
            let va = _mm256_loadu_pd(a.as_ptr().add(i * 4));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i * 4));
            acc = _mm256_fmadd_pd(va, vb, acc);
        }

        let mut lanes = [0.0f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
        let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

        for i in (chunks * 4)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn dot_sse2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    unsafe {
        let len = a.len();
        let chunks = len / 2;

        let mut acc = _mm_setzero_pd();
        for i in 0..chunks {
            let va = _mm_loadu_pd(a.as_ptr().add(i * 2));
            let vb = _mm_loadu_pd(b.as_ptr().add(i * 2));
            acc = _mm_add_pd(acc, _mm_mul_pd(va, vb));
        }

        let mut lanes = [0.0f64; 2];
        _mm_storeu_pd(lanes.as_mut_ptr(), acc);
        let mut sum = lanes[0] + lanes[1];

        for i in (chunks * 2)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    unsafe { dot_neon(&a[..len], &b[..len]) }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
unsafe fn dot_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::*;
    unsafe {
        let len = a.len();
        let chunks = len / 2;

        let mut acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let va = vld1q_f64(a.as_ptr().add(i * 2));
            let vb = vld1q_f64(b.as_ptr().add(i * 2));
            acc = vfmaq_f64(acc, va, vb);
        }

        let mut sum = vaddvq_f64(acc);
        for i in (chunks * 2)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Sum of `a[i] * b[i]` over the shared length of `a` and `b`.
#[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

// ==========================================
// Elementwise multiply - used to apply a gain envelope to a buffer
// ==========================================

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub fn mul_elementwise(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len().min(a.len()).min(b.len());
    let (dst, a, b) = (&mut dst[..len], &a[..len], &b[..len]);

    if std::is_x86_feature_detected!("avx2") {
        unsafe { mul_avx2(dst, a, b) }
    } else {
        unsafe { mul_sse2(dst, a, b) }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mul_avx2(dst: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::x86_64::*;
    unsafe {
        let len = dst.len();
        let chunks = len / 4;

        for i in 0..chunks {
            let va = _mm256_loadu_pd(a.as_ptr().add(i * 4));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i * 4));
            _mm256_storeu_pd(dst.as_mut_ptr().add(i * 4), _mm256_mul_pd(va, vb));
        }
        for i in (chunks * 4)..len {
            dst[i] = a[i] * b[i];
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mul_sse2(dst: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::x86_64::*;
    unsafe {
        let len = dst.len();
        let chunks = len / 2;

        for i in 0..chunks {
            let va = _mm_loadu_pd(a.as_ptr().add(i * 2));
            let vb = _mm_loadu_pd(b.as_ptr().add(i * 2));
            _mm_storeu_pd(dst.as_mut_ptr().add(i * 2), _mm_mul_pd(va, vb));
        }
        for i in (chunks * 2)..len {
            dst[i] = a[i] * b[i];
        }
    }
}

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub fn mul_elementwise(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len().min(a.len()).min(b.len());
    unsafe { mul_neon(&mut dst[..len], &a[..len], &b[..len]) }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
unsafe fn mul_neon(dst: &mut [f64], a: &[f64], b: &[f64]) {
    use std::arch::aarch64::*;
    unsafe {
        let len = dst.len();
        let chunks = len / 2;

        for i in 0..chunks {
            let va = vld1q_f64(a.as_ptr().add(i * 2));
            let vb = vld1q_f64(b.as_ptr().add(i * 2));
            vst1q_f64(dst.as_mut_ptr().add(i * 2), vmulq_f64(va, vb));
        }
        for i in (chunks * 2)..len {
            dst[i] = a[i] * b[i];
        }
    }
}

/// `dst[i] = a[i] * b[i]` over the shared length of the three slices.
#[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[inline]
pub fn mul_elementwise(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len().min(a.len()).min(b.len());
    for i in 0..len {
        dst[i] = a[i] * b[i];
    }
}

// ==========================================
// Scalar-mix (wet/dry crossfade) - used by Delay
// ==========================================

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub fn mix_scalar(dst: &mut [f64], dry: &[f64], wet: &[f64], mix: f64) {
    let len = dst.len().min(dry.len()).min(wet.len());
    let (dst, dry, wet) = (&mut dst[..len], &dry[..len], &wet[..len]);

    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { mix_avx2_fma(dst, dry, wet, mix) }
    } else {
        unsafe { mix_sse2(dst, dry, wet, mix) }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn mix_avx2_fma(dst: &mut [f64], dry: &[f64], wet: &[f64], mix: f64) {
    use std::arch::x86_64::*;
    unsafe {
        let len = dst.len();
        let chunks = len / 4;
        let vmix = _mm256_set1_pd(mix);
        let vinv = _mm256_set1_pd(1.0 - mix);

        for i in 0..chunks {
            let vdry = _mm256_loadu_pd(dry.as_ptr().add(i * 4));
            let vwet = _mm256_loadu_pd(wet.as_ptr().add(i * 4));
            let dry_term = _mm256_mul_pd(vdry, vinv);
            let result = _mm256_fmadd_pd(vwet, vmix, dry_term);
            _mm256_storeu_pd(dst.as_mut_ptr().add(i * 4), result);
        }
        for i in (chunks * 4)..len {
            dst[i] = dry[i] * (1.0 - mix) + wet[i] * mix;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mix_sse2(dst: &mut [f64], dry: &[f64], wet: &[f64], mix: f64) {
    use std::arch::x86_64::*;
    unsafe {
        let len = dst.len();
        let chunks = len / 2;
        let vmix = _mm_set1_pd(mix);
        let vinv = _mm_set1_pd(1.0 - mix);

        for i in 0..chunks {
            let vdry = _mm_loadu_pd(dry.as_ptr().add(i * 2));
            let vwet = _mm_loadu_pd(wet.as_ptr().add(i * 2));
            let result = _mm_add_pd(_mm_mul_pd(vdry, vinv), _mm_mul_pd(vwet, vmix));
            _mm_storeu_pd(dst.as_mut_ptr().add(i * 2), result);
        }
        for i in (chunks * 2)..len {
            dst[i] = dry[i] * (1.0 - mix) + wet[i] * mix;
        }
    }
}

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub fn mix_scalar(dst: &mut [f64], dry: &[f64], wet: &[f64], mix: f64) {
    let len = dst.len().min(dry.len()).min(wet.len());
    unsafe { mix_neon(&mut dst[..len], &dry[..len], &wet[..len], mix) }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
unsafe fn mix_neon(dst: &mut [f64], dry: &[f64], wet: &[f64], mix: f64) {
    use std::arch::aarch64::*;
    unsafe {
        let len = dst.len();
        let chunks = len / 2;
        let vmix = vdupq_n_f64(mix);
        let vinv = vdupq_n_f64(1.0 - mix);

        for i in 0..chunks {
            let vdry = vld1q_f64(dry.as_ptr().add(i * 2));
            let vwet = vld1q_f64(wet.as_ptr().add(i * 2));
            let result = vfmaq_f64(vmulq_f64(vdry, vinv), vwet, vmix);
            vst1q_f64(dst.as_mut_ptr().add(i * 2), result);
        }
        for i in (chunks * 2)..len {
            dst[i] = dry[i] * (1.0 - mix) + wet[i] * mix;
        }
    }
}

/// `dst[i] = dry[i] * (1 - mix) + wet[i] * mix` over the shared length of the three slices.
#[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[inline]
pub fn mix_scalar(dst: &mut [f64], dry: &[f64], wet: &[f64], mix: f64) {
    let len = dst.len().min(dry.len()).min(wet.len());
    for i in 0..len {
        dst[i] = dry[i] * (1.0 - mix) + wet[i] * mix;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f64> = (0..37).map(|i| i as f64 * 0.5).collect();
        let b: Vec<f64> = (0..37).map(|i| (i as f64 * 0.3).sin()).collect();

        let mut expected = 0.0;
        for i in 0..a.len() {
            expected += a[i] * b[i];
        }

        assert!((dot(&a, &b) - expected).abs() < 1e-9);
    }

    #[test]
    fn mul_elementwise_matches_scalar() {
        let a: Vec<f64> = (0..23).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..23).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let mut dst = vec![0.0; 23];

        mul_elementwise(&mut dst, &a, &b);

        for i in 0..23 {
            assert!((dst[i] - a[i] * b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn mix_scalar_matches_manual() {
        let dry: Vec<f64> = (0..19).map(|i| i as f64).collect();
        let wet: Vec<f64> = (0..19).map(|i| -(i as f64)).collect();
        let mut dst = vec![0.0; 19];

        mix_scalar(&mut dst, &dry, &wet, 0.25);

        for i in 0..19 {
            let expected = dry[i] * 0.75 + wet[i] * 0.25;
            assert!((dst[i] - expected).abs() < 1e-12);
        }
    }
}
