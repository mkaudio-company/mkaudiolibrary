// Runtime SIMD dispatch — selects the best available backend.
//
// Currently returns the compile-time selected backend.
// Platform-specific backends use `is_x86_feature_detected!` etc.
// at initialization time, then store the result in a static.

/// Query which SIMD backend is available at runtime.
pub fn detect_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "sim-avx512")]
        if is_x86_feature_detected!("avx512f") {
            return SimdBackend::Avx512;
        }

        #[cfg(feature = "sim-avx2")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdBackend::Avx2;
        }
    }

    // NEON is always available on aarch64
    #[cfg(all(target_arch = "aarch64", feature = "sim-neon"))]
    return SimdBackend::Neon;

    #[cfg(not(all(target_arch = "aarch64", feature = "sim-neon")))]
    SimdBackend::Scalar
}

/// A SIMD backend detected/selected by [`detect_backend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// No hardware SIMD; one lane at a time.
    Scalar,
    /// x86_64 AVX2 (8 `f32` lanes).
    Avx2,
    /// x86_64 AVX-512 (16 `f32` lanes).
    Avx512,
    /// aarch64 NEON (4 `f32` lanes).
    Neon,
}

impl SimdBackend {
    /// Number of `f32` lanes this backend processes at once.
    pub fn width(self) -> usize {
        match self {
            SimdBackend::Scalar => 1,
            SimdBackend::Avx2 => 8,
            SimdBackend::Avx512 => 16,
            SimdBackend::Neon => 4,
        }
    }
}
