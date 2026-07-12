// Runtime SIMD dispatch — selects the best available backend.
//
// Considers every backend compiled in: unconditionally under the `simd`
// feature (this crate's own runtime-detected `x86_64`/`aarch64` layer,
// which must work correctly on whatever CPU the binary actually ends up
// running on), or under the matching `sim-avx2`/`sim-avx512`/`sim-neon`
// opt-in feature (`crate::sim`'s compile-time-selected layer, which
// assumes the build targets hardware known to support it - `sim`'s own
// callers don't invoke this function to make that choice, they just
// monomorphize against one backend directly, but `simd`'s callers do).

/// Query which SIMD backend is available at runtime, among those compiled in.
pub fn detect_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(any(feature = "simd", feature = "sim-avx512"))]
        if is_x86_feature_detected!("avx512f") {
            return SimdBackend::Avx512;
        }

        #[cfg(any(feature = "simd", feature = "sim-avx2"))]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdBackend::Avx2;
        }

        // SSE2 is guaranteed present on every x86_64 target, so no runtime
        // check is needed - it's the floor beneath AVX2/AVX-512 for the
        // `simd` feature's portable, runtime-detected layer specifically
        // (`sim` has no equivalent opt-in feature for it).
        #[cfg(feature = "simd")]
        return SimdBackend::Sse2;
    }

    // NEON is always available on aarch64
    #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "sim-neon")))]
    return SimdBackend::Neon;

    #[allow(unreachable_code)]
    SimdBackend::Scalar
}

/// A SIMD backend detected/selected by [`detect_backend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// No hardware SIMD; one lane at a time.
    Scalar,
    /// x86_64 baseline SSE2 (4 `f32` lanes).
    Sse2,
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
            SimdBackend::Sse2 => 4,
            SimdBackend::Avx2 => 8,
            SimdBackend::Avx512 => 16,
            SimdBackend::Neon => 4,
        }
    }

    /// Short identifying name (`"scalar"`, `"sse2"`, `"avx2"`, `"avx512"`, `"neon"`).
    pub fn as_str(self) -> &'static str {
        match self {
            SimdBackend::Scalar => "scalar",
            SimdBackend::Sse2 => "sse2",
            SimdBackend::Avx2 => "avx2",
            SimdBackend::Avx512 => "avx512",
            SimdBackend::Neon => "neon",
        }
    }
}
