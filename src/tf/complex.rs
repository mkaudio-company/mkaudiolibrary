//! Minimal complex number type for the `tf` module.
//!
//! Hand-rolled rather than pulling in `num-complex`, matching the rest of
//! this crate's preference for small, dependency-free primitives.

use std::ops::{Add, Mul, Sub};

/// A complex number with `f32` components.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex64 {
    /// Construct from real and imaginary parts.
    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Construct from polar form: magnitude `r` and phase `theta` (radians).
    #[inline]
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex conjugate (`re - i*im`).
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Magnitude (Euclidean norm) `sqrt(re^2 + im^2)`.
    #[inline]
    pub fn norm(self) -> f32 {
        self.re.hypot(self.im)
    }

    /// Squared magnitude `re^2 + im^2`, cheaper than [`Complex64::norm`]
    /// when only relative magnitudes matter.
    #[inline]
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle in radians, `atan2(im, re)`.
    #[inline]
    pub fn arg(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Multiply both components by a real scalar `k`.
    #[inline]
    pub fn scale(self, k: f32) -> Self {
        Self {
            re: self.re * k,
            im: self.im * k,
        }
    }
}

impl Add for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polar_roundtrip() {
        let c = Complex64::from_polar(2.0, std::f32::consts::FRAC_PI_4);
        assert!((c.norm() - 2.0).abs() < 1e-6);
        assert!((c.arg() - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    }

    #[test]
    fn mul_matches_definition() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, -1.0);
        let product = a * b;
        assert_eq!(product, Complex64::new(1.0 * 3.0 - -2.0, -1.0 + 2.0 * 3.0));
    }
}
