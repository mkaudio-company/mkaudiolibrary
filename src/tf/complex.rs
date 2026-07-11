//! Minimal complex number type for the `tf` module.
//!
//! Hand-rolled rather than pulling in `num-complex`, matching the rest of
//! this crate's preference for small, dependency-free primitives.

use std::ops::{Add, Mul, Sub};

/// A complex number with `f64` components.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[inline]
    pub fn scale(self, k: f64) -> Self {
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
        let c = Complex64::from_polar(2.0, std::f64::consts::FRAC_PI_4);
        assert!((c.norm() - 2.0).abs() < 1e-12);
        assert!((c.arg() - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn mul_matches_definition() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, -1.0);
        let product = a * b;
        assert_eq!(
            product,
            Complex64::new(1.0 * 3.0 - 2.0 * -1.0, 1.0 * -1.0 + 2.0 * 3.0)
        );
    }
}
