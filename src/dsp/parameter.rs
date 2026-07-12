/// One-pole smoothed parameter for real-time control.
///
/// Smoothing equation: `value += coeff * (target - value)`
///
/// This prevents zipper noise when parameters change.
#[derive(Clone, Debug)]
pub struct Parameter {
    /// Current smoothed value.
    pub value: f32,
    /// Target value to smooth toward.
    pub target: f32,
    /// Smoothing coefficient (0..1). Smaller = slower smoothing.
    pub smoothing_coeff: f32,
}

impl Parameter {
    /// Create a new parameter with the given initial value.
    pub fn new(initial: f32) -> Self {
        Self {
            value: initial,
            target: initial,
            smoothing_coeff: 0.005,
        }
    }

    /// Create with a specific smoothing coefficient.
    pub fn with_smoothing(initial: f32, coeff: f32) -> Self {
        Self {
            value: initial,
            target: initial,
            smoothing_coeff: coeff,
        }
    }

    /// Compute the smoothing coefficient from a time constant in seconds.
    pub fn coeff_from_time_constant(time_constant_secs: f32, sample_rate: f32) -> f32 {
        if time_constant_secs <= 0.0 {
            return 1.0;
        }
        1.0 - (-1.0 / (time_constant_secs * sample_rate)).exp()
    }

    /// Set a new target value.
    #[inline]
    pub fn set(&mut self, target: f32) {
        self.target = target;
    }

    /// Reset to a value immediately (no smoothing).
    #[inline]
    pub fn reset(&mut self, value: f32) {
        self.value = value;
        self.target = value;
    }

    /// Advance one sample of smoothing.
    #[inline]
    pub fn step(&mut self) {
        self.value += self.smoothing_coeff * (self.target - self.value);
    }

    /// Check if the parameter has reached its target (within tolerance).
    #[inline]
    pub fn is_settled(&self) -> bool {
        (self.value - self.target).abs() < 1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_smoothing() {
        let mut p = Parameter::with_smoothing(0.0, 0.1);
        p.set(1.0);

        // After many steps, should approach target
        for _ in 0..1000 {
            p.step();
        }
        assert!((p.value - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_parameter_reset() {
        let mut p = Parameter::new(0.0);
        p.reset(5.0);
        assert_eq!(p.value, 5.0);
        assert_eq!(p.target, 5.0);
    }

    #[test]
    fn test_parameter_monotonic() {
        let mut p = Parameter::with_smoothing(0.0, 0.01);
        p.set(1.0);

        let mut prev = 0.0;
        for _ in 0..100 {
            p.step();
            assert!(p.value >= prev, "parameter decreased");
            prev = p.value;
        }
    }
}
