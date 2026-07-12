/// Generic Newton-Raphson solver for nonlinear circuit equations.
///
/// Used by tubes, diodes, transistors, and other nonlinear devices.
/// Each device provides its own `f(x)` and `f'(x)` via the callback.
pub struct NewtonSolver {
    /// Maximum iterations before giving up.
    pub max_iterations: u32,
    /// Convergence tolerance.
    pub tolerance: f32,
    /// Voltage clamp range: [-clamp, +clamp].
    pub voltage_clamp: f32,
}

impl Default for NewtonSolver {
    fn default() -> Self {
        Self {
            max_iterations: 4,
            tolerance: 1e-4,
            voltage_clamp: 500.0,
        }
    }
}

impl NewtonSolver {
    /// Solve `F(x) = 0` starting from `x0`.
    ///
    /// `eval` takes `x` and returns `(F(x), F'(x))`.
    /// Returns the converged value of `x`.
    #[inline]
    pub fn solve<F>(&self, mut x: f32, eval: F) -> f32
    where
        F: Fn(f32) -> (f32, f32),
    {
        for _ in 0..self.max_iterations {
            let (f_val, f_deriv) = eval(x);

            // Check convergence
            if f_val.abs() < self.tolerance {
                break;
            }

            // Guard against zero derivative
            let deriv = if f_deriv.abs() < 1e-12 {
                if f_deriv >= 0.0 { 1e-12 } else { -1e-12 }
            } else {
                f_deriv
            };

            // Newton step
            let dx = f_val / deriv;

            // Limit step size to prevent wild oscillations
            let max_step = 50.0;
            let dx = dx.clamp(-max_step, max_step);

            x -= dx;

            // Voltage clamping
            x = x.clamp(-self.voltage_clamp, self.voltage_clamp);
        }

        x
    }

    /// Solve returning the iteration count (for diagnostics).
    #[inline]
    pub fn solve_counted<F>(&self, mut x: f32, eval: F) -> (f32, u32)
    where
        F: Fn(f32) -> (f32, f32),
    {
        let mut iters = 0;
        for i in 0..self.max_iterations {
            let (f_val, f_deriv) = eval(x);
            iters = i + 1;

            if f_val.abs() < self.tolerance {
                break;
            }

            let deriv = if f_deriv.abs() < 1e-12 {
                if f_deriv >= 0.0 { 1e-12 } else { -1e-12 }
            } else {
                f_deriv
            };

            let dx = (f_val / deriv).clamp(-50.0, 50.0);
            x = (x - dx).clamp(-self.voltage_clamp, self.voltage_clamp);
        }

        (x, iters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_linear() {
        // Solve f(x) = x - 5 = 0 => x = 5
        let solver = NewtonSolver::default();
        let result = solver.solve(0.0, |x| (x - 5.0, 1.0));
        assert!((result - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_solve_quadratic() {
        // Solve f(x) = x^2 - 4 = 0 => x = 2 (starting from 1)
        let solver = NewtonSolver::default();
        let result = solver.solve(1.0, |x| (x * x - 4.0, 2.0 * x));
        assert!((result - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_voltage_clamping() {
        let solver = NewtonSolver {
            voltage_clamp: 10.0,
            ..Default::default()
        };
        // Function that would send x to infinity
        let result = solver.solve(0.0, |_x| (1.0, 1e-15));
        assert!(result.abs() <= 10.0);
    }

    #[test]
    fn test_convergence_count() {
        let solver = NewtonSolver::default();
        let (result, iters) = solver.solve_counted(1.0, |x| (x * x - 4.0, 2.0 * x));
        assert!((result - 2.0).abs() < 1e-4);
        assert!(iters <= 4, "took {iters} iterations");
    }
}
