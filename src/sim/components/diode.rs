//! Diode clipper stages (single and anti-parallel pairs) via the Shockley
//! diode equation solved with Newton-Raphson.

use crate::dsp::parameter::Parameter;
use crate::sim::components::CircuitComponent;
use crate::sim::core::solver::NewtonSolver;
use crate::simd::generic::{ScalarFloat, simd_exp};

/// Diode parameters for the Shockley equation: I = Is * (exp(V/(n*Vt)) - 1).
#[derive(Clone, Debug)]
pub struct DiodeParams {
    /// Saturation current (amps).
    pub is: f32,
    /// Emission coefficient.
    pub n: f32,
    /// Thermal voltage (~0.0258V at room temperature).
    pub vt: f32,
}

/// Standard silicon diode (1N4148-like).
pub const DIODE_SILICON: DiodeParams = DiodeParams {
    is: 1e-12,
    n: 1.9,
    vt: 0.0258,
};

/// Germanium diode (lower forward voltage).
pub const DIODE_GERMANIUM: DiodeParams = DiodeParams {
    is: 1e-6,
    n: 1.2,
    vt: 0.0258,
};

/// LED (higher forward voltage, very low saturation current).
pub const DIODE_LED: DiodeParams = DiodeParams {
    is: 1e-20,
    n: 2.0,
    vt: 0.0258,
};

// ---------------------------------------------------------------------------
// DiodeClipper
// ---------------------------------------------------------------------------

/// Single diode clipper circuit.
///
/// Circuit topology: input voltage source → resistor R → diode to ground.
///
/// Solves `F(Vd) = (Vin - Vd)/R - Is*(exp(Vd/(n*Vt)) - 1) = 0` per sample
/// using Newton-Raphson.
pub struct DiodeClipper {
    params: DiodeParams,
    solver: NewtonSolver,
    /// Load resistance (ohms).
    resistance: f32,
    /// Smoothed threshold parameter (voltage scale).
    threshold: Parameter,
    /// Previous diode voltage — used as Newton initial guess.
    prev_voltage: f32,
    sample_rate: f32,
}

impl DiodeClipper {
    /// Create a new diode clipper with the given diode model and load resistance.
    pub fn new(params: DiodeParams, resistance: f32) -> Self {
        Self {
            params,
            solver: NewtonSolver {
                max_iterations: 4,
                tolerance: 1e-4,
                voltage_clamp: 500.0,
            },
            resistance,
            threshold: Parameter::new(1.0),
            prev_voltage: 0.0,
            sample_rate: 44100.0,
        }
    }

    /// Set the threshold parameter (scales the input voltage range).
    pub fn set_threshold(&mut self, value: f32) {
        self.threshold.set(value);
    }

    /// Evaluate the Shockley diode current: I = Is * (exp(V/(n*Vt)) - 1).
    #[inline]
    #[cfg(test)]
    fn diode_current(&self, vd: f32) -> f32 {
        let exponent = vd / (self.params.n * self.params.vt);
        let exp_val = simd_exp(ScalarFloat(exponent)).0;
        self.params.is * (exp_val - 1.0)
    }

    /// Solve for the diode voltage given an input voltage.
    ///
    /// F(Vd) = (Vin - Vd)/R - Is*(exp(Vd/(n*Vt)) - 1) = 0
    /// F'(Vd) = -1/R - Is/(n*Vt) * exp(Vd/(n*Vt))
    #[inline]
    fn solve_diode_voltage(&mut self, vin: f32) -> f32 {
        let r = self.resistance;
        let is = self.params.is;
        let n_vt = self.params.n * self.params.vt;
        let initial_guess = self.prev_voltage;

        let vd = self.solver.solve(initial_guess, |vd| {
            let exponent = vd / n_vt;
            let exp_val = simd_exp(ScalarFloat(exponent)).0;

            // F(Vd) = (Vin - Vd)/R - Is*(exp(Vd/(n*Vt)) - 1)
            let f = (vin - vd) / r - is * (exp_val - 1.0);
            // F'(Vd) = -1/R - Is/(n*Vt) * exp(Vd/(n*Vt))
            let f_prime = -1.0 / r - is / n_vt * exp_val;

            (f, f_prime)
        });

        self.prev_voltage = vd;
        vd
    }
}

impl CircuitComponent for DiodeClipper {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.prev_voltage = 0.0;
        self.threshold.reset(1.0);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());

        for i in 0..len {
            self.threshold.step();

            // Scale input to voltage range (threshold controls the voltage swing)
            let vin = input[i] * self.threshold.value;

            // Solve for diode voltage
            let vd = self.solve_diode_voltage(vin);

            // Normalize output back to audio range
            let out = if self.threshold.value.abs() > 1e-12 {
                vd / self.threshold.value
            } else {
                0.0
            };
            output[i] = out.clamp(-1.0, 1.0);
        }
    }

    fn update_parameters(&mut self) {
        // Parameters are smoothed per-sample in process_block
    }
}

// ---------------------------------------------------------------------------
// AntiParallelDiodeClipper
// ---------------------------------------------------------------------------

/// Anti-parallel diode clipper (two diodes in opposite directions).
///
/// Produces symmetric clipping. The total diode current is:
/// `I_total = 2*Is*sinh(Vd/(n*Vt))`
///
/// Solves `F(Vd) = (Vin - Vd)/R - 2*Is*sinh(Vd/(n*Vt)) = 0`.
pub struct AntiParallelDiodeClipper {
    params: DiodeParams,
    solver: NewtonSolver,
    /// Load resistance (ohms).
    resistance: f32,
    /// Smoothed threshold parameter (voltage scale).
    threshold: Parameter,
    /// Previous diode voltage — used as Newton initial guess.
    prev_voltage: f32,
    sample_rate: f32,
}

impl AntiParallelDiodeClipper {
    /// Create a new anti-parallel diode clipper.
    pub fn new(params: DiodeParams, resistance: f32) -> Self {
        Self {
            params,
            solver: NewtonSolver {
                max_iterations: 4,
                tolerance: 1e-4,
                voltage_clamp: 500.0,
            },
            resistance,
            threshold: Parameter::new(1.0),
            prev_voltage: 0.0,
            sample_rate: 44100.0,
        }
    }

    /// Set the threshold parameter (scales the input voltage range).
    pub fn set_threshold(&mut self, value: f32) {
        self.threshold.set(value);
    }

    /// Solve for the diode voltage given an input voltage.
    ///
    /// F(Vd) = (Vin - Vd)/R - 2*Is*sinh(Vd/(n*Vt)) = 0
    /// F'(Vd) = -1/R - 2*Is/(n*Vt) * cosh(Vd/(n*Vt))
    ///
    /// sinh(x) = (exp(x) - exp(-x)) / 2
    /// cosh(x) = (exp(x) + exp(-x)) / 2
    #[inline]
    fn solve_diode_voltage(&mut self, vin: f32) -> f32 {
        let r = self.resistance;
        let is = self.params.is;
        let n_vt = self.params.n * self.params.vt;
        let initial_guess = self.prev_voltage;

        let vd = self.solver.solve(initial_guess, |vd| {
            let exponent = vd / n_vt;
            let exp_pos = simd_exp(ScalarFloat(exponent)).0;
            let exp_neg = simd_exp(ScalarFloat(-exponent)).0;

            let sinh_val = (exp_pos - exp_neg) * 0.5;
            let cosh_val = (exp_pos + exp_neg) * 0.5;

            // F(Vd) = (Vin - Vd)/R - 2*Is*sinh(Vd/(n*Vt))
            let f = (vin - vd) / r - 2.0 * is * sinh_val;
            // F'(Vd) = -1/R - 2*Is/(n*Vt) * cosh(Vd/(n*Vt))
            let f_prime = -1.0 / r - 2.0 * is / n_vt * cosh_val;

            (f, f_prime)
        });

        self.prev_voltage = vd;
        vd
    }
}

impl CircuitComponent for AntiParallelDiodeClipper {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.prev_voltage = 0.0;
        self.threshold.reset(1.0);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());

        for i in 0..len {
            self.threshold.step();

            let vin = input[i] * self.threshold.value;
            let vd = self.solve_diode_voltage(vin);

            let out = if self.threshold.value.abs() > 1e-12 {
                vd / self.threshold.value
            } else {
                0.0
            };
            output[i] = out.clamp(-1.0, 1.0);
        }
    }

    fn update_parameters(&mut self) {
        // Parameters are smoothed per-sample in process_block
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]
    use super::*;

    #[test]
    fn test_silicon_diode_iv_curve() {
        // Verify Shockley equation for a silicon diode.
        let clipper = DiodeClipper::new(DIODE_SILICON.clone(), 4700.0);

        for i in 0..100 {
            let vd = -0.5 + 1.5 * (i as f32) / 100.0;
            let current = clipper.diode_current(vd);

            // Current should be non-NaN and non-Inf
            assert!(!current.is_nan(), "I(Vd={vd}) is NaN");
            assert!(!current.is_infinite(), "I(Vd={vd}) is Inf");

            // For reverse bias (vd < 0), current should be very small
            if vd < -0.1 {
                assert!(
                    current.abs() < 1e-10,
                    "Reverse current at Vd={vd} is {current}, expected ~0"
                );
            }

            // For strong forward bias (vd > 0.8), current should be positive and growing
            if vd > 0.8 {
                assert!(
                    current > 1e-6,
                    "Forward current at Vd={vd} is {current}, expected significant"
                );
            }
        }
    }

    #[test]
    fn test_germanium_diode_iv_curve() {
        // Germanium has higher Is, so it turns on at lower voltages.
        let clipper = DiodeClipper::new(DIODE_GERMANIUM.clone(), 4700.0);

        // Germanium should conduct more at 0.3V than silicon does
        let current_ge = clipper.diode_current(0.3);
        let clipper_si = DiodeClipper::new(DIODE_SILICON.clone(), 4700.0);
        let current_si = clipper_si.diode_current(0.3);

        assert!(
            current_ge > current_si,
            "Germanium current ({current_ge}) should exceed silicon ({current_si}) at 0.3V"
        );
    }

    #[test]
    fn test_shockley_equation_matches() {
        // Directly verify Shockley equation: I = Is * (exp(V/(n*Vt)) - 1)
        let params = DIODE_SILICON.clone();
        let clipper = DiodeClipper::new(params.clone(), 4700.0);

        let vd = 0.5;
        let expected = params.is * ((vd / (params.n * params.vt)).exp() - 1.0);
        let actual = clipper.diode_current(vd);

        let rel_err = if expected.abs() > 1e-20 {
            (actual - expected).abs() / expected.abs()
        } else {
            (actual - expected).abs()
        };
        assert!(
            rel_err < 1e-4,
            "Shockley mismatch at {vd}V: expected={expected}, actual={actual}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_diode_clipper_sine() {
        let mut clipper = DiodeClipper::new(DIODE_SILICON.clone(), 4700.0);
        clipper.set_threshold(5.0); // 5V peak-to-peak input range
        clipper.prepare(44100.0);

        let block_size = 256;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        // Generate a 1kHz sine wave at full scale
        for i in 0..block_size {
            input[i] = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        clipper.process_block(&input, &mut output);

        // Verify: bounded, no NaN, non-silent
        let mut has_nonzero = false;
        for &s in &output {
            assert!(!s.is_nan(), "output contains NaN");
            assert!(!s.is_infinite(), "output contains Inf");
            assert!(s.abs() <= 1.0, "output {s} exceeds [-1, 1]");
            if s.abs() > 0.001 {
                has_nonzero = true;
            }
        }
        assert!(has_nonzero, "output is all zeros");

        // The diode should clip the positive half more than the negative half
        let max_pos = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_neg = output.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            max_pos < 1.0 || max_neg.abs() < 1.0,
            "expected clipping to reduce amplitude"
        );
    }

    #[test]
    fn test_anti_parallel_clipper_sine() {
        let mut clipper = AntiParallelDiodeClipper::new(DIODE_SILICON.clone(), 4700.0);
        clipper.set_threshold(5.0);
        clipper.prepare(44100.0);

        let block_size = 256;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        for i in 0..block_size {
            input[i] = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        clipper.process_block(&input, &mut output);

        // Verify: bounded, no NaN, non-silent
        let mut has_nonzero = false;
        for &s in &output {
            assert!(!s.is_nan(), "output contains NaN");
            assert!(!s.is_infinite(), "output contains Inf");
            assert!(s.abs() <= 1.0, "output {s} exceeds [-1, 1]");
            if s.abs() > 0.001 {
                has_nonzero = true;
            }
        }
        assert!(has_nonzero, "output is all zeros");
    }

    #[test]
    fn test_anti_parallel_clipper_symmetric() {
        // Anti-parallel diodes should produce symmetric clipping.
        let mut clipper = AntiParallelDiodeClipper::new(DIODE_SILICON.clone(), 4700.0);
        clipper.threshold.reset(5.0); // instant set (no smoothing ramp)
        clipper.prepare(44100.0);
        clipper.threshold.reset(5.0); // re-set after prepare resets it

        // Process several cycles so the solver reaches steady state
        let block_size = 2048;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        let freq = 440.0;
        for i in 0..block_size {
            input[i] = (2.0 * std::f32::consts::PI * freq * i as f32 / 44100.0).sin();
        }

        clipper.process_block(&input, &mut output);

        // Measure peaks from the second half (after settling)
        let half = block_size / 2;
        let max_pos = output[half..]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let max_neg = output[half..].iter().cloned().fold(f32::INFINITY, f32::min);

        // Symmetric: |max_pos| ≈ |max_neg|
        let asymmetry = (max_pos.abs() - max_neg.abs()).abs();
        assert!(
            asymmetry < 0.1,
            "anti-parallel clipping is asymmetric: pos={max_pos}, neg={max_neg}, asymmetry={asymmetry}"
        );
    }
}
