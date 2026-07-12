//! Vacuum tube stages (triode/pentode) via Koren equations solved with
//! Newton-Raphson.

use crate::dsp::parameter::Parameter;
use crate::sim::components::CircuitComponent;
use crate::sim::core::solver::NewtonSolver;
use crate::simd::generic::{ScalarFloat, simd_log1pexp, simd_sigmoid};

/// Koren triode model parameters.
#[derive(Clone, Debug)]
pub struct TriodeParams {
    /// Amplification factor (mu).
    pub mu: f32,
    /// Scaling constant.
    pub k: f32,
    /// Curvature constant.
    pub vk: f32,
    /// Plate load resistance (ohms).
    pub plate_resistance: f32,
    /// Supply voltage B+ (volts).
    pub supply_voltage: f32,
    /// Grid bias voltage (volts, typically negative).
    pub bias: f32,
}

/// 12AX7 preset — the most common preamp triode in guitar amps.
pub const PARAMS_12AX7: TriodeParams = TriodeParams {
    mu: 100.0,
    k: 2.1e-6,
    vk: 1.3,
    plate_resistance: 100_000.0,
    supply_voltage: 250.0,
    bias: -1.5,
};

/// 12AT7 preset — lower gain, higher current.
pub const PARAMS_12AT7: TriodeParams = TriodeParams {
    mu: 60.0,
    k: 3.1e-6,
    vk: 1.5,
    plate_resistance: 80_000.0,
    supply_voltage: 250.0,
    bias: -2.0,
};

/// Koren triode amplifier stage.
///
/// Models a single triode with plate load resistor.
/// Circuit: B+ → Rp → Plate → Tube → Cathode → Ground.
///
/// Solves: `F(Vp) = B+ - Rp*Ip(Vg, Vp) - Vp = 0` using Newton-Raphson.
pub struct TriodeStage {
    params: TriodeParams,
    solver: NewtonSolver,
    /// Previous plate voltage — used as initial guess for Newton solver.
    prev_vp: f32,
    /// Input gain scaling (maps audio [-1,1] to grid voltage range).
    input_gain: Parameter,
    /// Output gain scaling.
    output_gain: Parameter,
    sample_rate: f32,
}

impl TriodeStage {
    /// Create a triode stage from the given Koren model parameters.
    pub fn new(params: TriodeParams) -> Self {
        let initial_vp = params.supply_voltage * 0.6; // reasonable starting point
        Self {
            params,
            solver: NewtonSolver {
                max_iterations: 4,
                tolerance: 1e-4,
                voltage_clamp: 500.0,
            },
            prev_vp: initial_vp,
            input_gain: Parameter::new(1.0),
            output_gain: Parameter::new(1.0),
            sample_rate: 44100.0,
        }
    }

    /// Compute plate current using Koren triode equation.
    /// `Ip = k * (log1pexp(E))^2` where `E = (Vg + Vp/mu) / Vk`
    #[inline]
    fn plate_current(&self, vg: f32, vp: f32) -> f32 {
        let e = (vg + vp / self.params.mu) / self.params.vk;
        let l = simd_log1pexp(ScalarFloat(e)).0;
        self.params.k * l * l
    }

    /// Compute derivative of plate current w.r.t. plate voltage.
    /// `dIp/dVp = (2*k*L / (mu*Vk)) * sigmoid(E)`
    #[inline]
    fn plate_current_derivative(&self, vg: f32, vp: f32) -> f32 {
        let e = (vg + vp / self.params.mu) / self.params.vk;
        let l = simd_log1pexp(ScalarFloat(e)).0;
        let sig = simd_sigmoid(ScalarFloat(e)).0;
        (2.0 * self.params.k * l) / (self.params.mu * self.params.vk) * sig
    }

    /// Solve for plate voltage given grid voltage.
    /// Solves: `F(Vp) = B+ - Rp*Ip(Vg, Vp) - Vp = 0`
    #[inline]
    fn solve_plate_voltage(&mut self, vg: f32) -> f32 {
        let b_plus = self.params.supply_voltage;
        let rp = self.params.plate_resistance;
        let initial_guess = self.prev_vp;

        let vp = self.solver.solve(initial_guess, |vp| {
            let ip = self.plate_current(vg, vp);
            let dip_dvp = self.plate_current_derivative(vg, vp);

            // F(Vp) = B+ - Rp*Ip - Vp
            let f = b_plus - rp * ip - vp;
            // F'(Vp) = -Rp * dIp/dVp - 1
            let f_prime = -rp * dip_dvp - 1.0;

            (f, f_prime)
        });

        self.prev_vp = vp;
        vp
    }
}

impl CircuitComponent for TriodeStage {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.prev_vp = self.params.supply_voltage * 0.6;
        self.input_gain.reset(1.0);
        self.output_gain.reset(1.0);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let bias = self.params.bias;
        let b_plus = self.params.supply_voltage;

        for i in 0..input.len().min(output.len()) {
            self.input_gain.step();
            self.output_gain.step();

            // Map input audio to grid voltage
            let vg = bias + input[i] * self.input_gain.value;

            // Solve for plate voltage
            let vp = self.solve_plate_voltage(vg);

            // Normalize output: map plate voltage to [-1, 1] range
            // Center around quiescent point and scale
            let vp_quiescent = b_plus * 0.6;
            let out = (vp - vp_quiescent) / (b_plus * 0.4) * self.output_gain.value;
            output[i] = out.clamp(-1.0, 1.0);
        }
    }

    fn update_parameters(&mut self) {
        // Parameters are smoothed per-sample in process_block
    }
}

/// Pentode model parameters.
#[derive(Clone, Debug)]
pub struct PentodeParams {
    /// Scaling constant.
    pub k: f32,
    /// Curvature constant.
    pub vk: f32,
    /// Plate resistance parameter (Early voltage).
    pub va: f32,
    /// Plate load resistance (ohms).
    pub plate_resistance: f32,
    /// Supply voltage B+ (volts).
    pub supply_voltage: f32,
    /// Grid bias voltage (volts).
    pub bias: f32,
}

/// EL34 preset — common power pentode.
pub const PARAMS_EL34: PentodeParams = PentodeParams {
    k: 1.8e-6,
    vk: 1.5,
    va: 300.0,
    plate_resistance: 2_000.0,
    supply_voltage: 450.0,
    bias: -30.0,
};

/// Pentode amplifier stage.
///
/// `Ip = k * (log1pexp(Vg/Vk))^2 * (1 + Vp/Va)`
pub struct PentodeStage {
    params: PentodeParams,
    solver: NewtonSolver,
    prev_vp: f32,
    input_gain: Parameter,
    output_gain: Parameter,
    sample_rate: f32,
}

impl PentodeStage {
    /// Create a pentode stage from the given model parameters.
    pub fn new(params: PentodeParams) -> Self {
        Self {
            params,
            solver: NewtonSolver {
                max_iterations: 4,
                tolerance: 1e-4,
                voltage_clamp: 500.0,
            },
            prev_vp: 300.0,
            input_gain: Parameter::new(1.0),
            output_gain: Parameter::new(1.0),
            sample_rate: 44100.0,
        }
    }

    /// Pentode plate current: `Ip = k * (log1pexp(Vg/Vk))^2 * (1 + Vp/Va)`
    #[inline]
    fn plate_current(&self, vg: f32, vp: f32) -> f32 {
        let e = vg / self.params.vk;
        let l = simd_log1pexp(ScalarFloat(e)).0;
        self.params.k * l * l * (1.0 + vp / self.params.va)
    }

    /// Derivative of pentode plate current w.r.t. Vp.
    /// `dIp/dVp = k * (log1pexp(Vg/Vk))^2 / Va`
    #[inline]
    fn plate_current_derivative_vp(&self, vg: f32, _vp: f32) -> f32 {
        let e = vg / self.params.vk;
        let l = simd_log1pexp(ScalarFloat(e)).0;
        self.params.k * l * l / self.params.va
    }

    #[inline]
    fn solve_plate_voltage(&mut self, vg: f32) -> f32 {
        let b_plus = self.params.supply_voltage;
        let rp = self.params.plate_resistance;
        let initial_guess = self.prev_vp;

        let vp = self.solver.solve(initial_guess, |vp| {
            let ip = self.plate_current(vg, vp);
            let dip_dvp = self.plate_current_derivative_vp(vg, vp);

            let f = b_plus - rp * ip - vp;
            let f_prime = -rp * dip_dvp - 1.0;

            (f, f_prime)
        });

        self.prev_vp = vp;
        vp
    }
}

impl CircuitComponent for PentodeStage {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.prev_vp = self.params.supply_voltage * 0.65;
        self.input_gain.reset(1.0);
        self.output_gain.reset(1.0);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let bias = self.params.bias;
        let b_plus = self.params.supply_voltage;

        for i in 0..input.len().min(output.len()) {
            self.input_gain.step();
            self.output_gain.step();

            let vg = bias + input[i] * self.input_gain.value;
            let vp = self.solve_plate_voltage(vg);

            let vp_quiescent = b_plus * 0.65;
            let out = (vp - vp_quiescent) / (b_plus * 0.35) * self.output_gain.value;
            output[i] = out.clamp(-1.0, 1.0);
        }
    }

    fn update_parameters(&mut self) {}
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]
    use super::*;

    #[test]
    fn test_triode_transfer_curve() {
        let params = PARAMS_12AX7;
        let mut stage = TriodeStage::new(params.clone());
        stage.prepare(44100.0);

        // Test DC transfer curve: grid voltages from -5V to 0V
        for i in 0..50 {
            let vg = -5.0 + 5.0 * (i as f32) / 50.0;
            let ip = stage.plate_current(vg, 150.0);

            // Plate current should be non-negative
            assert!(ip >= 0.0, "Ip({vg}) = {ip} < 0");
            // Plate current should be bounded
            assert!(ip < 0.01, "Ip({vg}) = {ip} too large");
        }

        // At strong negative bias, current should be near zero
        let ip_cutoff = stage.plate_current(-5.0, 150.0);
        assert!(ip_cutoff < 1e-6, "Ip at cutoff = {ip_cutoff}");

        // At zero bias, should have some current
        let ip_zero = stage.plate_current(0.0, 150.0);
        assert!(ip_zero > 1e-7, "Ip at zero bias = {ip_zero}");
    }

    #[test]
    fn test_triode_newton_convergence() {
        let params = PARAMS_12AX7;
        let mut stage = TriodeStage::new(params);
        stage.prepare(44100.0);

        // Test that Newton converges in <= 4 iterations for normal operating points
        for i in 0..20 {
            let vg = -3.0 + 3.0 * (i as f32) / 20.0;
            let b_plus = stage.params.supply_voltage;
            let rp = stage.params.plate_resistance;

            let (vp, iters) = stage.solver.solve_counted(b_plus * 0.6, |vp| {
                let ip = stage.plate_current(vg, vp);
                let dip = stage.plate_current_derivative(vg, vp);
                (b_plus - rp * ip - vp, -rp * dip - 1.0)
            });

            assert!(iters <= 4, "Newton took {iters} iterations for Vg={vg}");
            assert!(vp.is_finite(), "Vp is not finite for Vg={vg}");
        }
    }

    #[test]
    fn test_triode_audio_sine() {
        let params = PARAMS_12AX7;
        let mut stage = TriodeStage::new(params);
        stage.prepare(44100.0);

        let block_size = 128;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        // Generate sine wave input (moderate level)
        for i in 0..block_size {
            input[i] = 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        stage.process_block(&input, &mut output);

        // Verify: non-zero output, no NaN/Inf, bounded
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

        // Verify asymmetric clipping: positive and negative peaks differ
        let max_pos = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_neg = output.iter().cloned().fold(f32::INFINITY, f32::min);
        // Triode should clip asymmetrically
        assert!(
            (max_pos.abs() - max_neg.abs()).abs() > 0.001,
            "clipping is too symmetric: pos={max_pos}, neg={max_neg}"
        );
    }

    #[test]
    fn test_pentode_basic() {
        let params = PARAMS_EL34;
        let mut stage = PentodeStage::new(params);
        stage.prepare(44100.0);

        let block_size = 64;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        for i in 0..block_size {
            input[i] = 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
        }

        stage.process_block(&input, &mut output);

        for &s in &output {
            assert!(!s.is_nan(), "pentode output NaN");
            assert!(!s.is_infinite(), "pentode output Inf");
        }
    }
}
