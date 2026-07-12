//! BJT and MOSFET gain stages via the Ebers-Moll and square-law equations
//! solved with Newton-Raphson.

use crate::dsp::parameter::Parameter;
use crate::sim::components::CircuitComponent;
use crate::sim::core::solver::NewtonSolver;
use crate::simd::generic::{ScalarFloat, simd_exp};

// ===========================================================================
// BJT
// ===========================================================================

/// BJT (bipolar junction transistor) parameters for a common-emitter stage.
#[derive(Clone, Debug)]
pub struct BjtParams {
    /// Current gain (beta / hFE).
    pub beta: f32,
    /// Saturation current (amps).
    pub is: f32,
    /// Thermal voltage (~0.0258V at room temperature).
    pub vt: f32,
    /// Collector resistance (ohms).
    pub rc: f32,
    /// Supply voltage (volts).
    pub vcc: f32,
    /// Base bias voltage (volts).
    pub vbias: f32,
}

/// 2N3904 NPN small-signal transistor.
pub const BJT_2N3904: BjtParams = BjtParams {
    beta: 200.0,
    is: 1e-14,
    vt: 0.0258,
    rc: 10_000.0,
    vcc: 9.0,
    vbias: 0.7,
};

/// Common-emitter BJT amplifier stage.
///
/// The collector current follows the Ebers-Moll model (forward active):
/// `Ic = beta * Is * (exp(Vbe/Vt) - 1)`
///
/// Circuit: Vcc → Rc → Collector → BJT → Emitter → Ground.
///
/// Solves `F(Vc) = Vcc - Rc*Ic(Vbe) - Vc = 0` using Newton-Raphson.
pub struct BjtStage {
    params: BjtParams,
    solver: NewtonSolver,
    /// Previous collector voltage — Newton initial guess.
    prev_vc: f32,
    /// Input gain (maps audio [-1,1] to Vbe variation around bias).
    input_gain: Parameter,
    /// Output gain scaling.
    output_gain: Parameter,
    sample_rate: f32,
}

impl BjtStage {
    /// Create a BJT gain stage from the given Ebers-Moll model parameters.
    pub fn new(params: BjtParams) -> Self {
        let initial_vc = params.vcc * 0.5;
        Self {
            params,
            solver: NewtonSolver {
                max_iterations: 4,
                tolerance: 1e-4,
                voltage_clamp: 500.0,
            },
            prev_vc: initial_vc,
            input_gain: Parameter::new(0.05), // small signal around bias
            output_gain: Parameter::new(1.0),
            sample_rate: 44100.0,
        }
    }

    /// Collector current: `Ic = beta * Is * (exp(Vbe/Vt) - 1)`.
    #[inline]
    pub fn collector_current(&self, vbe: f32) -> f32 {
        let exponent = vbe / self.params.vt;
        let exp_val = simd_exp(ScalarFloat(exponent)).0;
        self.params.beta * self.params.is * (exp_val - 1.0)
    }

    /// Derivative of Ic with respect to Vbe.
    /// `dIc/dVbe = beta * Is / Vt * exp(Vbe/Vt)`
    #[inline]
    #[allow(dead_code)]
    fn collector_current_derivative_vbe(&self, vbe: f32) -> f32 {
        let exponent = vbe / self.params.vt;
        let exp_val = simd_exp(ScalarFloat(exponent)).0;
        self.params.beta * self.params.is / self.params.vt * exp_val
    }

    /// Solve for collector voltage given Vbe.
    ///
    /// F(Vc) = Vcc - Rc*Ic(Vbe) - Vc = 0
    ///
    /// Since Ic depends on Vbe (not Vc directly in forward-active),
    /// the Newton equation simplifies. However we keep the solver for
    /// consistency and to handle saturation-region clamping gracefully.
    ///
    /// F'(Vc) = -1  (Ic is independent of Vc in forward-active)
    #[inline]
    fn solve_collector_voltage(&mut self, vbe: f32) -> f32 {
        let vcc = self.params.vcc;
        let rc = self.params.rc;
        let initial_guess = self.prev_vc;

        let vc = self.solver.solve(initial_guess, |vc| {
            let ic = self.collector_current(vbe);
            // Clamp Ic to prevent negative Vc (saturation)
            let ic = ic.max(0.0).min(vcc / rc);

            // F(Vc) = Vcc - Rc*Ic - Vc
            let f = vcc - rc * ic - vc;
            // F'(Vc) = -1 (Ic doesn't depend on Vc in forward-active)
            let f_prime = -1.0;

            (f, f_prime)
        });

        // Clamp to valid range [0, Vcc]
        let vc = vc.clamp(0.0, vcc);
        self.prev_vc = vc;
        vc
    }
}

impl CircuitComponent for BjtStage {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.prev_vc = self.params.vcc * 0.5;
        self.input_gain.reset(0.05);
        self.output_gain.reset(1.0);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let vcc = self.params.vcc;
        let vbias = self.params.vbias;
        let len = input.len().min(output.len());

        for i in 0..len {
            self.input_gain.step();
            self.output_gain.step();

            // Map audio input to Vbe variation around the bias point
            let vbe = vbias + input[i] * self.input_gain.value;

            // Solve for collector voltage
            let vc = self.solve_collector_voltage(vbe);

            // Normalize: center around quiescent collector voltage, scale to [-1,1]
            let vc_quiescent = vcc * 0.5;
            let out = (vc - vc_quiescent) / (vcc * 0.5) * self.output_gain.value;
            output[i] = out.clamp(-1.0, 1.0);
        }
    }

    fn update_parameters(&mut self) {
        // Parameters are smoothed per-sample in process_block
    }
}

// ===========================================================================
// MOSFET
// ===========================================================================

/// MOSFET parameters for a common-source amplifier stage.
#[derive(Clone, Debug)]
pub struct MosfetParams {
    /// Transconductance constant k (A/V^2).
    pub k: f32,
    /// Threshold voltage Vth (volts).
    pub vth: f32,
    /// Drain resistance (ohms).
    pub rd: f32,
    /// Supply voltage (volts).
    pub vdd: f32,
    /// Gate bias voltage (volts).
    pub vbias: f32,
}

/// 2N7000 N-channel enhancement MOSFET.
pub const MOSFET_2N7000: MosfetParams = MosfetParams {
    k: 0.1,
    vth: 2.0,
    rd: 1_000.0,
    vdd: 12.0,
    vbias: 3.0,
};

/// Common-source MOSFET amplifier stage.
///
/// The drain current in saturation follows the square-law model:
/// `Id = (k/2) * max(0, Vgs - Vth)^2`
///
/// Circuit: Vdd → Rd → Drain → MOSFET → Source → Ground.
///
/// In the triode region, `Id` also depends on Vds, so Newton-Raphson is used
/// for generality. In saturation the solution is direct, but the solver
/// handles the transition between regions smoothly.
pub struct MosfetStage {
    params: MosfetParams,
    solver: NewtonSolver,
    /// Previous drain voltage — Newton initial guess.
    prev_vd: f32,
    /// Input gain (maps audio [-1,1] to Vgs variation around bias).
    input_gain: Parameter,
    /// Output gain scaling.
    output_gain: Parameter,
    sample_rate: f32,
}

impl MosfetStage {
    /// Create a MOSFET gain stage from the given square-law model parameters.
    pub fn new(params: MosfetParams) -> Self {
        let initial_vd = params.vdd * 0.5;
        Self {
            params,
            solver: NewtonSolver {
                max_iterations: 4,
                tolerance: 1e-4,
                voltage_clamp: 500.0,
            },
            prev_vd: initial_vd,
            input_gain: Parameter::new(0.5),
            output_gain: Parameter::new(1.0),
            sample_rate: 44100.0,
        }
    }

    /// Drain current in saturation: `Id = (k/2) * max(0, Vgs - Vth)^2`.
    #[inline]
    pub fn drain_current_saturation(&self, vgs: f32) -> f32 {
        let overdrive = (vgs - self.params.vth).max(0.0);
        (self.params.k / 2.0) * overdrive * overdrive
    }

    /// Drain current accounting for triode/saturation regions.
    ///
    /// - Cutoff: Vgs < Vth → Id = 0
    /// - Triode: Vds < Vgs - Vth → Id = k * ((Vgs - Vth)*Vds - Vds^2/2)
    /// - Saturation: Vds >= Vgs - Vth → Id = (k/2) * (Vgs - Vth)^2
    #[inline]
    pub fn drain_current(&self, vgs: f32, vds: f32) -> f32 {
        let overdrive = vgs - self.params.vth;
        if overdrive <= 0.0 {
            // Cutoff
            0.0
        } else if vds < overdrive {
            // Triode region
            self.params.k * (overdrive * vds - vds * vds / 2.0)
        } else {
            // Saturation region
            (self.params.k / 2.0) * overdrive * overdrive
        }
    }

    /// Derivative of drain current with respect to Vd (= -dId/dVds since Vds = Vd for grounded source).
    ///
    /// In saturation, dId/dVds = 0 → dId/dVd = 0.
    /// In triode, dId/dVds = k * (Vgs - Vth - Vds) → dId/dVd = k * (Vgs - Vth - Vd).
    #[inline]
    fn drain_current_derivative_vd(&self, vgs: f32, vd: f32) -> f32 {
        let overdrive = vgs - self.params.vth;
        if overdrive <= 0.0 {
            0.0
        } else if vd < overdrive {
            // Triode: dId/dVd = k * (overdrive - Vd)
            self.params.k * (overdrive - vd)
        } else {
            // Saturation: Id independent of Vd
            0.0
        }
    }

    /// Solve for drain voltage given Vgs.
    ///
    /// F(Vd) = Vdd - Rd*Id(Vgs, Vd) - Vd = 0
    /// F'(Vd) = -Rd * dId/dVd - 1
    #[inline]
    fn solve_drain_voltage(&mut self, vgs: f32) -> f32 {
        let vdd = self.params.vdd;
        let rd = self.params.rd;
        let initial_guess = self.prev_vd;

        let vd = self.solver.solve(initial_guess, |vd| {
            let vd_clamped = vd.max(0.0);
            let id = self.drain_current(vgs, vd_clamped);
            let did_dvd = self.drain_current_derivative_vd(vgs, vd_clamped);

            // F(Vd) = Vdd - Rd*Id - Vd
            let f = vdd - rd * id - vd;
            // F'(Vd) = -Rd * dId/dVd - 1
            let f_prime = -rd * did_dvd - 1.0;

            (f, f_prime)
        });

        let vd = vd.clamp(0.0, vdd);
        self.prev_vd = vd;
        vd
    }
}

impl CircuitComponent for MosfetStage {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.prev_vd = self.params.vdd * 0.5;
        self.input_gain.reset(0.5);
        self.output_gain.reset(1.0);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let vdd = self.params.vdd;
        let vbias = self.params.vbias;
        let len = input.len().min(output.len());

        for i in 0..len {
            self.input_gain.step();
            self.output_gain.step();

            // Map audio input to Vgs variation around bias
            let vgs = vbias + input[i] * self.input_gain.value;

            // Solve for drain voltage
            let vd = self.solve_drain_voltage(vgs);

            // Normalize output
            let vd_quiescent = vdd * 0.5;
            let out = (vd - vd_quiescent) / (vdd * 0.5) * self.output_gain.value;
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

    // -----------------------------------------------------------------------
    // BJT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bjt_ic_vs_vbe() {
        // Verify Ic follows exponential relationship with Vbe.
        let stage = BjtStage::new(BJT_2N3904.clone());

        let mut prev_ic = 0.0f32;
        for i in 0..50 {
            let vbe = 0.3 + 0.4 * (i as f32) / 50.0; // 0.3V to 0.7V
            let ic = stage.collector_current(vbe);

            assert!(!ic.is_nan(), "Ic(Vbe={vbe}) is NaN");
            assert!(!ic.is_infinite(), "Ic(Vbe={vbe}) is Inf");
            assert!(ic >= 0.0, "Ic(Vbe={vbe}) = {ic} < 0");

            // Current should be monotonically increasing with Vbe
            assert!(
                ic >= prev_ic,
                "Ic not monotonic: Ic({vbe})={ic} < prev={prev_ic}"
            );
            prev_ic = ic;
        }
    }

    #[test]
    fn test_bjt_exponential_transfer() {
        // Verify that Ic matches the expected exponential formula.
        let params = BJT_2N3904.clone();
        let stage = BjtStage::new(params.clone());

        let vbe = 0.6;
        let expected = params.beta * params.is * ((vbe / params.vt).exp() - 1.0);
        let actual = stage.collector_current(vbe);

        let rel_err = if expected.abs() > 1e-20 {
            (actual - expected).abs() / expected.abs()
        } else {
            (actual - expected).abs()
        };
        assert!(
            rel_err < 1e-3,
            "Ic mismatch at Vbe={vbe}: expected={expected}, actual={actual}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_bjt_audio_sine() {
        let mut stage = BjtStage::new(BJT_2N3904.clone());
        stage.prepare(44100.0);

        let block_size = 256;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        for i in 0..block_size {
            input[i] = 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        stage.process_block(&input, &mut output);

        let mut has_nonzero = false;
        for &s in &output {
            assert!(!s.is_nan(), "BJT output contains NaN");
            assert!(!s.is_infinite(), "BJT output contains Inf");
            assert!(s.abs() <= 1.0, "BJT output {s} exceeds [-1, 1]");
            if s.abs() > 0.001 {
                has_nonzero = true;
            }
        }
        assert!(has_nonzero, "BJT output is all zeros");
    }

    #[test]
    fn test_bjt_gain_and_clipping() {
        // With a sine input, the BJT stage should amplify and eventually clip.
        let mut stage = BjtStage::new(BJT_2N3904.clone());
        stage.prepare(44100.0);

        // Larger input to exercise clipping
        let block_size = 256;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        for i in 0..block_size {
            input[i] = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        stage.process_block(&input, &mut output);

        // Output should be bounded
        for &s in &output {
            assert!(s.abs() <= 1.0, "BJT output {s} exceeds [-1, 1]");
            assert!(!s.is_nan(), "BJT output contains NaN");
        }
    }

    // -----------------------------------------------------------------------
    // MOSFET tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mosfet_id_vs_vgs_square_law() {
        // Verify Id follows square-law in saturation.
        let stage = MosfetStage::new(MOSFET_2N7000.clone());

        // Below threshold: Id = 0
        let id_below = stage.drain_current_saturation(1.5);
        assert!(
            id_below.abs() < 1e-10,
            "Id below threshold should be ~0, got {id_below}"
        );

        // At threshold: Id = 0
        let id_at = stage.drain_current_saturation(2.0);
        assert!(
            id_at.abs() < 1e-10,
            "Id at threshold should be ~0, got {id_at}"
        );

        // Above threshold: Id = (k/2) * (Vgs - Vth)^2
        let vgs = 4.0;
        let expected = (MOSFET_2N7000.k / 2.0) * (vgs - MOSFET_2N7000.vth).powi(2);
        let actual = stage.drain_current_saturation(vgs);
        assert!(
            (actual - expected).abs() < 1e-6,
            "Square-law mismatch: expected={expected}, actual={actual}"
        );
    }

    #[test]
    fn test_mosfet_id_monotonic() {
        let stage = MosfetStage::new(MOSFET_2N7000.clone());

        let mut prev_id = 0.0f32;
        for i in 0..100 {
            let vgs = 1.0 + 4.0 * (i as f32) / 100.0;
            let id = stage.drain_current_saturation(vgs);

            assert!(!id.is_nan(), "Id(Vgs={vgs}) is NaN");
            assert!(id >= 0.0, "Id(Vgs={vgs}) = {id} < 0");
            assert!(
                id >= prev_id,
                "Id not monotonic: Id({vgs})={id} < prev={prev_id}"
            );
            prev_id = id;
        }
    }

    #[test]
    fn test_mosfet_triode_region() {
        let stage = MosfetStage::new(MOSFET_2N7000.clone());

        let vgs = 4.0; // overdrive = 2.0V
        let vds_triode = 1.0; // less than overdrive → triode region
        let id_triode = stage.drain_current(vgs, vds_triode);

        let expected_triode = MOSFET_2N7000.k
            * ((vgs - MOSFET_2N7000.vth) * vds_triode - vds_triode * vds_triode / 2.0);
        assert!(
            (id_triode - expected_triode).abs() < 1e-6,
            "Triode region mismatch: expected={expected_triode}, actual={id_triode}"
        );

        let vds_sat = 3.0; // greater than overdrive → saturation
        let id_sat = stage.drain_current(vgs, vds_sat);
        let expected_sat = (MOSFET_2N7000.k / 2.0) * (vgs - MOSFET_2N7000.vth).powi(2);
        assert!(
            (id_sat - expected_sat).abs() < 1e-6,
            "Saturation region mismatch: expected={expected_sat}, actual={id_sat}"
        );
    }

    #[test]
    fn test_mosfet_audio_sine() {
        let mut stage = MosfetStage::new(MOSFET_2N7000.clone());
        stage.prepare(44100.0);

        let block_size = 256;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        for i in 0..block_size {
            input[i] = 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        stage.process_block(&input, &mut output);

        let mut has_nonzero = false;
        for &s in &output {
            assert!(!s.is_nan(), "MOSFET output contains NaN");
            assert!(!s.is_infinite(), "MOSFET output contains Inf");
            assert!(s.abs() <= 1.0, "MOSFET output {s} exceeds [-1, 1]");
            if s.abs() > 0.001 {
                has_nonzero = true;
            }
        }
        assert!(has_nonzero, "MOSFET output is all zeros");
    }

    #[test]
    fn test_no_nan_inf_sweep() {
        // Sweep a wide range of inputs through both BJT and MOSFET stages.
        let mut bjt = BjtStage::new(BJT_2N3904.clone());
        bjt.prepare(44100.0);

        let mut mosfet = MosfetStage::new(MOSFET_2N7000.clone());
        mosfet.prepare(44100.0);

        let block_size = 64;
        let mut output = vec![0.0f32; block_size];

        for level in &[0.01f32, 0.1, 0.5, 1.0] {
            let input: Vec<f32> = (0..block_size)
                .map(|i| level * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
                .collect();

            bjt.process_block(&input, &mut output);
            for &s in &output {
                assert!(!s.is_nan(), "BJT NaN at level={level}");
                assert!(!s.is_infinite(), "BJT Inf at level={level}");
            }

            mosfet.process_block(&input, &mut output);
            for &s in &output {
                assert!(!s.is_nan(), "MOSFET NaN at level={level}");
                assert!(!s.is_infinite(), "MOSFET Inf at level={level}");
            }
        }
    }
}
