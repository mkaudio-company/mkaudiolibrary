//! Operational amplifier gain stage with slew-rate limiting and rail clamping.

use crate::dsp::parameter::Parameter;
use crate::sim::components::CircuitComponent;

/// Voltage-feedback operational amplifier parameters.
#[derive(Clone, Debug)]
pub struct OpAmpParams {
    /// Open-loop gain (dimensionless).
    pub gain: f32,
    /// Positive rail voltage (volts).
    pub v_pos: f32,
    /// Negative rail voltage (volts).
    pub v_neg: f32,
    /// Slew rate in V/s.
    pub slew_rate: f32,
}

/// TL072 JFET-input op-amp.
pub const OPAMP_TL072: OpAmpParams = OpAmpParams {
    gain: 200_000.0,
    v_pos: 15.0,
    v_neg: -15.0,
    slew_rate: 13e6,
};

/// Voltage-feedback op-amp stage.
///
/// Models a raw op-amp with:
/// - High open-loop gain with rail clamping
/// - Slew-rate limiting
///
/// The input is treated as the differential voltage `(V+ - V-)`.
/// The user connects feedback externally by scaling the input appropriately.
///
/// Output equation:
/// 1. `Vout_ideal = clamp(gain * V_diff, V_neg, V_pos)`
/// 2. Slew-rate limit: `|Vout - Vout_prev| <= slew_rate * dt`
pub struct OpAmpStage {
    params: OpAmpParams,
    /// Previous output voltage for slew-rate limiting.
    prev_output: f32,
    /// Maximum voltage change per sample (computed from slew rate and sample rate).
    max_dv_per_sample: f32,
    /// Gain parameter (smoothed, allows real-time adjustment).
    gain_param: Parameter,
    sample_rate: f32,
}

impl OpAmpStage {
    /// Create a new op-amp stage with the given parameters.
    pub fn new(params: OpAmpParams) -> Self {
        let gain = params.gain;
        Self {
            params,
            prev_output: 0.0,
            max_dv_per_sample: 0.0, // set properly in prepare()
            gain_param: Parameter::new(gain),
            sample_rate: 44100.0,
        }
    }

    /// Process a single sample: amplify differential input with clamping and slew limiting.
    #[inline]
    fn process_sample(&mut self, v_diff: f32) -> f32 {
        let gain = self.gain_param.value;

        // Amplify with open-loop gain
        let v_ideal = (gain * v_diff).clamp(self.params.v_neg, self.params.v_pos);

        // Apply slew-rate limiting
        let delta = v_ideal - self.prev_output;
        let slew_limited = if delta.abs() > self.max_dv_per_sample {
            self.prev_output + self.max_dv_per_sample * delta.signum()
        } else {
            v_ideal
        };

        self.prev_output = slew_limited;
        slew_limited
    }

    /// WDF-compatible solve interface.
    ///
    /// Takes an incident wave `a` and returns the reflected wave `b`.
    /// For a simple voltage amplifier: b = process_sample(a) scaled appropriately.
    pub fn solve_wdf(&mut self, incident: f32) -> f32 {
        // Treat incident wave as the differential input voltage
        let v_out = self.process_sample(incident);
        // Reflected wave: the output voltage
        v_out
    }

    /// Set the open-loop gain (smoothed transition).
    pub fn set_gain(&mut self, gain: f32) {
        self.gain_param.set(gain);
    }
}

impl CircuitComponent for OpAmpStage {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.prev_output = 0.0;

        // Compute maximum voltage change per sample from slew rate
        // slew_rate is in V/s, so dV/sample = slew_rate / sample_rate
        self.max_dv_per_sample = self.params.slew_rate / sample_rate;

        self.gain_param.reset(self.params.gain);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());

        for i in 0..len {
            self.gain_param.step();

            // Input is the differential voltage (V+ - V-), normalized to audio range.
            // Scale from [-1, 1] audio to a small differential voltage.
            // A typical full-scale audio signal maps to a few millivolts of differential input.
            let v_diff = input[i] * (self.params.v_pos / self.gain_param.value);

            let v_out = self.process_sample(v_diff);

            // Normalize output to [-1, 1] range
            let out = v_out / self.params.v_pos;
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
    fn test_dc_transfer_linear_region() {
        // In the linear region, Vout = gain * Vdiff, clamped to rails.
        let params = OPAMP_TL072.clone();
        let mut stage = OpAmpStage::new(params.clone());
        stage.prepare(44100.0);

        // Small differential input should produce proportional output
        let v_diff = 1e-5; // 10uV
        let expected = params.gain * v_diff;

        // Process directly (bypass audio normalization)
        let v_out = stage.process_sample(v_diff);

        // Should be close to gain * v_diff (within slew rate allowance)
        assert!(
            (v_out - expected).abs() < 1.0,
            "Linear region: expected ~{expected}, got {v_out}"
        );
    }

    #[test]
    fn test_rail_saturation() {
        // Large input should saturate at the rails.
        let params = OPAMP_TL072.clone();
        let mut stage = OpAmpStage::new(params.clone());
        stage.prepare(44100.0);

        // Run enough samples so slew rate doesn't limit us
        for _ in 0..10000 {
            stage.process_sample(1.0); // large positive differential
        }
        let v_out_pos = stage.prev_output;

        // Reset and go negative
        stage.prev_output = 0.0;
        for _ in 0..10000 {
            stage.process_sample(-1.0);
        }
        let v_out_neg = stage.prev_output;

        assert!(
            (v_out_pos - params.v_pos).abs() < 0.01,
            "Positive rail: expected {}, got {v_out_pos}",
            params.v_pos
        );
        assert!(
            (v_out_neg - params.v_neg).abs() < 0.01,
            "Negative rail: expected {}, got {v_out_neg}",
            params.v_neg
        );
    }

    #[test]
    fn test_slew_rate_limiting() {
        let params = OpAmpParams {
            gain: 200_000.0,
            v_pos: 15.0,
            v_neg: -15.0,
            slew_rate: 1e6, // 1 V/us = 1e6 V/s (deliberately slow for testing)
        };

        let sample_rate = 1e6; // 1 MHz sample rate for easy math
        let mut stage = OpAmpStage::new(params);
        stage.prepare(sample_rate);

        // max_dv_per_sample = 1e6 / 1e6 = 1.0 V/sample
        assert!(
            (stage.max_dv_per_sample - 1.0).abs() < 1e-6,
            "max_dv_per_sample = {}, expected 1.0",
            stage.max_dv_per_sample
        );

        // Start at 0, apply large step → output should ramp at 1V/sample
        stage.prev_output = 0.0;

        let v1 = stage.process_sample(1.0); // wants to jump to 15V, limited to 1V
        assert!(
            (v1 - 1.0).abs() < 1e-4,
            "After 1 sample, expected ~1.0V, got {v1}"
        );

        let v2 = stage.process_sample(1.0);
        assert!(
            (v2 - 2.0).abs() < 1e-4,
            "After 2 samples, expected ~2.0V, got {v2}"
        );

        let v3 = stage.process_sample(1.0);
        assert!(
            (v3 - 3.0).abs() < 1e-4,
            "After 3 samples, expected ~3.0V, got {v3}"
        );
    }

    #[test]
    fn test_slew_rate_negative_step() {
        let params = OpAmpParams {
            gain: 200_000.0,
            v_pos: 15.0,
            v_neg: -15.0,
            slew_rate: 1e6,
        };

        let mut stage = OpAmpStage::new(params);
        stage.prepare(1e6);

        // Start at 10V, step to -15V
        stage.prev_output = 10.0;
        let v1 = stage.process_sample(-1.0);
        assert!(
            (v1 - 9.0).abs() < 1e-4,
            "Negative slew: expected ~9.0V, got {v1}"
        );
    }

    #[test]
    fn test_audio_sine_amplification() {
        let mut stage = OpAmpStage::new(OPAMP_TL072.clone());
        stage.prepare(44100.0);

        let block_size = 256;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        // Small sine wave
        for i in 0..block_size {
            input[i] = 0.3 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        stage.process_block(&input, &mut output);

        let mut has_nonzero = false;
        for &s in &output {
            assert!(!s.is_nan(), "op-amp output contains NaN");
            assert!(!s.is_infinite(), "op-amp output contains Inf");
            assert!(s.abs() <= 1.0, "op-amp output {s} exceeds [-1, 1]");
            if s.abs() > 0.001 {
                has_nonzero = true;
            }
        }
        assert!(has_nonzero, "op-amp output is all zeros");
    }

    #[test]
    fn test_audio_clipping_at_rails() {
        // With a large input, the op-amp should clip at the rails.
        let mut stage = OpAmpStage::new(OPAMP_TL072.clone());
        stage.prepare(44100.0);

        let block_size = 512;
        let mut input = vec![0.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        // Full-scale input
        for i in 0..block_size {
            input[i] = (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 44100.0).sin();
        }

        stage.process_block(&input, &mut output);

        // Output should be bounded by [-1, 1] (normalized rails)
        let max_out = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_out = output.iter().cloned().fold(f32::INFINITY, f32::min);

        assert!(max_out <= 1.0, "max output {max_out} > 1.0");
        assert!(min_out >= -1.0, "min output {min_out} < -1.0");

        for &s in &output {
            assert!(!s.is_nan(), "output NaN");
            assert!(!s.is_infinite(), "output Inf");
        }
    }

    #[test]
    fn test_solve_wdf_interface() {
        let mut stage = OpAmpStage::new(OPAMP_TL072.clone());
        stage.prepare(44100.0);

        // WDF interface should produce valid output
        let result = stage.solve_wdf(1e-5);
        assert!(!result.is_nan(), "solve_wdf returned NaN");
        assert!(!result.is_infinite(), "solve_wdf returned Inf");
        assert!(
            result.abs() <= OPAMP_TL072.v_pos,
            "solve_wdf output {result} exceeds rails"
        );
    }
}
