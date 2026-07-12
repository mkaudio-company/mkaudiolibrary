//! Potentiometer voltage divider with linear/log/reverse-log taper curves.

use crate::dsp::parameter::Parameter;
use crate::sim::components::CircuitComponent;

/// Taper curves for potentiometers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Taper {
    /// Linear taper (B-type): position maps directly to resistance ratio.
    Linear,
    /// Logarithmic taper (A-type): slow start, fast finish — matches perceived loudness.
    Logarithmic,
    /// Reverse logarithmic (C-type): fast start, slow finish.
    ReverseLogarithmic,
}

/// Minimum resistance clamp (ohms) to avoid division by zero.
const MIN_R: f32 = 1.0;

/// Denominator constant for the logarithmic taper: `exp(3.0) - 1.0`.
const LOG_DENOM: f32 = 19.085_537; // precomputed: e^3 - 1

/// Potentiometer modeled as a voltage divider with smoothed position.
///
/// The wiper divides the total resistance into a top leg (wiper to top terminal)
/// and a bottom leg (wiper to bottom terminal).  A taper curve reshapes the raw
/// position into an effective electrical position before computing leg resistances.
pub struct Potentiometer {
    /// Total end-to-end resistance (ohms).
    total_resistance: f32,
    /// Knob position 0.0..1.0, smoothed via one-pole filter for click-free operation.
    position: Parameter,
    /// Taper curve applied to the raw position.
    taper: Taper,
}

impl Potentiometer {
    /// Create a new potentiometer with the given total resistance and taper.
    ///
    /// The initial position is set to 0.5 (center).
    pub fn new(total_resistance: f32, taper: Taper) -> Self {
        Self {
            total_resistance,
            position: Parameter::with_smoothing(0.5, 0.005),
            taper,
        }
    }

    /// Set the target wiper position, clamped to 0.0..=1.0.
    pub fn set_position(&mut self, pos: f32) {
        self.position.set(pos.clamp(0.0, 1.0));
    }

    /// Apply the taper curve to the current (smoothed) raw position.
    ///
    /// - Linear: identity.
    /// - Logarithmic: `(exp(pos * 3) - 1) / (exp(3) - 1)` — slow start, fast finish.
    /// - ReverseLogarithmic: mirror of logarithmic around 0.5.
    fn effective_position(&self) -> f32 {
        let pos = self.position.value;
        match self.taper {
            Taper::Linear => pos,
            Taper::Logarithmic => ((pos * 3.0).exp() - 1.0) / LOG_DENOM,
            Taper::ReverseLogarithmic => 1.0 - (((1.0 - pos) * 3.0).exp() - 1.0) / LOG_DENOM,
        }
    }

    /// Resistance from the wiper to the top terminal (ohms).
    ///
    /// Clamped to `MIN_R` (1.0 ohm) to prevent singularities.
    pub fn resistance_top(&self) -> f32 {
        let eff = self.effective_position();
        (self.total_resistance * (1.0 - eff)).max(MIN_R)
    }

    /// Resistance from the wiper to the bottom terminal (ohms).
    ///
    /// Clamped to `MIN_R` (1.0 ohm) to prevent singularities.
    pub fn resistance_bottom(&self) -> f32 {
        let eff = self.effective_position();
        (self.total_resistance * eff).max(MIN_R)
    }
}

impl CircuitComponent for Potentiometer {
    fn prepare(&mut self, _sample_rate: f32) {
        self.position.reset(self.position.target);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            self.position.step();
            let eff = self.effective_position();
            let r_bottom = (self.total_resistance * eff).max(MIN_R);
            output[i] = input[i] * r_bottom / self.total_resistance;
        }
    }

    fn update_parameters(&mut self) {
        // Parameters are smoothed per-sample in process_block.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resistance_sum_linear() {
        let mut pot = Potentiometer::new(10_000.0, Taper::Linear);
        pot.position.reset(0.5);
        let sum = pot.resistance_top() + pot.resistance_bottom();
        assert!(
            (sum - 10_000.0).abs() < 2.0 * MIN_R,
            "resistance sum {sum} != total_resistance (tolerance for MIN_R clamp)"
        );
    }

    #[test]
    fn test_resistance_sum_extremes() {
        for &taper in &[Taper::Linear, Taper::Logarithmic, Taper::ReverseLogarithmic] {
            for &pos in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                let mut pot = Potentiometer::new(100_000.0, taper);
                pot.position.reset(pos);
                let r_top = pot.resistance_top();
                let r_bot = pot.resistance_bottom();
                // With MIN_R clamp the sum may slightly exceed total_resistance at extremes.
                assert!(
                    r_top + r_bot >= 100_000.0 - 1.0,
                    "resistance sum too low for taper={taper:?}, pos={pos}: top={r_top}, bot={r_bot}"
                );
                assert!(
                    r_top + r_bot <= 100_000.0 + 2.0 * MIN_R,
                    "resistance sum too high for taper={taper:?}, pos={pos}: top={r_top}, bot={r_bot}"
                );
            }
        }
    }

    #[test]
    fn test_linear_taper_center_split() {
        let mut pot = Potentiometer::new(10_000.0, Taper::Linear);
        pot.position.reset(0.5);

        let r_top = pot.resistance_top();
        let r_bot = pot.resistance_bottom();
        assert!(
            (r_top - r_bot).abs() < 1.0,
            "linear taper at 0.5 should give equal split: top={r_top}, bot={r_bot}"
        );
    }

    #[test]
    fn test_log_taper_unequal_at_center() {
        let mut pot = Potentiometer::new(10_000.0, Taper::Logarithmic);
        pot.position.reset(0.5);

        let r_top = pot.resistance_top();
        let r_bot = pot.resistance_bottom();
        // Log taper at 0.5 should yield more resistance at top (effective pos < 0.5).
        assert!(
            r_top > r_bot,
            "log taper at 0.5 should have r_top > r_bot: top={r_top}, bot={r_bot}"
        );
    }

    #[test]
    fn test_reverse_log_taper_unequal_at_center() {
        let mut pot = Potentiometer::new(10_000.0, Taper::ReverseLogarithmic);
        pot.position.reset(0.5);

        let r_top = pot.resistance_top();
        let r_bot = pot.resistance_bottom();
        // Reverse-log taper at 0.5: effective pos > 0.5, so r_bot > r_top.
        assert!(
            r_bot > r_top,
            "reverse-log taper at 0.5 should have r_bot > r_top: top={r_top}, bot={r_bot}"
        );
    }

    #[test]
    fn test_smooth_parameter_transition() {
        let mut pot = Potentiometer::new(10_000.0, Taper::Linear);
        pot.prepare(44100.0);
        pot.set_position(0.0);
        pot.position.reset(0.0); // start at 0
        pot.set_position(1.0); // target 1

        let block_size = 256;
        let input = vec![1.0f32; block_size];
        let mut output = vec![0.0f32; block_size];

        pot.process_block(&input, &mut output);

        // Output should increase monotonically (position moving from 0 toward 1).
        for i in 1..block_size {
            assert!(
                output[i] >= output[i - 1] - 1e-7,
                "non-monotonic at sample {i}: {} -> {}",
                output[i - 1],
                output[i]
            );
        }
    }

    #[test]
    fn test_voltage_divider_output() {
        let mut pot = Potentiometer::new(10_000.0, Taper::Linear);
        pot.position.reset(1.0); // full CW — all resistance at bottom

        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        pot.process_block(&input, &mut output);

        // At position 1.0, effective_pos ≈ 1.0, so output ≈ input.
        for &s in &output {
            assert!(
                (s - 1.0).abs() < 0.01,
                "full CW output should be ~1.0, got {s}"
            );
        }
    }

    #[test]
    fn test_voltage_divider_ccw() {
        let mut pot = Potentiometer::new(10_000.0, Taper::Linear);
        pot.position.reset(0.0); // full CCW — minimal resistance at bottom

        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        pot.process_block(&input, &mut output);

        // At position 0.0, effective_pos = 0.0, r_bottom = MIN_R, output ≈ 0.
        for &s in &output {
            assert!(s.abs() < 0.01, "full CCW output should be ~0.0, got {s}");
        }
    }
}
