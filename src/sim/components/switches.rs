//! On/off switch with smoothed conductance to avoid audible zipper clicks.

use crate::dsp::parameter::Parameter;
use crate::sim::components::CircuitComponent;

/// Resistance when the switch is closed (ohms).
const R_ON: f32 = 0.1;

/// Resistance when the switch is open (ohms) — effectively an open circuit.
const R_OFF: f32 = 50_000_000.0;

/// An ideal switch modeled through smoothed conductance for click-free transitions.
///
/// When ON the switch presents a very low resistance (R_ON = 0.1 ohm) and passes
/// signal through virtually unattenuated.  When OFF the switch presents a very high
/// resistance (R_OFF = 50 M-ohm) and attenuates the signal to near zero.
///
/// The transition between states is smoothed by ramping conductance through a
/// one-pole filter so that no discontinuities (clicks) appear in the audio path.
pub struct Switch {
    /// Current logical switch state.
    state: bool,
    /// Conductance parameter (1/R), smoothed for click-free transitions.
    conductance: Parameter,
    /// Resistance when ON (ohms).
    r_on: f32,
    /// Resistance when OFF (ohms).
    r_off: f32,
}

impl Switch {
    /// Create a new switch in the OFF state.
    pub fn new() -> Self {
        Self {
            state: false,
            conductance: Parameter::with_smoothing(1.0 / R_OFF, 0.005),
            r_on: R_ON,
            r_off: R_OFF,
        }
    }

    /// Set the switch state.
    ///
    /// The conductance smoothly transitions to the target value.
    pub fn set_state(&mut self, on: bool) {
        self.state = on;
        if on {
            self.conductance.set(1.0 / self.r_on);
        } else {
            self.conductance.set(1.0 / self.r_off);
        }
    }

    /// Toggle the switch state.
    pub fn toggle(&mut self) {
        self.set_state(!self.state);
    }

    /// Returns `true` if the switch is logically ON.
    pub fn is_on(&self) -> bool {
        self.state
    }

    /// Current effective resistance (ohms), derived from smoothed conductance.
    pub fn current_resistance(&self) -> f32 {
        1.0 / self.conductance.value
    }

    /// Current smoothed conductance (siemens).
    pub fn current_conductance(&self) -> f32 {
        self.conductance.value
    }
}

impl Default for Switch {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitComponent for Switch {
    fn prepare(&mut self, _sample_rate: f32) {
        if self.state {
            self.conductance.reset(1.0 / self.r_on);
        } else {
            self.conductance.reset(1.0 / self.r_off);
        }
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            self.conductance.step();
            // Attenuation factor: R_ON * conductance.
            // When ON:  R_ON * (1/R_ON)  = 1.0  → full pass-through.
            // When OFF: R_ON * (1/R_OFF) ≈ 0.0  → full attenuation.
            let gain = self.r_on * self.conductance.value;
            output[i] = input[i] * gain;
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
    fn test_switch_on_passes_signal() {
        let mut sw = Switch::new();
        sw.set_state(true);
        sw.prepare(44100.0); // reset conductance to ON immediately

        let input = [1.0f32; 64];
        let mut output = [0.0f32; 64];
        sw.process_block(&input, &mut output);

        // After prepare the conductance is already at the ON target, so
        // output should be very close to input.
        for &s in &output {
            assert!(
                (s - 1.0).abs() < 0.01,
                "ON switch should pass signal, got {s}"
            );
        }
    }

    #[test]
    fn test_switch_off_blocks_signal() {
        let mut sw = Switch::new();
        sw.prepare(44100.0); // default is OFF

        let input = [1.0f32; 64];
        let mut output = [0.0f32; 64];
        sw.process_block(&input, &mut output);

        for &s in &output {
            assert!(s.abs() < 1e-4, "OFF switch should block signal, got {s}");
        }
    }

    #[test]
    fn test_toggle() {
        let mut sw = Switch::new();
        assert!(!sw.is_on());
        sw.toggle();
        assert!(sw.is_on());
        sw.toggle();
        assert!(!sw.is_on());
    }

    #[test]
    fn test_smooth_transition_on_to_off() {
        let mut sw = Switch::new();
        sw.set_state(true);
        sw.prepare(44100.0); // start in ON state

        // Now switch off — conductance should ramp down smoothly.
        sw.set_state(false);

        let block_size = 512;
        let input = vec![1.0f32; block_size];
        let mut output = vec![0.0f32; block_size];
        sw.process_block(&input, &mut output);

        // Output should decrease monotonically.
        for i in 1..block_size {
            assert!(
                output[i] <= output[i - 1] + 1e-7,
                "non-monotonic decrease at sample {i}: {} -> {}",
                output[i - 1],
                output[i]
            );
        }

        // First sample should still be close to 1.0 (just started ramping).
        assert!(
            output[0] > 0.9,
            "transition should start near 1.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_smooth_transition_off_to_on() {
        let mut sw = Switch::new();
        sw.prepare(44100.0); // start in OFF state

        // Now switch on — conductance should ramp up smoothly.
        sw.set_state(true);

        let block_size = 512;
        let input = vec![1.0f32; block_size];
        let mut output = vec![0.0f32; block_size];
        sw.process_block(&input, &mut output);

        // Output should increase monotonically.
        for i in 1..block_size {
            assert!(
                output[i] >= output[i - 1] - 1e-7,
                "non-monotonic increase at sample {i}: {} -> {}",
                output[i - 1],
                output[i]
            );
        }
    }

    #[test]
    fn test_current_resistance_on() {
        let mut sw = Switch::new();
        sw.set_state(true);
        sw.prepare(44100.0);

        let r = sw.current_resistance();
        assert!(
            (r - R_ON).abs() < 0.01,
            "ON resistance should be ~{R_ON}, got {r}"
        );
    }

    #[test]
    fn test_current_resistance_off() {
        let mut sw = Switch::new();
        sw.prepare(44100.0);

        let r = sw.current_resistance();
        assert!(
            (r - R_OFF).abs() / R_OFF < 0.01,
            "OFF resistance should be ~{R_OFF}, got {r}"
        );
    }

    #[test]
    fn test_no_nan_or_inf() {
        let mut sw = Switch::new();
        sw.prepare(44100.0);

        // Rapidly toggle and process to stress-test for NaN/Inf.
        let input = vec![0.7f32; 64];
        let mut output = vec![0.0f32; 64];

        for _ in 0..10 {
            sw.toggle();
            sw.process_block(&input, &mut output);
            for &s in &output {
                assert!(!s.is_nan(), "output contains NaN");
                assert!(!s.is_infinite(), "output contains Inf");
            }
        }
    }
}
