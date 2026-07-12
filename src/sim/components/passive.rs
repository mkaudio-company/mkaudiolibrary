//! Passive component wrappers: `CircuitComponent` implementations backed by WDF elements.
//!
//! These wrap the raw WDF components from [`crate::sim::wdf`] into the
//! [`CircuitComponent`] interface used by the rest of the engine, processing
//! audio in block-based fashion.

use crate::dsp::parameter::Parameter;
use crate::sim::components::CircuitComponent;
use crate::sim::wdf::adaptors::SeriesAdaptor;
use crate::sim::wdf::components::{WdfCapacitor, WdfComponent, WdfResistor};

// ---------------------------------------------------------------------------
// Resistor
// ---------------------------------------------------------------------------

/// A simple resistive attenuator as a `CircuitComponent`.
///
/// Implements a voltage divider using the WDF resistor model.
/// The output is scaled by the ratio `R_load / (R + R_load)` where
/// `R_load` is a fixed reference impedance (e.g., the next stage's input impedance).
pub struct Resistor {
    /// The WDF resistor element.
    wdf: WdfResistor,
    /// Load resistance for the voltage divider.
    r_load: f32,
    /// Attenuation factor: R_load / (R + R_load). Precomputed.
    attenuation: f32,
    /// Smoothed resistance parameter (allows real-time changes).
    resistance_param: Parameter,
}

impl Resistor {
    /// Create a new resistor component.
    ///
    /// `resistance`: the series resistance in ohms.
    /// `r_load`: the load resistance (next stage input impedance).
    pub fn new(resistance: f32, r_load: f32) -> Self {
        let atten = r_load / (resistance + r_load);
        Self {
            wdf: WdfResistor::new(resistance),
            r_load,
            attenuation: atten,
            resistance_param: Parameter::new(resistance),
        }
    }

    /// Set the resistance value (smoothed).
    pub fn set_resistance(&mut self, r: f32) {
        self.resistance_param.set(r);
    }
}

impl CircuitComponent for Resistor {
    fn prepare(&mut self, _sample_rate: f32) {
        self.resistance_param.reset(self.wdf.resistance());
        self.attenuation = self.r_load / (self.wdf.resistance() + self.r_load);
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            self.resistance_param.step();
            if !self.resistance_param.is_settled() {
                let r = self.resistance_param.value;
                self.attenuation = self.r_load / (r + self.r_load);
            }
            output[i] = input[i] * self.attenuation;
        }
    }

    fn update_parameters(&mut self) {
        // Smoothing happens per-sample in process_block.
    }
}

// ---------------------------------------------------------------------------
// Capacitor (RC high-pass / coupling cap)
// ---------------------------------------------------------------------------

/// An RC coupling capacitor as a `CircuitComponent`.
///
/// Models a series capacitor with a load resistor to ground, forming a
/// first-order high-pass filter. This is the standard coupling/blocking
/// capacitor found in audio circuits.
///
/// ```text
/// Circuit: input --[C]--+-- output
///                       |
///                    [R_load]
///                       |
///                      GND
/// ```
///
/// Implemented as a WDF series adaptor with the capacitor and a resistor.
pub struct Capacitor {
    /// The WDF series adaptor tree: series(C, R_load).
    adaptor: SeriesAdaptor<WdfCapacitor, WdfResistor>,
    /// Sample rate.
    sample_rate: f32,
}

impl Capacitor {
    /// Create a new coupling capacitor.
    ///
    /// `capacitance`: capacitance in farads (e.g., 100e-9 for 100nF).
    /// `r_load`: load resistance in ohms (e.g., 1_000_000 for 1M input impedance).
    pub fn new(capacitance: f32, r_load: f32) -> Self {
        let cap = WdfCapacitor::new(capacitance);
        let res = WdfResistor::new(r_load);
        let adaptor = SeriesAdaptor::new(cap, res);
        Self {
            adaptor,
            sample_rate: 44100.0,
        }
    }

    /// Get a reference to the internal WDF capacitor.
    pub fn wdf_capacitor(&self) -> &WdfCapacitor {
        &self.adaptor.left
    }
}

impl CircuitComponent for Capacitor {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.adaptor.left.set_sample_rate(sample_rate);
        self.adaptor.left.reset();
        self.adaptor.update_impedance();
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            // Bottom-up: get reflected wave from tree
            let b_tree = self.adaptor.reflected_cached();

            // Source: ideal voltage source with Vs = input sample.
            // a_root = Vs - b_tree
            let a_root = input[i] - b_tree;

            // Compute output voltage across the load resistor (right child)
            // before propagating incident waves.
            let diff = a_root - (self.adaptor.b_left + self.adaptor.b_right);
            let a_r = self.adaptor.b_right + (1.0 - self.adaptor.gamma) * diff;
            let v_out = a_r + self.adaptor.b_right; // V = a + b

            output[i] = v_out;

            // Top-down: propagate incident waves
            self.adaptor.incident(a_root);
        }
    }

    fn update_parameters(&mut self) {}
}

// ---------------------------------------------------------------------------
// Inductor (RL low-pass)
// ---------------------------------------------------------------------------

/// An RL low-pass filter as a `CircuitComponent`.
///
/// Models a series inductor with a load resistor, forming a first-order
/// low-pass filter.
///
/// ```text
/// Circuit: input --[L]--+-- output
///                       |
///                    [R_load]
///                       |
///                      GND
/// ```
pub struct Inductor {
    /// WDF series adaptor tree: series(L, R_load).
    adaptor: SeriesAdaptor<crate::sim::wdf::components::WdfInductor, WdfResistor>,
    /// Sample rate.
    sample_rate: f32,
}

impl Inductor {
    /// Create a new RL low-pass inductor component.
    ///
    /// `inductance`: inductance in henries.
    /// `r_load`: load resistance in ohms.
    pub fn new(inductance: f32, r_load: f32) -> Self {
        let ind = crate::sim::wdf::components::WdfInductor::new(inductance);
        let res = WdfResistor::new(r_load);
        let adaptor = SeriesAdaptor::new(ind, res);
        Self {
            adaptor,
            sample_rate: 44100.0,
        }
    }
}

impl CircuitComponent for Inductor {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.adaptor.left.set_sample_rate(sample_rate);
        self.adaptor.left.reset();
        self.adaptor.update_impedance();
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            let b_tree = self.adaptor.reflected_cached();
            let a_root = input[i] - b_tree;

            // Output voltage across the load resistor (right child).
            let diff = a_root - (self.adaptor.b_left + self.adaptor.b_right);
            let a_r = self.adaptor.b_right + (1.0 - self.adaptor.gamma) * diff;
            let v_out = a_r + self.adaptor.b_right;

            output[i] = v_out;

            self.adaptor.incident(a_root);
        }
    }

    fn update_parameters(&mut self) {}
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]
    use super::*;

    #[test]
    fn test_resistor_attenuation() {
        let mut r = Resistor::new(1000.0, 1000.0);
        r.prepare(44100.0);

        let input = [1.0f32; 64];
        let mut output = [0.0f32; 64];
        r.process_block(&input, &mut output);

        // R_load / (R + R_load) = 0.5
        for &s in &output {
            assert!((s - 0.5).abs() < 0.01, "Expected 0.5, got {s}");
        }
    }

    #[test]
    fn test_capacitor_dc_blocking() {
        let mut cap = Capacitor::new(100e-9, 10_000.0); // 100nF, 10k load → tau=1ms
        cap.prepare(44100.0);

        // Feed DC for 500ms (many time constants)
        let n = 22050;
        let input = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];
        cap.process_block(&input, &mut output);

        // After many time constants, output should approach zero (DC blocked).
        let last_avg: f32 = output[n - 100..].iter().sum::<f32>() / 100.0;
        assert!(
            last_avg.abs() < 0.5,
            "DC should be blocked: last_avg={last_avg}"
        );
    }

    #[test]
    fn test_capacitor_ac_passthrough() {
        let mut cap = Capacitor::new(1e-6, 10_000.0); // 1uF, 10k load
        cap.prepare(44100.0);

        // 1kHz sine — should pass through the coupling cap.
        let n = 4410;
        let mut input = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        for i in 0..n {
            input[i] = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        cap.process_block(&input, &mut output);

        // After settling (skip first 2000 samples), output should have
        // significant amplitude.
        let rms_out: f32 = output[2000..].iter().map(|x| x * x).sum::<f32>() / (n - 2000) as f32;
        let rms_in: f32 = input[2000..].iter().map(|x| x * x).sum::<f32>() / (n - 2000) as f32;

        let ratio = rms_out.sqrt() / rms_in.sqrt();
        assert!(
            ratio > 0.1,
            "1kHz should pass through coupling cap: ratio={ratio}"
        );
    }

    #[test]
    fn test_inductor_lowpass() {
        let mut ind = Inductor::new(0.1, 1000.0); // 100mH, 1k load
        ind.prepare(44100.0);

        // Low frequency (100 Hz) should pass, high frequency (10 kHz) should be attenuated.
        let n = 8820; // 200ms

        // Test low frequency
        let mut input_low = vec![0.0f32; n];
        let mut output_low = vec![0.0f32; n];
        for i in 0..n {
            input_low[i] = (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 44100.0).sin();
        }
        ind.process_block(&input_low, &mut output_low);

        let rms_low_in: f32 = input_low[n / 2..].iter().map(|x| x * x).sum::<f32>();
        let rms_low_out: f32 = output_low[n / 2..].iter().map(|x| x * x).sum::<f32>();

        // Reset for high frequency test
        ind.prepare(44100.0);
        let mut input_high = vec![0.0f32; n];
        let mut output_high = vec![0.0f32; n];
        for i in 0..n {
            input_high[i] = (2.0 * std::f32::consts::PI * 10000.0 * i as f32 / 44100.0).sin();
        }
        ind.process_block(&input_high, &mut output_high);

        let rms_high_in: f32 = input_high[n / 2..].iter().map(|x| x * x).sum::<f32>();
        let rms_high_out: f32 = output_high[n / 2..].iter().map(|x| x * x).sum::<f32>();

        let ratio_low = rms_low_out / rms_low_in;
        let ratio_high = rms_high_out / rms_high_in;

        assert!(
            ratio_low > ratio_high,
            "Inductor should pass low freq better than high: ratio_low={ratio_low}, ratio_high={ratio_high}"
        );
    }
}
