//! RLC filter components using Wave Digital Filter trees.
//!
//! These combine WDF resistors, capacitors, and inductors via series/parallel
//! adaptors to build standard filter topologies. Each filter implements
//! [`CircuitComponent`] for block-based audio processing.

use crate::sim::components::CircuitComponent;
use crate::sim::wdf::adaptors::SeriesAdaptor;
use crate::sim::wdf::components::{WdfCapacitor, WdfComponent, WdfInductor, WdfResistor};

// ---------------------------------------------------------------------------
// RC Low-Pass Filter
// ---------------------------------------------------------------------------

/// First-order RC low-pass filter using a WDF tree.
///
/// ```text
/// Circuit: Vin --[R]--+-- Vout
///                     |
///                    [C]
///                     |
///                    GND
/// ```
///
/// WDF tree topology: series(R, C), output measured across C.
/// Cutoff frequency: `fc = 1 / (2 * pi * R * C)`.
pub struct RcLowPass {
    adaptor: SeriesAdaptor<WdfResistor, WdfCapacitor>,
    sample_rate: f32,
}

impl RcLowPass {
    /// Create an RC low-pass filter.
    ///
    /// `resistance`: resistance in ohms.
    /// `capacitance`: capacitance in farads.
    pub fn new(resistance: f32, capacitance: f32) -> Self {
        let res = WdfResistor::new(resistance);
        let cap = WdfCapacitor::new(capacitance);
        let adaptor = SeriesAdaptor::new(res, cap);
        Self {
            adaptor,
            sample_rate: 44100.0,
        }
    }

    /// Create from a target cutoff frequency and resistance.
    ///
    /// Computes `C = 1 / (2 * pi * fc * R)`.
    pub fn from_cutoff(cutoff_hz: f32, resistance: f32) -> Self {
        let c = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz * resistance);
        Self::new(resistance, c)
    }

    /// Get the cutoff frequency in Hz.
    pub fn cutoff_hz(&self) -> f32 {
        let r = self.adaptor.left.resistance();
        let c = self.adaptor.right.capacitance();
        1.0 / (2.0 * std::f32::consts::PI * r * c)
    }
}

impl CircuitComponent for RcLowPass {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.adaptor.right.set_sample_rate(sample_rate);
        self.adaptor.right.reset();
        self.adaptor.update_impedance();
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            // Bottom-up
            let b_tree = self.adaptor.reflected_cached();

            // Ideal voltage source: a = Vs - b
            let a_root = input[i] - b_tree;

            // Output = voltage across capacitor (right child).
            let diff = a_root - (self.adaptor.b_left + self.adaptor.b_right);
            let a_c = self.adaptor.b_right + (1.0 - self.adaptor.gamma) * diff;
            output[i] = a_c + self.adaptor.b_right; // V = a + b

            // Top-down
            self.adaptor.incident(a_root);
        }
    }

    fn update_parameters(&mut self) {}
}

// ---------------------------------------------------------------------------
// RC High-Pass Filter
// ---------------------------------------------------------------------------

/// First-order RC high-pass filter using a WDF tree.
///
/// ```text
/// Circuit: Vin --[C]--+-- Vout
///                     |
///                    [R]
///                     |
///                    GND
/// ```
///
/// WDF tree topology: series(C, R), output measured across R.
/// Cutoff frequency: `fc = 1 / (2 * pi * R * C)`.
pub struct RcHighPass {
    adaptor: SeriesAdaptor<WdfCapacitor, WdfResistor>,
    sample_rate: f32,
}

impl RcHighPass {
    /// Create an RC high-pass filter.
    pub fn new(resistance: f32, capacitance: f32) -> Self {
        let cap = WdfCapacitor::new(capacitance);
        let res = WdfResistor::new(resistance);
        let adaptor = SeriesAdaptor::new(cap, res);
        Self {
            adaptor,
            sample_rate: 44100.0,
        }
    }

    /// Create from a target cutoff frequency and resistance.
    pub fn from_cutoff(cutoff_hz: f32, resistance: f32) -> Self {
        let c = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz * resistance);
        Self::new(resistance, c)
    }

    /// Get the cutoff frequency in Hz.
    pub fn cutoff_hz(&self) -> f32 {
        let r = self.adaptor.right.resistance();
        let c = self.adaptor.left.capacitance();
        1.0 / (2.0 * std::f32::consts::PI * r * c)
    }
}

impl CircuitComponent for RcHighPass {
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

            // Output = voltage across resistor (right child).
            let diff = a_root - (self.adaptor.b_left + self.adaptor.b_right);
            let a_r = self.adaptor.b_right + (1.0 - self.adaptor.gamma) * diff;
            output[i] = a_r + self.adaptor.b_right;

            self.adaptor.incident(a_root);
        }
    }

    fn update_parameters(&mut self) {}
}

// ---------------------------------------------------------------------------
// RLC Band-Pass Filter
// ---------------------------------------------------------------------------

/// Second-order RLC band-pass filter using a WDF tree.
///
/// ```text
/// Vin --[L]--[C]--+-- Vout
///                 |
///                [R]
///                 |
///                GND
/// ```
///
/// WDF tree: series(series(L, C), R). Output across R.
///
/// Resonant frequency: `f0 = 1 / (2*pi*sqrt(L*C))`
/// Quality factor: `Q = (1/R) * sqrt(L/C)`
pub struct RlcBandPass {
    /// Outer series adaptor: series(inner_lc, R).
    adaptor: SeriesAdaptor<SeriesAdaptor<WdfInductor, WdfCapacitor>, WdfResistor>,
    sample_rate: f32,
}

impl RlcBandPass {
    /// Create an RLC band-pass filter.
    ///
    /// `resistance`: R in ohms.
    /// `inductance`: L in henries.
    /// `capacitance`: C in farads.
    pub fn new(resistance: f32, inductance: f32, capacitance: f32) -> Self {
        let ind = WdfInductor::new(inductance);
        let cap = WdfCapacitor::new(capacitance);
        let inner = SeriesAdaptor::new(ind, cap);
        let res = WdfResistor::new(resistance);
        let adaptor = SeriesAdaptor::new(inner, res);
        Self {
            adaptor,
            sample_rate: 44100.0,
        }
    }

    /// Create from a target center frequency, Q factor, and resistance.
    ///
    /// Computes L and C from: `f0 = 1/(2*pi*sqrt(LC))`, `Q = (1/R)*sqrt(L/C)`.
    pub fn from_frequency(center_hz: f32, q: f32, resistance: f32) -> Self {
        let omega0 = 2.0 * std::f32::consts::PI * center_hz;
        // From Q = (1/R)*sqrt(L/C) and omega0 = 1/sqrt(LC):
        // L = Q*R/omega0
        // C = 1/(Q*R*omega0)
        let l = q * resistance / omega0;
        let c = 1.0 / (q * resistance * omega0);
        Self::new(resistance, l, c)
    }

    /// Get the resonant frequency in Hz.
    pub fn resonant_hz(&self) -> f32 {
        let l = self.adaptor.left.left.inductance();
        let c = self.adaptor.left.right.capacitance();
        1.0 / (2.0 * std::f32::consts::PI * (l * c).sqrt())
    }
}

impl CircuitComponent for RlcBandPass {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.adaptor.left.left.set_sample_rate(sample_rate);
        self.adaptor.left.right.set_sample_rate(sample_rate);
        self.adaptor.left.left.reset();
        self.adaptor.left.right.reset();
        self.adaptor.left.update_impedance();
        self.adaptor.update_impedance();
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            let b_tree = self.adaptor.reflected_cached();
            let a_root = input[i] - b_tree;

            // Output = voltage across R (right child of outer adaptor).
            let diff = a_root - (self.adaptor.b_left + self.adaptor.b_right);
            let a_r = self.adaptor.b_right + (1.0 - self.adaptor.gamma) * diff;
            output[i] = a_r + self.adaptor.b_right;

            self.adaptor.incident(a_root);
        }
    }

    fn update_parameters(&mut self) {}
}

// ---------------------------------------------------------------------------
// RLC Low-Pass Filter (second order)
// ---------------------------------------------------------------------------

/// Second-order RLC low-pass filter using a WDF tree.
///
/// ```text
/// Vin --[R]--[L]--+-- Vout
///                 |
///                [C]
///                 |
///                GND
/// ```
///
/// WDF tree: series(series(R, L), C). Output across C.
///
/// Resonant frequency: `f0 = 1 / (2*pi*sqrt(L*C))`
/// Damping: controlled by R.
pub struct RlcLowPass {
    adaptor: SeriesAdaptor<SeriesAdaptor<WdfResistor, WdfInductor>, WdfCapacitor>,
    sample_rate: f32,
}

impl RlcLowPass {
    /// Create an RLC low-pass filter.
    pub fn new(resistance: f32, inductance: f32, capacitance: f32) -> Self {
        let res = WdfResistor::new(resistance);
        let ind = WdfInductor::new(inductance);
        let inner = SeriesAdaptor::new(res, ind);
        let cap = WdfCapacitor::new(capacitance);
        let adaptor = SeriesAdaptor::new(inner, cap);
        Self {
            adaptor,
            sample_rate: 44100.0,
        }
    }

    /// Create from target cutoff frequency and damping ratio.
    ///
    /// `zeta` is the damping ratio (0.707 for Butterworth).
    pub fn from_cutoff(cutoff_hz: f32, zeta: f32, resistance: f32) -> Self {
        let omega0 = 2.0 * std::f32::consts::PI * cutoff_hz;
        // omega0 = 1/sqrt(LC), zeta = R/(2) * sqrt(C/L)
        // => L = R/(2*zeta*omega0), C = 2*zeta/(R*omega0)
        let l = resistance / (2.0 * zeta * omega0);
        let c = 2.0 * zeta / (resistance * omega0);
        Self::new(resistance, l, c)
    }

    /// Get the resonant frequency in Hz.
    pub fn resonant_hz(&self) -> f32 {
        let l = self.adaptor.left.right.inductance();
        let c = self.adaptor.right.capacitance();
        1.0 / (2.0 * std::f32::consts::PI * (l * c).sqrt())
    }
}

impl CircuitComponent for RlcLowPass {
    fn prepare(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.adaptor.left.right.set_sample_rate(sample_rate);
        self.adaptor.right.set_sample_rate(sample_rate);
        self.adaptor.left.right.reset();
        self.adaptor.right.reset();
        self.adaptor.left.update_impedance();
        self.adaptor.update_impedance();
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            let b_tree = self.adaptor.reflected_cached();
            let a_root = input[i] - b_tree;

            // Output = voltage across C (right child of outer adaptor).
            let diff = a_root - (self.adaptor.b_left + self.adaptor.b_right);
            let a_c = self.adaptor.b_right + (1.0 - self.adaptor.gamma) * diff;
            output[i] = a_c + self.adaptor.b_right;

            self.adaptor.incident(a_root);
        }
    }

    fn update_parameters(&mut self) {}
}

// ---------------------------------------------------------------------------
// Generic RLC Filter (user-configurable topology)
// ---------------------------------------------------------------------------

/// A flexible RLC filter wrapping a WDF adaptor tree as a `CircuitComponent`.
///
/// This provides a high-level interface for building common filter topologies.
/// For custom topologies, use the WDF primitives directly.
pub enum RlcFilter {
    /// First-order RC low-pass.
    RcLow(RcLowPass),
    /// First-order RC high-pass.
    RcHigh(RcHighPass),
    /// Second-order RLC band-pass.
    RlcBand(RlcBandPass),
    /// Second-order RLC low-pass.
    RlcLow(RlcLowPass),
}

impl CircuitComponent for RlcFilter {
    fn prepare(&mut self, sample_rate: f32) {
        match self {
            RlcFilter::RcLow(f) => f.prepare(sample_rate),
            RlcFilter::RcHigh(f) => f.prepare(sample_rate),
            RlcFilter::RlcBand(f) => f.prepare(sample_rate),
            RlcFilter::RlcLow(f) => f.prepare(sample_rate),
        }
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        match self {
            RlcFilter::RcLow(f) => f.process_block(input, output),
            RlcFilter::RcHigh(f) => f.process_block(input, output),
            RlcFilter::RlcBand(f) => f.process_block(input, output),
            RlcFilter::RlcLow(f) => f.process_block(input, output),
        }
    }

    fn update_parameters(&mut self) {
        match self {
            RlcFilter::RcLow(f) => f.update_parameters(),
            RlcFilter::RcHigh(f) => f.update_parameters(),
            RlcFilter::RlcBand(f) => f.update_parameters(),
            RlcFilter::RlcLow(f) => f.update_parameters(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]
    use super::*;

    #[test]
    fn test_rc_lowpass_dc_passthrough() {
        let mut lpf = RcLowPass::new(1000.0, 100e-9);
        lpf.prepare(192000.0);

        // DC should pass through a low-pass filter.
        let n = 19200; // 100ms at 192kHz
        let input = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];
        lpf.process_block(&input, &mut output);

        // After settling, output should approach input.
        let last = output[n - 1];
        assert!(
            (last - 1.0).abs() < 0.05,
            "DC should pass through LPF: last={last}"
        );
    }

    #[test]
    fn test_rc_lowpass_attenuates_high_freq() {
        let r = 1000.0;
        let c = 100e-9;
        let fc = 1.0 / (2.0 * std::f32::consts::PI * r * c); // ~1591 Hz
        let fs = 192000.0;

        // Test at 10x cutoff frequency — should be attenuated ~20 dB.
        let test_freq = fc * 10.0;

        let mut lpf = RcLowPass::new(r, c);
        lpf.prepare(fs);

        let n = (8.0 * fs / test_freq) as usize; // 8 cycles
        let mut input = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        for i in 0..n {
            input[i] = (2.0 * std::f32::consts::PI * test_freq * i as f32 / fs).sin();
        }

        lpf.process_block(&input, &mut output);

        // Measure peak in the second half (after settling).
        let peak_out: f32 = output[n / 2..]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        // At 10x cutoff, analytical attenuation = 1/sqrt(1+100) ≈ 0.0995 (-20 dB).
        assert!(
            peak_out < 0.2,
            "High freq should be attenuated: peak_out={peak_out}"
        );
    }

    #[test]
    fn test_rc_highpass_blocks_dc() {
        let mut hpf = RcHighPass::new(10000.0, 1e-6);
        hpf.prepare(44100.0);

        let n = 44100; // 1 second
        let input = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];
        hpf.process_block(&input, &mut output);

        // DC should be blocked.
        let last_avg: f32 = output[n - 1000..].iter().sum::<f32>() / 1000.0;
        assert!(
            last_avg.abs() < 0.01,
            "DC should be blocked by HPF: last_avg={last_avg}"
        );
    }

    #[test]
    fn test_rc_highpass_passes_high_freq() {
        let r = 10000.0;
        let c = 1e-6;
        let _fc = 1.0 / (2.0 * std::f32::consts::PI * r * c); // ~15.9 Hz
        let fs = 44100.0;

        let mut hpf = RcHighPass::new(r, c);
        hpf.prepare(fs);

        // 1kHz should pass through easily (well above cutoff).
        let test_freq = 1000.0;
        let n = (4.0 * fs / test_freq) as usize;
        let mut input = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        for i in 0..n {
            input[i] = (2.0 * std::f32::consts::PI * test_freq * i as f32 / fs).sin();
        }

        hpf.process_block(&input, &mut output);

        let peak_out: f32 = output[n / 2..]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        assert!(
            peak_out > 0.5,
            "High freq should pass through HPF: peak_out={peak_out}"
        );
    }

    #[test]
    fn test_rlc_bandpass_peak_at_resonance() {
        let f0 = 1000.0; // 1 kHz center
        let q = 5.0;
        let r = 100.0;

        let mut bpf = RlcBandPass::from_frequency(f0, q, r);
        bpf.prepare(192000.0);

        let fs = 192000.0;

        // Measure response at resonance, below, and above.
        let test_freqs = [200.0, 1000.0, 5000.0];
        let mut peaks = Vec::new();

        for &freq in &test_freqs {
            bpf.prepare(fs); // Reset state

            let n = (8.0 * fs / freq) as usize;
            let mut input = vec![0.0f32; n];
            let mut output = vec![0.0f32; n];

            for i in 0..n {
                input[i] = (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin();
            }

            bpf.process_block(&input, &mut output);

            let peak: f32 = output[n / 2..]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            peaks.push(peak);
        }

        // Response at resonance (1kHz) should be greater than off-resonance.
        assert!(
            peaks[1] > peaks[0],
            "BPF should peak at resonance: at_200Hz={}, at_1kHz={}",
            peaks[0],
            peaks[1]
        );
        assert!(
            peaks[1] > peaks[2],
            "BPF should peak at resonance: at_5kHz={}, at_1kHz={}",
            peaks[2],
            peaks[1]
        );
    }

    #[test]
    fn test_rlc_lowpass_second_order() {
        // Butterworth (zeta=0.707) second-order low-pass at 1kHz.
        let mut lpf = RlcLowPass::from_cutoff(1000.0, 0.707, 600.0);
        lpf.prepare(192000.0);

        let fs = 192000.0;

        // DC should pass.
        let n = 19200;
        let input = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];
        lpf.process_block(&input, &mut output);

        let dc_out = output[n - 1];
        assert!(
            (dc_out - 1.0).abs() < 0.1,
            "DC should pass through 2nd-order LPF: dc_out={dc_out}"
        );

        // High frequency (10kHz) should be heavily attenuated (2nd order = -40 dB/decade).
        lpf.prepare(fs);
        let test_freq = 10000.0;
        let n2 = (8.0 * fs / test_freq) as usize;
        let mut input2 = vec![0.0f32; n2];
        let mut output2 = vec![0.0f32; n2];

        for i in 0..n2 {
            input2[i] = (2.0 * std::f32::consts::PI * test_freq * i as f32 / fs).sin();
        }

        lpf.process_block(&input2, &mut output2);

        let peak_high: f32 = output2[n2 / 2..]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        assert!(
            peak_high < 0.05,
            "10kHz should be heavily attenuated by 2nd-order LPF: peak={peak_high}"
        );
    }

    #[test]
    fn test_rlc_filter_enum() {
        let mut filter = RlcFilter::RcLow(RcLowPass::new(1000.0, 100e-9));
        filter.prepare(44100.0);

        let input = [0.5f32; 64];
        let mut output = [0.0f32; 64];
        filter.process_block(&input, &mut output);

        // Should produce output without panicking.
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_rc_lowpass_frequency_response_accuracy() {
        // Quantitative check: at the cutoff frequency, gain should be -3dB (0.707).
        let r = 1000.0;
        let c = 100e-9;
        let fc = 1.0 / (2.0 * std::f32::consts::PI * r * c);
        let fs = 192000.0;

        let mut lpf = RcLowPass::new(r, c);
        lpf.prepare(fs);

        let n = (16.0 * fs / fc) as usize; // Many cycles for accuracy
        let mut input = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        for i in 0..n {
            input[i] = (2.0 * std::f32::consts::PI * fc * i as f32 / fs).sin();
        }

        lpf.process_block(&input, &mut output);

        let peak: f32 = output[n * 3 / 4..]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        // At cutoff: |H(fc)| = 1/sqrt(2) ≈ 0.707.
        let expected = 1.0 / 2.0f32.sqrt();
        let error = (peak - expected).abs();
        assert!(
            error < 0.05,
            "At cutoff freq ({fc:.0} Hz): peak={peak:.4}, expected={expected:.4}, error={error:.4}"
        );
    }

    #[test]
    fn test_no_nan_or_inf() {
        // Stress test: process a variety of inputs and verify no NaN/Inf.
        let filters: Vec<Box<dyn CircuitComponent>> = vec![
            Box::new(RcLowPass::new(1000.0, 100e-9)),
            Box::new(RcHighPass::new(10000.0, 1e-6)),
            Box::new(RlcBandPass::from_frequency(1000.0, 5.0, 100.0)),
            Box::new(RlcLowPass::from_cutoff(1000.0, 0.707, 600.0)),
        ];

        for mut filter in filters {
            filter.prepare(44100.0);

            // Impulse
            let mut input = vec![0.0f32; 1024];
            input[0] = 1.0;
            let mut output = vec![0.0f32; 1024];
            filter.process_block(&input, &mut output);

            for (i, &s) in output.iter().enumerate() {
                assert!(s.is_finite(), "Non-finite output at sample {i}: {s}");
            }
        }
    }
}
