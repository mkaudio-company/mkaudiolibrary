//! WDF one-port components: resistor, capacitor, inductor, voltage source.
//!
//! Each component implements the [`WdfComponent`] trait, which defines
//! the wave scattering interface used by adaptors to build WDF trees.

/// Core trait for all WDF one-port elements.
///
/// Wave variables:
/// - `a`: incident wave (arriving at the port)
/// - `b`: reflected wave (leaving the port)
/// - `R`: port resistance (characteristic impedance of the port)
///
/// Voltage and current at the port:
/// ```text
/// V = a + b
/// I = (a - b) / (2R)
/// ```
pub trait WdfComponent {
    /// Accept an incident wave from the parent adaptor.
    ///
    /// For reactive elements (capacitor, inductor), this updates internal
    /// state so that the next call to `reflected()` returns the correct wave.
    fn incident(&mut self, a: f32);

    /// Return the reflected wave for the current sample.
    ///
    /// Called during the bottom-up pass before `incident()` is called.
    fn reflected(&self) -> f32;

    /// Return the port resistance (ohms).
    ///
    /// For frequency-dependent elements, this depends on sample rate.
    fn port_resistance(&self) -> f32;
}

// ---------------------------------------------------------------------------
// Resistor
// ---------------------------------------------------------------------------

/// WDF ideal resistor.
///
/// A resistor absorbs all incident energy (matched termination).
/// Reflected wave is always zero.
pub struct WdfResistor {
    resistance: f32,
}

impl WdfResistor {
    /// Create a new WDF resistor with the given resistance in ohms.
    pub fn new(resistance: f32) -> Self {
        assert!(resistance > 0.0, "Resistance must be positive");
        Self { resistance }
    }

    /// Get the resistance value.
    pub fn resistance(&self) -> f32 {
        self.resistance
    }

    /// Set the resistance value.
    pub fn set_resistance(&mut self, r: f32) {
        assert!(r > 0.0, "Resistance must be positive");
        self.resistance = r;
    }
}

impl WdfComponent for WdfResistor {
    #[inline]
    fn incident(&mut self, _a: f32) {
        // Resistor is a matched load — nothing to store.
    }

    #[inline]
    fn reflected(&self) -> f32 {
        0.0
    }

    #[inline]
    fn port_resistance(&self) -> f32 {
        self.resistance
    }
}

// ---------------------------------------------------------------------------
// Capacitor
// ---------------------------------------------------------------------------

/// WDF capacitor using trapezoidal (bilinear) discretization.
///
/// Port resistance: `R = 1 / (2 * C * fs)`
///
/// The capacitor stores one sample of state. The reflected wave equals
/// the previous incident wave (unit delay in wave domain):
/// ```text
/// b[n] = a[n-1]
/// ```
pub struct WdfCapacitor {
    capacitance: f32,
    sample_rate: f32,
    /// Port resistance, recomputed when capacitance or sample rate changes.
    port_resistance: f32,
    /// State: the reflected wave for the current sample (= previous incident wave).
    state: f32,
}

impl WdfCapacitor {
    /// Create a new WDF capacitor with the given capacitance in farads.
    ///
    /// Call `set_sample_rate()` before processing to set the port resistance.
    pub fn new(capacitance: f32) -> Self {
        assert!(capacitance > 0.0, "Capacitance must be positive");
        Self {
            capacitance,
            sample_rate: 44100.0,
            port_resistance: 1.0 / (2.0 * capacitance * 44100.0),
            state: 0.0,
        }
    }

    /// Set the sample rate and recompute port resistance.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.port_resistance = 1.0 / (2.0 * self.capacitance * sample_rate);
    }

    /// Get the capacitance value in farads.
    pub fn capacitance(&self) -> f32 {
        self.capacitance
    }

    /// Set the capacitance and recompute port resistance.
    pub fn set_capacitance(&mut self, c: f32) {
        assert!(c > 0.0, "Capacitance must be positive");
        self.capacitance = c;
        self.port_resistance = 1.0 / (2.0 * c * self.sample_rate);
    }

    /// Reset the internal state (clear stored energy).
    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

impl WdfComponent for WdfCapacitor {
    #[inline]
    fn incident(&mut self, a: f32) {
        // Store incident wave as state for next sample's reflected wave.
        self.state = a;
    }

    #[inline]
    fn reflected(&self) -> f32 {
        self.state
    }

    #[inline]
    fn port_resistance(&self) -> f32 {
        self.port_resistance
    }
}

// ---------------------------------------------------------------------------
// Inductor
// ---------------------------------------------------------------------------

/// WDF inductor using trapezoidal (bilinear) discretization.
///
/// Port resistance: `R = 2 * L * fs`
///
/// The inductor reflects with a sign flip:
/// ```text
/// b[n] = -a[n-1]
/// ```
pub struct WdfInductor {
    inductance: f32,
    sample_rate: f32,
    /// Port resistance, recomputed when inductance or sample rate changes.
    port_resistance: f32,
    /// State: the reflected wave for the current sample (= negative previous incident).
    state: f32,
}

impl WdfInductor {
    /// Create a new WDF inductor with the given inductance in henries.
    ///
    /// Call `set_sample_rate()` before processing to set the port resistance.
    pub fn new(inductance: f32) -> Self {
        assert!(inductance > 0.0, "Inductance must be positive");
        Self {
            inductance,
            sample_rate: 44100.0,
            port_resistance: 2.0 * inductance * 44100.0,
            state: 0.0,
        }
    }

    /// Set the sample rate and recompute port resistance.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.port_resistance = 2.0 * self.inductance * sample_rate;
    }

    /// Get the inductance value in henries.
    pub fn inductance(&self) -> f32 {
        self.inductance
    }

    /// Set the inductance and recompute port resistance.
    pub fn set_inductance(&mut self, l: f32) {
        assert!(l > 0.0, "Inductance must be positive");
        self.inductance = l;
        self.port_resistance = 2.0 * l * self.sample_rate;
    }

    /// Reset the internal state (clear stored energy).
    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

impl WdfComponent for WdfInductor {
    #[inline]
    fn incident(&mut self, a: f32) {
        // Inductor reflects with sign inversion.
        self.state = -a;
    }

    #[inline]
    fn reflected(&self) -> f32 {
        self.state
    }

    #[inline]
    fn port_resistance(&self) -> f32 {
        self.port_resistance
    }
}

// ---------------------------------------------------------------------------
// Ideal Voltage Source
// ---------------------------------------------------------------------------

/// WDF ideal voltage source (Thevenin source with zero internal resistance).
///
/// Used as a root element to drive a WDF tree. The source sets the voltage
/// across its port; the reflected wave is computed to enforce `V = Vs`.
///
/// In wave variables: `V = a + b = Vs`, so `b = Vs - a`.
///
/// This element is typically connected at the root of the tree where
/// the "incident" wave is the reflected wave from the tree below,
/// and the "reflected" wave is what gets sent back down.
pub struct WdfIdealVoltageSource {
    /// Source voltage.
    voltage: f32,
    /// Port resistance (set to match the subtree for maximum power transfer,
    /// but the ideal source forces voltage regardless).
    port_resistance: f32,
    /// Stored incident wave.
    a: f32,
}

impl WdfIdealVoltageSource {
    /// Create a new ideal voltage source with port resistance `r`.
    ///
    /// The port resistance should match the subtree's port resistance
    /// to avoid numerical issues.
    pub fn new(port_resistance: f32) -> Self {
        Self {
            voltage: 0.0,
            port_resistance,
            a: 0.0,
        }
    }

    /// Set the source voltage.
    #[inline]
    pub fn set_voltage(&mut self, v: f32) {
        self.voltage = v;
    }

    /// Get the current source voltage.
    pub fn voltage(&self) -> f32 {
        self.voltage
    }

    /// Set the port resistance (should match the connected subtree).
    pub fn set_port_resistance(&mut self, r: f32) {
        self.port_resistance = r;
    }
}

impl WdfComponent for WdfIdealVoltageSource {
    #[inline]
    fn incident(&mut self, a: f32) {
        self.a = a;
    }

    #[inline]
    fn reflected(&self) -> f32 {
        // Enforce V = Vs: since V = a + b, we need b = Vs - a.
        // But here `a` is the wave coming from the subtree (their reflected),
        // and we return the wave going back (their incident).
        // For an ideal voltage source: b = 2*Vs - a (using V = (a+b) convention without /2)
        // Or b = Vs - a if using V = a + b.
        //
        // Using V = a + b convention (no factor of 2):
        self.voltage - self.a
    }

    #[inline]
    fn port_resistance(&self) -> f32 {
        self.port_resistance
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resistor_absorbs_energy() {
        let mut r = WdfResistor::new(1000.0);

        // Reflected wave should always be zero regardless of incident wave.
        assert_eq!(r.reflected(), 0.0);
        r.incident(1.0);
        assert_eq!(r.reflected(), 0.0);
        r.incident(-5.0);
        assert_eq!(r.reflected(), 0.0);

        assert_eq!(r.port_resistance(), 1000.0);
    }

    #[test]
    fn test_capacitor_state_update() {
        let mut c = WdfCapacitor::new(1e-6); // 1 uF
        c.set_sample_rate(48000.0);

        // Port resistance = 1/(2*C*fs) = 1/(2*1e-6*48000) = 10.4167
        let expected_r = 1.0 / (2.0 * 1e-6 * 48000.0);
        assert!((c.port_resistance() - expected_r).abs() < 1e-3);

        // Initial state should be zero.
        assert_eq!(c.reflected(), 0.0);

        // After incident(1.0), the state updates so next reflected() returns 1.0.
        c.incident(1.0);
        assert_eq!(c.reflected(), 1.0);

        // After incident(0.5), reflected returns 0.5.
        c.incident(0.5);
        assert_eq!(c.reflected(), 0.5);
    }

    #[test]
    fn test_inductor_sign_flip() {
        let mut l = WdfInductor::new(0.01); // 10 mH
        l.set_sample_rate(44100.0);

        // Port resistance = 2*L*fs = 2*0.01*44100 = 882
        let expected_r = 2.0 * 0.01 * 44100.0;
        assert!((l.port_resistance() - expected_r).abs() < 1e-2);

        // Initial state should be zero.
        assert_eq!(l.reflected(), 0.0);

        // After incident(1.0), state = -1.0 (sign flip).
        l.incident(1.0);
        assert_eq!(l.reflected(), -1.0);

        // After incident(-3.0), state = 3.0.
        l.incident(-3.0);
        assert_eq!(l.reflected(), 3.0);
    }

    #[test]
    fn test_capacitor_rc_step_response() {
        // Verify that a WDF capacitor + resistor in series driven by a step
        // voltage source produces the correct RC time constant.
        //
        // Circuit: Vs --[R]--+--[C]-- GND
        //
        // For this test, we manually compute the wave scattering rather than
        // using the adaptor (which is tested separately in adaptors.rs).
        //
        // RC time constant: tau = R*C
        // After t = tau, voltage across C should be ~63.2% of Vs.

        let r_val = 1000.0; // 1 kOhm
        let c_val = 1e-6; // 1 uF
        let tau = r_val * c_val; // 1 ms
        let fs = 48000.0;

        let mut cap = WdfCapacitor::new(c_val);
        cap.set_sample_rate(fs);

        let r_c = cap.port_resistance(); // 1/(2*C*fs)
        let r_r = r_val; // Resistor port resistance = R

        // Series adaptor parameters
        let r_total = r_r + r_c;
        let gamma = r_r / r_total;

        let vs = 1.0; // Step voltage

        let samples_at_tau = (tau * fs) as usize;

        let mut vc = 0.0f32; // voltage across capacitor

        for _ in 0..samples_at_tau {
            // Bottom-up: get reflected waves
            let b_r = 0.0; // resistor always reflects 0
            let b_c = cap.reflected(); // capacitor reflects stored state

            // Series adaptor reflected wave (goes up to source)
            let b_root = b_r + b_c;

            // Source sends incident wave into the tree.
            // For ideal voltage source: a_root = 2*Vs - b_root
            // (using V = a + b convention, source forces V = Vs,
            //  so a + b = Vs => a = Vs - b ... but this is the wave
            //  going INTO the tree from the source.)
            //
            // Actually, the source port sees b_root as the reflected wave
            // from the tree. The source sends back: a_root = 2*Vs - b_root
            // because V_source = (a_root + b_root)/1 = Vs (in normalized convention).
            //
            // Using V = a + b: a_root = Vs - b_root ... but let me use the
            // actual convention. In WDF with V = a + b:
            //   V_port = a + b, I_port = (a - b)/(2R)
            //   For the source port looking into the tree:
            //   b_root is reflected FROM tree, a_root is sent INTO tree.
            //   V_tree = a_root + b_root
            //   For Vs, we want a_root + b_root = Vs => a_root = Vs - b_root
            //
            // This is correct for an ideal voltage source.
            let a_root = vs - b_root;

            // Top-down: distribute incident waves to children
            // Series adaptor: current is the same through both children.
            // a_child = b_child + gamma_child * (a_root - b_root)
            // where gamma_child = R_child / R_total for the series adaptor.
            let diff = a_root - b_root;
            let a_r = b_r + gamma * diff; // incident to resistor
            let a_c = b_c + (1.0 - gamma) * diff; // incident to capacitor

            // Feed incident waves to components
            // Resistor doesn't need incident (but we call for completeness)
            let _ = a_r; // resistor ignores incident
            cap.incident(a_c);

            // Compute voltage across capacitor: V_c = a_c + b_c
            vc = a_c + b_c;
        }

        // After tau seconds, V_c should be approximately 1 - e^(-1) = 0.6321
        let expected = 1.0 - (-1.0f32).exp(); // 0.6321
        let error = (vc - expected).abs();
        assert!(
            error < 0.02,
            "RC step response error too large: vc={vc}, expected={expected}, error={error}"
        );
    }

    #[test]
    fn test_voltage_source() {
        let mut vs = WdfIdealVoltageSource::new(100.0);
        vs.set_voltage(5.0);

        // With incident wave 0, reflected should be Vs - 0 = 5.0
        vs.incident(0.0);
        assert_eq!(vs.reflected(), 5.0);

        // With incident wave 3, reflected should be 5 - 3 = 2.0
        vs.incident(3.0);
        assert_eq!(vs.reflected(), 2.0);
    }

    #[test]
    fn test_capacitor_reset() {
        let mut c = WdfCapacitor::new(1e-6);
        c.set_sample_rate(44100.0);
        c.incident(5.0);
        assert_eq!(c.reflected(), 5.0);
        c.reset();
        assert_eq!(c.reflected(), 0.0);
    }

    #[test]
    fn test_inductor_reset() {
        let mut l = WdfInductor::new(0.01);
        l.set_sample_rate(44100.0);
        l.incident(5.0);
        assert_eq!(l.reflected(), -5.0);
        l.reset();
        assert_eq!(l.reflected(), 0.0);
    }
}
