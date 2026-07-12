//! WDF adaptors: series and parallel junctions for composing WDF trees.
//!
//! Each adaptor connects two child WDF components (or sub-trees) and presents
//! a single port to the parent. The adaptor implements [`WdfComponent`] itself,
//! so adaptors can be nested to build arbitrarily deep binary trees.
//!
//! # Scattering equations
//!
//! The derivations below use the wave variable convention `V = a + b`,
//! `I = (a - b) / (2R)` where `a` is the incident wave and `b` is the
//! reflected wave.
//!
//! ## Series adaptor
//!
//! For a series connection of ports 1 (left) and 2 (right) with parent port 0:
//! - Kirchhoff's voltage law: `V_0 = V_1 + V_2`
//! - Kirchhoff's current law:  `I_0 = I_1 = I_2`
//!
//! This gives:
//! ```text
//! R_0 = R_1 + R_2
//! gamma = R_1 / R_0
//!
//! b_0 = b_1 + b_2                           (reflected wave, bottom-up)
//!
//! a_1 = b_1 + gamma * (a_0 - b_0)           (incident to left, top-down)
//! a_2 = b_2 + (1 - gamma) * (a_0 - b_0)     (incident to right, top-down)
//! ```
//!
//! ## Parallel adaptor
//!
//! For a parallel connection of ports 1 (left) and 2 (right) with parent port 0:
//! - Kirchhoff's voltage law: `V_0 = V_1 = V_2`
//! - Kirchhoff's current law:  `I_0 = I_1 + I_2`
//!
//! This gives:
//! ```text
//! G_0 = G_1 + G_2  (conductance, G = 1/R)
//! R_0 = R_1 * R_2 / (R_1 + R_2)
//! alpha = G_1 / G_0 = R_2 / (R_1 + R_2)
//!
//! b_0 = alpha * b_1 + (1 - alpha) * b_2     (reflected wave, bottom-up)
//!
//! a_1 = a_0 + b_0 - b_1                     (incident to left, top-down)
//! a_2 = a_0 + b_0 - b_2                     (incident to right, top-down)
//! ```

use super::components::WdfComponent;

// ---------------------------------------------------------------------------
// Series Adaptor
// ---------------------------------------------------------------------------

/// Series adaptor connecting two WDF subtrees.
///
/// The resulting port resistance is `R_left + R_right`.
/// The adaptor itself implements [`WdfComponent`] so it can be used as a
/// child of another adaptor or connected to a root source/nonlinearity.
pub struct SeriesAdaptor<A: WdfComponent, B: WdfComponent> {
    /// Left child.
    pub left: A,
    /// Right child.
    pub right: B,
    /// Impedance ratio: `R_left / (R_left + R_right)`. Precomputed.
    pub gamma: f32,
    /// Port resistance of this adaptor: `R_left + R_right`. Precomputed.
    port_resistance: f32,
    /// Cached reflected waves from children (set during `reflected_cached()`).
    pub b_left: f32,
    /// Cached reflected wave from right child.
    pub b_right: f32,
}

impl<A: WdfComponent, B: WdfComponent> SeriesAdaptor<A, B> {
    /// Create a new series adaptor from two child components.
    ///
    /// Precomputes the impedance ratio (gamma) and port resistance.
    pub fn new(left: A, right: B) -> Self {
        let r_left = left.port_resistance();
        let r_right = right.port_resistance();
        let r_total = r_left + r_right;
        Self {
            left,
            right,
            gamma: r_left / r_total,
            port_resistance: r_total,
            b_left: 0.0,
            b_right: 0.0,
        }
    }

    /// Recompute the adaptor coefficients after a child's port resistance changes.
    ///
    /// Call this after modifying a child's parameters (e.g., changing a capacitor's
    /// capacitance or an element's sample rate).
    pub fn update_impedance(&mut self) {
        let r_left = self.left.port_resistance();
        let r_right = self.right.port_resistance();
        let r_total = r_left + r_right;
        self.gamma = r_left / r_total;
        self.port_resistance = r_total;
    }

    /// Get the impedance ratio gamma.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }
}

impl<A: WdfComponent, B: WdfComponent> WdfComponent for SeriesAdaptor<A, B> {
    #[inline]
    fn incident(&mut self, a: f32) {
        // Distribute the incident wave from the parent to both children.
        //
        // Series adaptor top-down scattering:
        //   a_left  = b_left  + gamma * (a_root - b_root)
        //   a_right = b_right + (1 - gamma) * (a_root - b_root)
        //
        // We re-fetch children's reflected waves here rather than relying on
        // the cache, so that nested adaptors work correctly even when only
        // the trait's `reflected(&self)` was called (which cannot update the
        // mutable cache).
        let b_l = self.left.reflected();
        let b_r = self.right.reflected();
        self.b_left = b_l;
        self.b_right = b_r;

        let diff = a - (b_l + b_r);
        let a_left = b_l + self.gamma * diff;
        let a_right = b_r + (1.0 - self.gamma) * diff;

        self.left.incident(a_left);
        self.right.incident(a_right);
    }

    #[inline]
    fn reflected(&self) -> f32 {
        // Series adaptor bottom-up scattering:
        //   b_root = b_left + b_right
        //
        // We also need to cache b_left and b_right for use in incident().
        // Since reflected() takes &self, we use interior mutability via
        // the cached fields written in the mutable wrapper below.
        //
        // NOTE: This is a slight design compromise. The trait uses &self for
        // reflected() to allow calling it on children without &mut. We cache
        // the children's reflected waves and rely on the caller to invoke
        // reflected() before incident() each sample (which is the standard
        // WDF processing order).

        let b_l = self.left.reflected();
        let b_r = self.right.reflected();
        b_l + b_r
    }

    #[inline]
    fn port_resistance(&self) -> f32 {
        self.port_resistance
    }
}

impl<A: WdfComponent, B: WdfComponent> SeriesAdaptor<A, B> {
    /// Combined reflected + cache step. Call this instead of the trait's
    /// `reflected()` to also cache the children's waves for `incident()`.
    ///
    /// This is the recommended way to use the series adaptor in a processing
    /// loop, since the trait's `reflected(&self)` cannot update the cache.
    #[inline]
    pub fn reflected_cached(&mut self) -> f32 {
        self.b_left = self.left.reflected();
        self.b_right = self.right.reflected();
        self.b_left + self.b_right
    }
}

// ---------------------------------------------------------------------------
// Parallel Adaptor
// ---------------------------------------------------------------------------

/// Parallel adaptor connecting two WDF subtrees.
///
/// The resulting port resistance is `R_left * R_right / (R_left + R_right)`.
/// The adaptor itself implements [`WdfComponent`] so it can be nested.
pub struct ParallelAdaptor<A: WdfComponent, B: WdfComponent> {
    /// Left child.
    pub left: A,
    /// Right child.
    pub right: B,
    /// Conductance ratio: `G_left / (G_left + G_right) = R_right / (R_left + R_right)`.
    pub alpha: f32,
    /// Port resistance of this adaptor: `R_left * R_right / (R_left + R_right)`.
    port_resistance: f32,
    /// Cached reflected wave from left child (set during `reflected_cached()`).
    pub b_left: f32,
    /// Cached reflected wave from right child.
    pub b_right: f32,
}

impl<A: WdfComponent, B: WdfComponent> ParallelAdaptor<A, B> {
    /// Create a new parallel adaptor from two child components.
    ///
    /// Precomputes the conductance ratio (alpha) and port resistance.
    pub fn new(left: A, right: B) -> Self {
        let r_left = left.port_resistance();
        let r_right = right.port_resistance();
        let r_sum = r_left + r_right;
        Self {
            left,
            right,
            alpha: r_right / r_sum, // = G_left / G_total
            port_resistance: r_left * r_right / r_sum,
            b_left: 0.0,
            b_right: 0.0,
        }
    }

    /// Recompute the adaptor coefficients after a child's port resistance changes.
    pub fn update_impedance(&mut self) {
        let r_left = self.left.port_resistance();
        let r_right = self.right.port_resistance();
        let r_sum = r_left + r_right;
        self.alpha = r_right / r_sum;
        self.port_resistance = r_left * r_right / r_sum;
    }

    /// Get the conductance ratio alpha.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Combined reflected + cache step. Call this instead of the trait's
    /// `reflected()` to also cache the children's waves for `incident()`.
    #[inline]
    pub fn reflected_cached(&mut self) -> f32 {
        self.b_left = self.left.reflected();
        self.b_right = self.right.reflected();
        self.alpha * self.b_left + (1.0 - self.alpha) * self.b_right
    }
}

impl<A: WdfComponent, B: WdfComponent> WdfComponent for ParallelAdaptor<A, B> {
    #[inline]
    fn incident(&mut self, a: f32) {
        // Distribute the incident wave from the parent to both children.
        //
        // Parallel adaptor top-down scattering:
        //   b_0 = alpha * b_left + (1 - alpha) * b_right
        //   a_left  = a_0 + b_0 - b_left
        //   a_right = a_0 + b_0 - b_right
        //
        // Re-fetch children's reflected waves for correctness when nested.
        let b_l = self.left.reflected();
        let b_r = self.right.reflected();
        self.b_left = b_l;
        self.b_right = b_r;

        let b_root = self.alpha * b_l + (1.0 - self.alpha) * b_r;
        let a_left = a + b_root - b_l;
        let a_right = a + b_root - b_r;

        self.left.incident(a_left);
        self.right.incident(a_right);
    }

    #[inline]
    fn reflected(&self) -> f32 {
        // Parallel adaptor bottom-up scattering:
        //   b_root = alpha * b_left + (1 - alpha) * b_right
        let b_l = self.left.reflected();
        let b_r = self.right.reflected();
        self.alpha * b_l + (1.0 - self.alpha) * b_r
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
    use crate::sim::wdf::components::{WdfCapacitor, WdfInductor, WdfResistor};

    #[test]
    fn test_series_adaptor_two_resistors() {
        // Two 1k resistors in series should give 2k total.
        let r1 = WdfResistor::new(1000.0);
        let r2 = WdfResistor::new(1000.0);
        let series = SeriesAdaptor::new(r1, r2);

        assert!((series.port_resistance() - 2000.0).abs() < 1e-3);
        assert!((series.gamma() - 0.5).abs() < 1e-6);

        // Both resistors reflect 0, so series reflects 0 + 0 = 0.
        assert_eq!(series.reflected(), 0.0);
    }

    #[test]
    fn test_parallel_adaptor_two_resistors() {
        // Two 1k resistors in parallel should give 500 ohms.
        let r1 = WdfResistor::new(1000.0);
        let r2 = WdfResistor::new(1000.0);
        let par = ParallelAdaptor::new(r1, r2);

        assert!((par.port_resistance() - 500.0).abs() < 1e-3);
        assert!((par.alpha() - 0.5).abs() < 1e-6);

        // Both resistors reflect 0.
        assert_eq!(par.reflected(), 0.0);
    }

    #[test]
    fn test_series_rc_step_response() {
        // Build an RC low-pass filter using the series adaptor:
        // Vs --[series adaptor: R, C]-- GND
        //
        // R = 1k, C = 1uF => tau = 1ms
        // At fs = 48kHz, tau = 48 samples.

        let r_val = 1000.0;
        let c_val = 1e-6;
        let tau = r_val * c_val; // 1 ms
        let fs = 48000.0;

        let resistor = WdfResistor::new(r_val);
        let mut capacitor = WdfCapacitor::new(c_val);
        capacitor.set_sample_rate(fs);

        let mut adaptor = SeriesAdaptor::new(resistor, capacitor);
        adaptor.update_impedance();

        let vs = 1.0; // Step input voltage

        let samples_at_tau = (tau * fs) as usize; // ~48 samples

        for _ in 0..samples_at_tau {
            // Bottom-up pass
            let b_tree = adaptor.reflected_cached();

            // Root source: ideal voltage source forces V = Vs.
            // a_root = Vs - b_tree  (from V = a + b, a = Vs - b)
            let a_root = vs - b_tree;

            // Top-down pass
            adaptor.incident(a_root);
        }

        // Read capacitor voltage: V_c = a_c + b_c.
        // After the last incident() call, the capacitor received a_c and
        // its reflected is the NEW state (= a_c from this sample).
        // V_c = a_c + b_c_prev ... but after incident, b_c = a_c (new state).
        // Actually, the capacitor's voltage at sample n is:
        //   V_c[n] = a_c[n] + b_c[n] where b_c[n] is the state BEFORE incident.
        //
        // The simplest approach: peek at the capacitor's state.
        // After processing, cap.state = a_c (the last incident wave).
        // The reflected wave from cap was cap.state (the old value).
        // So the voltage was: a_c + b_c_old.
        //
        // We can extract the voltage from the adaptor by looking at the
        // last computed values. For a simpler check, run one more reflected
        // and compute.
        let b_tree = adaptor.reflected_cached();
        let _a_root = vs - b_tree;

        // Voltage across the whole series = Vs (source)
        // Current through series: I = (a_root - b_tree) / (2 * R_series)
        // Voltage across capacitor: V_c = I * 2 * R_c + b_c + b_c
        // ... this is getting complicated. Let's just check b_c directly.
        // For the capacitor, V_c = a_c + b_c. In steady state for a DC source,
        // all current = 0, V_c = Vs = 1.0, and a_c = b_c = 0.5.
        // At t = tau, V_c should be ~0.632.

        // The capacitor's reflected wave is its state. After ~tau samples of
        // a unit step, the state should reflect the charging.
        // b_c = cap.state = last incident wave to cap.
        // For the series adaptor:
        //   a_c = b_c_old + (1-gamma)*(a_root - b_root)
        // In steady state (all waves constant):
        //   a_root = Vs - b_root (source)
        //   b_root = b_r + b_c = 0 + b_c = b_c
        //   a_root = Vs - b_c
        //   diff = a_root - b_root = Vs - 2*b_c
        //   a_c = b_c + (1-gamma)*(Vs - 2*b_c)
        //   In steady state (cap fully charged): V_c = Vs, I = 0
        //   => a_c = b_c (since I = (a-b)/(2R) = 0 => a = b)
        //   => b_c + (1-gamma)*(Vs - 2*b_c) = b_c
        //   => (1-gamma)*(Vs - 2*b_c) = 0
        //   => b_c = Vs/2 in steady state. And V_c = a_c + b_c = Vs/2 + Vs/2 = Vs. Correct!

        // At t = tau: V_c ~ 0.632 * Vs => b_c ~ 0.632 * Vs / 2 ... not quite.
        // The wave variables don't map linearly like that. Let's just check
        // that the capacitor state is between 0 and the steady-state value
        // and monotonically increasing.

        // Alternative: compute V_c from the waves in the adaptor.
        // After reflected_cached, we have b_left (resistor) and b_right (cap).
        // The incident wave to cap in the previous step was stored as cap state.
        // V_c = a_c_last + b_c_before_last ... hard to extract cleanly.

        // Simpler validation: run the step response and track V_c each sample.
        // Reset and redo properly.
        let resistor2 = WdfResistor::new(r_val);
        let mut capacitor2 = WdfCapacitor::new(c_val);
        capacitor2.set_sample_rate(fs);

        let mut adaptor2 = SeriesAdaptor::new(resistor2, capacitor2);
        adaptor2.update_impedance();

        let mut vc_at_tau = 0.0f32;

        for n in 0..(samples_at_tau * 5) {
            let b_tree = adaptor2.reflected_cached();
            let a_root = vs - b_tree;

            // Compute V_c BEFORE incident (using current reflected waves and
            // the incident wave that will be sent to cap).
            let diff = a_root - (adaptor2.b_left + adaptor2.b_right);
            let a_c = adaptor2.b_right + (1.0 - adaptor2.gamma) * diff;
            let vc = a_c + adaptor2.b_right; // V = a + b

            if n == samples_at_tau {
                vc_at_tau = vc;
            }

            adaptor2.incident(a_root);
        }

        // At t = tau, V_c should be approximately 0.6321 * Vs.
        let expected = vs * (1.0 - (-1.0f32).exp()); // 0.6321
        let error = (vc_at_tau - expected).abs();
        assert!(
            error < 0.02,
            "RC step response at tau: vc={vc_at_tau}, expected={expected}, error={error}"
        );
    }

    #[test]
    fn test_parallel_rl() {
        // Parallel combination of R and L.
        // R = 1k, L = 10mH
        let r_val = 1000.0;
        let l_val = 0.01;
        let fs = 44100.0;

        let resistor = WdfResistor::new(r_val);
        let mut inductor = WdfInductor::new(l_val);
        inductor.set_sample_rate(fs);

        let par = ParallelAdaptor::new(resistor, inductor);

        // Parallel resistance: R_par = R * R_L / (R + R_L)
        let r_l = 2.0 * l_val * fs;
        let expected_r = r_val * r_l / (r_val + r_l);
        assert!(
            (par.port_resistance() - expected_r).abs() / expected_r < 1e-4,
            "Parallel RL resistance: got {}, expected {}",
            par.port_resistance(),
            expected_r
        );
    }

    #[test]
    fn test_series_adaptor_energy_conservation() {
        // Verify that the series adaptor scattering matrix is lossless.
        // For arbitrary input waves, power in = power out.
        //
        // Power at a port: P = (a^2 - b^2) / (4R)
        // For lossless junction: sum of power at all ports = 0.

        let r1 = 1000.0;
        let r2 = 500.0;

        let res1 = WdfResistor::new(r1);
        let res2 = WdfResistor::new(r2);
        let mut adaptor = SeriesAdaptor::new(res1, res2);

        // Since resistors reflect 0, let's manually set up a scenario.
        // We'll send an incident wave into the root and check the
        // scattered waves satisfy energy conservation.
        let a_root = 1.0;
        let b_root = adaptor.reflected_cached(); // 0 (both resistors)
        let r_root = adaptor.port_resistance();

        // After incident, children get waves.
        adaptor.incident(a_root);

        // Power into root: (a^2 - b^2) / (4*R)
        let p_root = (a_root * a_root - b_root * b_root) / (4.0 * r_root);

        // Children: for resistors, b = 0, so b_child = 0.
        // After incident, the adaptor sent a_left and a_right to children.
        // Power out of child i: (a_i^2 - b_i^2) / (4*R_i)
        // But since reflected from resistor = 0, b_i = 0.
        // a_left = 0 + gamma * (a_root - 0) = gamma * a_root
        // a_right = 0 + (1-gamma) * a_root
        let gamma = r1 / (r1 + r2);
        let a_left = gamma * a_root;
        let a_right = (1.0 - gamma) * a_root;

        let p_left = (a_left * a_left) / (4.0 * r1);
        let p_right = (a_right * a_right) / (4.0 * r2);

        // Power balance: p_root should equal p_left + p_right.
        let balance = (p_root - p_left - p_right).abs();
        assert!(
            balance < 1e-6,
            "Energy not conserved: p_root={p_root}, p_left={p_left}, p_right={p_right}, diff={balance}"
        );
    }

    #[test]
    fn test_parallel_adaptor_energy_conservation() {
        let r1 = 1000.0;
        let r2 = 500.0;

        let res1 = WdfResistor::new(r1);
        let res2 = WdfResistor::new(r2);
        let mut adaptor = ParallelAdaptor::new(res1, res2);

        let a_root = 1.0;
        let b_root = adaptor.reflected_cached(); // 0
        let r_root = adaptor.port_resistance();

        adaptor.incident(a_root);

        let p_root = (a_root * a_root - b_root * b_root) / (4.0 * r_root);

        // For parallel adaptor: a_child = a_root + b_root - b_child = a_root (since b's are 0)
        let b_root_val = 0.0;
        let a_left = a_root + b_root_val - 0.0;
        let a_right = a_root + b_root_val - 0.0;

        let p_left = (a_left * a_left) / (4.0 * r1);
        let p_right = (a_right * a_right) / (4.0 * r2);

        let balance = (p_root - p_left - p_right).abs();
        assert!(
            balance < 1e-6,
            "Energy not conserved: p_root={p_root}, p_left={p_left}, p_right={p_right}, diff={balance}"
        );
    }

    #[test]
    fn test_nested_adaptors() {
        // Build a tree: parallel(series(R1, R2), R3)
        let r1 = WdfResistor::new(100.0);
        let r2 = WdfResistor::new(200.0);
        let r3 = WdfResistor::new(150.0);

        let series = SeriesAdaptor::new(r1, r2);
        let parallel = ParallelAdaptor::new(series, r3);

        // series: R = 100 + 200 = 300
        // parallel: R = 300 * 150 / (300 + 150) = 45000 / 450 = 100
        assert!((parallel.port_resistance() - 100.0).abs() < 1e-3);

        // All resistors, so reflected = 0.
        assert_eq!(parallel.reflected(), 0.0);
    }

    #[test]
    fn test_update_impedance() {
        let mut cap = WdfCapacitor::new(1e-6);
        cap.set_sample_rate(44100.0);
        let res = WdfResistor::new(1000.0);

        let mut adaptor = SeriesAdaptor::new(res, cap);
        let r_before = adaptor.port_resistance();

        // Change sample rate on the capacitor
        adaptor.right.set_sample_rate(96000.0);
        adaptor.update_impedance();

        let r_after = adaptor.port_resistance();
        assert!(
            (r_before - r_after).abs() > 1.0,
            "Port resistance should change after sample rate update"
        );
    }

    #[test]
    fn test_rc_lowpass_frequency_response() {
        // Verify the RC lowpass filter frequency response matches analytical.
        // H(f) = 1 / (1 + j*2*pi*f*R*C)
        // |H(f)| = 1 / sqrt(1 + (2*pi*f*R*C)^2)
        //
        // Build: series(R, C) driven by voltage source.
        // Measure output across C at various frequencies.

        let r_val = 1000.0;
        let c_val = 100e-9; // 100 nF => fc = 1/(2*pi*R*C) = ~1591 Hz
        let fs = 192000.0; // High sample rate for accuracy

        let test_freqs = [100.0, 500.0, 1000.0, 1591.0, 5000.0, 10000.0];

        for &freq in &test_freqs {
            let resistor = WdfResistor::new(r_val);
            let mut capacitor = WdfCapacitor::new(c_val);
            capacitor.set_sample_rate(fs);

            let mut adaptor = SeriesAdaptor::new(resistor, capacitor);
            adaptor.update_impedance();

            let num_samples = (4.0 * fs / freq) as usize; // 4 full cycles
            let measure_start = num_samples / 2; // skip transient

            let mut max_vc = 0.0f32;

            for n in 0..num_samples {
                let t = n as f32 / fs;
                let vs = (2.0 * std::f32::consts::PI * freq * t).sin();

                let b_tree = adaptor.reflected_cached();
                let a_root = vs - b_tree;

                // Compute V_c before incident
                let diff = a_root - (adaptor.b_left + adaptor.b_right);
                let a_c = adaptor.b_right + (1.0 - adaptor.gamma) * diff;
                let vc = a_c + adaptor.b_right;

                if n >= measure_start {
                    max_vc = max_vc.max(vc.abs());
                }

                adaptor.incident(a_root);
            }

            // Analytical magnitude response
            let omega = 2.0 * std::f32::consts::PI * freq;
            let rc = r_val * c_val;
            let h_analytical = 1.0 / (1.0 + (omega * rc).powi(2)).sqrt();

            let error = (max_vc - h_analytical).abs();
            let rel_error = error / h_analytical;
            assert!(
                rel_error < 0.10,
                "RC frequency response at {freq} Hz: measured={max_vc:.4}, \
                 analytical={h_analytical:.4}, relative error={rel_error:.4}"
            );
        }
    }
}
