//! Digital signal processing components for real-time audio.
//!
//! This module provides a collection of DSP primitives including:
//!
//! - **Utility functions** - dB/ratio conversion
//! - **Convolution** - FIR filtering with impulse response
//! - **Saturation** - Asymmetric logarithmic waveshaping for analog-style harmonics
//! - **Circuit simulation** - Real-time transient analysis using Modified Nodal Analysis (MNA)
//! - **Dynamics** - Compression and limiting with envelope detection
//! - **Time-based** - Delay with feedback and wet/dry mix
//!
//! All processors use thread-safe buffers and are designed for real-time audio processing.
//!
//! ## Example: Processing with Saturation
//!
//! ```ignore
//! use mkaudiolibrary::buffer::Buffer;
//! use mkaudiolibrary::dsp::Saturation;
//!
//! let sat = Saturation::new(10.0, 10.0, 1.0, 1.0, 0.0, false);
//! let input = Buffer::from_slice(&[0.0, 0.5, 1.0, -0.5, -1.0]);
//! let output = Buffer::new(5);
//!
//! sat.run(&input, &output);
//! ```
//!
//! ## Example: Circuit Simulation
//!
//! ```ignore
//! use mkaudiolibrary::dsp::{Circuit, Resistor, Capacitor};
//!
//! // Create a simple RC lowpass filter
//! let mut circuit = Circuit::new(44100.0, 2);
//! circuit.add_component(Box::new(Resistor::new(1, 2, 1000.0)));   // 1kΩ
//! circuit.add_component(Box::new(Capacitor::new(2, 0, 1e-6)));    // 1µF
//! circuit.preprocess(10.0);
//!
//! let output = circuit.process(1.0, 2);  // Input 1V, probe node 2
//! ```

use std::alloc::LayoutError;
use no_denormals::*;

use crate::buffer::*;

// ==========================================
// Utility Functions
// ==========================================

/// Convert a linear ratio to decibels.
///
/// Formula: `dB = 20 * log10(ratio)`
#[inline]
pub fn ratio_to_db(ratio : f64) -> f64 { 20.0 * ratio.log10() }

/// Convert decibels to a linear ratio.
///
/// Formula: `ratio = 10^(dB / 20)`
#[inline]
pub fn db_to_ratio(db : f64) -> f64 { 10.0f64.powf(db / 20.0) }

// ==========================================
// Convolution
// ==========================================

/// FIR convolution processor with impulse response.
///
/// Performs discrete convolution of input signal with a kernel (impulse response).
/// Uses a thread-safe push buffer to maintain history across buffer boundaries,
/// enabling seamless streaming convolution.
///
/// # Thread Safety
/// The internal buffer uses `RwLock` for concurrent access.
pub struct Convolution
{
    buffer : PushBuffer<f64>,
    kernel : Box<[f64]>
}
impl Convolution
{
    /// Create a new convolution processor with the given impulse response.
    pub fn new(kernel : &[f64]) -> Result<Self, LayoutError>
    {
        let conv = Self
        {
            buffer : PushBuffer::<f64>::new(kernel.len())?,
            kernel : kernel.to_vec().into_boxed_slice()
        };
        conv.buffer.set_index(conv.buffer.len());
        Ok(conv)
    }

    /// Get the length of the impulse response.
    pub fn kernel_len(&self) -> usize { self.kernel.len() }

    /// Process a single sample through convolution.
    #[inline]
    pub fn process(&self, input : f64) -> f64
    {
        let mut guard = self.buffer.write();
        guard.push(input);
        let mut sum = 0.0;
        for i in 0..self.kernel.len()
        {
            sum += guard[i] * self.kernel[i];
        }
        sum
    }

    /// Convolve input buffer with impulse response, writing to output.
    pub fn run(&self, input : &Buffer<f64>, output : &Buffer<f64>)
    {
        let input_guard = input.read();
        let mut output_guard = output.write();
        let mut buffer_guard = self.buffer.write();

        no_denormals(||
        {
            for index in 0..input_guard.len().min(output_guard.len())
            {
                buffer_guard.push(input_guard[index]);
                let mut sum = 0.0;
                for i in 0..self.kernel.len()
                {
                    sum += buffer_guard[i] * self.kernel[i];
                }
                output_guard[index] = sum;
            }
        });
    }
}

// ==========================================
// Saturation (Numeric Modeling)
// ==========================================

/// Asymmetric logarithmic saturation model for analog-style harmonic generation.
///
/// This waveshaper uses a logarithmic curve with independent parameters for
/// positive and negative signal excursions, enabling asymmetric distortion
/// characteristics similar to tube amplifiers and tape saturation.
///
/// # Parameters
///
/// - **Alpha (α)** - Drive/knee control: Higher values create sharper knee transitions
/// - **Beta (β)** - Compression/gain: Controls output amplitude and compression amount
/// - **Delta (δ)** - DC bias: Shifts the crossover point between positive/negative curves
/// - **Gamma (γ)** - Polarity: Boolean to flip output polarity
///
/// # Transfer Function
///
/// For input `x` and bias `δ`:
/// - When `x >= δ`: `output = β+ * log2(1 + α+ * (x - δ)) / log2(1 + α+)`
/// - When `x < δ`:  `output = -β- * log2(1 + α- * (δ - x)) / log2(1 + α-)`
pub struct Saturation
{
    drive_alpha_plus : f64,
    drive_alpha_minus : f64,
    compression_beta_plus : f64,
    compression_beta_minus : f64,
    bias_delta : f64,
    flip_polarity : bool,
    norm_factor_plus : f64,
    norm_factor_minus : f64
}
impl Saturation
{
    /// Create a new saturation processor with asymmetric parameters.
    ///
    /// # Arguments
    /// * `alpha_plus` - Drive/knee for positive signal (higher = sharper knee)
    /// * `alpha_minus` - Drive/knee for negative signal
    /// * `beta_plus` - Output gain/compression for positive signal
    /// * `beta_minus` - Output gain/compression for negative signal
    /// * `delta_bias` - DC bias offset (shifts crossover point)
    /// * `flip` - Invert output polarity if true
    pub fn new(alpha_plus : f64, alpha_minus : f64,
               beta_plus : f64, beta_minus : f64,
               delta_bias : f64, flip : bool) -> Self
    {
        let drive_alpha_plus = alpha_plus.max(1e-4);
        let drive_alpha_minus = alpha_minus.max(1e-4);

        let norm_factor_plus = 1.0 / (1.0 + drive_alpha_plus).log2();
        let norm_factor_minus = 1.0 / (1.0 + drive_alpha_minus).log2();

        Self
        {
            drive_alpha_plus,
            drive_alpha_minus,
            compression_beta_plus : beta_plus,
            compression_beta_minus : beta_minus,
            bias_delta : delta_bias,
            flip_polarity : flip,
            norm_factor_plus,
            norm_factor_minus
        }
    }

    /// Process a single sample through the saturation curve.
    #[inline]
    pub fn process(&self, input_sample : f64) -> f64
    {
        let output_value = if input_sample >= self.bias_delta
        {
            // Processing positive side (x >= δ)
            let relative_input = input_sample - self.bias_delta;
            let log_out = (1.0 + self.drive_alpha_plus * relative_input).log2();
            self.compression_beta_plus * (log_out * self.norm_factor_plus)
        }
        else
        {
            // Processing negative side (x < δ)
            let relative_input = self.bias_delta - input_sample;
            let log_out = (1.0 + self.drive_alpha_minus * relative_input).log2();
            -self.compression_beta_minus * (log_out * self.norm_factor_minus)
        };

        if self.flip_polarity { -output_value } else { output_value }
    }

    /// Process a buffer of samples.
    pub fn run(&self, input : &Buffer<f64>, output : &Buffer<f64>)
    {
        let input_guard = input.read();
        let mut output_guard = output.write();

        no_denormals(||
        {
            for index in 0..input_guard.len().min(output_guard.len())
            {
                output_guard[index] = self.process(input_guard[index]);
            }
        });
    }
}

// ==========================================
// Circuit Simulation (Modified Nodal Analysis)
// ==========================================

/// Trait for circuit components used in MNA simulation.
///
/// Components connect two nodes and contribute to the circuit's
/// admittance matrix (Y) and current source vector (J).
///
/// # Node Indexing
/// - Node 0 is ground (reference)
/// - Nodes 1+ are circuit nodes (1-indexed in the circuit, 0-indexed in arrays)
pub trait Component
{
    /// Get the node indices this component connects (node_a, node_b).
    /// Node 0 represents ground.
    fn nodes(&self) -> (i32, i32);

    /// Return the equivalent conductance for the static Y matrix.
    /// Called once during preprocessing.
    fn get_conductance(&self, dt : f64) -> f64;

    /// Return the equivalent current source for the dynamic J vector.
    /// Called every sample for reactive components.
    fn get_current_source(&self, dt : f64) -> f64;

    /// Update internal state after solving node voltages.
    /// Called every sample for reactive components to store history.
    fn update_state(&mut self, v_a : f64, v_b : f64, dt : f64);
}

/// Resistor component (memoryless).
///
/// A linear resistor with constant conductance G = 1/R.
/// Contributes only to the static Y matrix.
pub struct Resistor
{
    node_a : i32,
    node_b : i32,
    conductance : f64
}
impl Resistor
{
    /// Create a new resistor between two nodes.
    ///
    /// # Arguments
    /// * `n1`, `n2` - Node indices (0 = ground)
    /// * `resistance` - Resistance in Ohms
    pub fn new(n1 : i32, n2 : i32, resistance : f64) -> Self
    {
        Self { node_a : n1, node_b : n2, conductance : 1.0 / resistance }
    }
}
impl Component for Resistor
{
    fn nodes(&self) -> (i32, i32) { (self.node_a, self.node_b) }
    fn get_conductance(&self, _dt : f64) -> f64 { self.conductance }
    fn get_current_source(&self, _dt : f64) -> f64 { 0.0 }
    fn update_state(&mut self, _v_a : f64, _v_b : f64, _dt : f64) {}
}

/// Capacitor component with voltage memory.
///
/// Uses backward Euler companion model:
/// - Equivalent conductance: G = C / dt
/// - Equivalent current source: J = G * V_previous
pub struct Capacitor
{
    node_a : i32,
    node_b : i32,
    capacitance : f64,
    prev_voltage : f64
}
impl Capacitor
{
    /// Create a new capacitor between two nodes.
    ///
    /// # Arguments
    /// * `n1`, `n2` - Node indices (0 = ground)
    /// * `capacitance` - Capacitance in Farads
    pub fn new(n1 : i32, n2 : i32, capacitance : f64) -> Self
    {
        Self { node_a : n1, node_b : n2, capacitance, prev_voltage : 0.0 }
    }
}
impl Component for Capacitor
{
    fn nodes(&self) -> (i32, i32) { (self.node_a, self.node_b) }
    fn get_conductance(&self, dt : f64) -> f64 { self.capacitance / dt }
    fn get_current_source(&self, dt : f64) -> f64 { (self.capacitance / dt) * self.prev_voltage }
    fn update_state(&mut self, v_a : f64, v_b : f64, _dt : f64) { self.prev_voltage = v_a - v_b; }
}

/// Inductor component with current memory.
///
/// Uses backward Euler companion model:
/// - Equivalent conductance: G = dt / L
/// - Equivalent current source: J = -I_previous
pub struct Inductor
{
    node_a : i32,
    node_b : i32,
    inductance : f64,
    prev_current : f64
}
impl Inductor
{
    /// Create a new inductor between two nodes.
    ///
    /// # Arguments
    /// * `n1`, `n2` - Node indices (0 = ground)
    /// * `inductance` - Inductance in Henrys
    pub fn new(n1 : i32, n2 : i32, inductance : f64) -> Self
    {
        Self { node_a : n1, node_b : n2, inductance, prev_current : 0.0 }
    }
}
impl Component for Inductor
{
    fn nodes(&self) -> (i32, i32) { (self.node_a, self.node_b) }
    fn get_conductance(&self, dt : f64) -> f64 { dt / self.inductance }
    fn get_current_source(&self, _dt : f64) -> f64 { -self.prev_current }

    fn update_state(&mut self, v_a : f64, v_b : f64, dt : f64)
    {
        let voltage = v_a - v_b;
        self.prev_current += (voltage * dt) / self.inductance;
    }
}

/// Real-time circuit simulation engine using Modified Nodal Analysis (MNA).
///
/// Solves linear circuits sample-by-sample using companion models for
/// reactive components (capacitors, inductors). The algorithm:
///
/// 1. **Preprocess** (once): Build static admittance matrix Y from components
/// 2. **Per-sample**: Update current vector J, solve Y*V=J, update component states
///
/// # Node Convention
/// - Node 0 is ground (implicit, not stored in arrays)
/// - Node 1 is the input node (voltage source injection point)
/// - Other nodes are numbered 2, 3, etc.
///
/// # Solver
/// Uses Gaussian elimination with partial pivoting for numerical stability.
pub struct Circuit
{
    components : Vec<Box<dyn Component + Send + Sync>>,
    num_nodes : usize,
    y_static : Box<[f64]>,
    y_work : Box<[f64]>,
    j : Box<[f64]>,
    nodes : Box<[f64]>,
    dt : f64
}

impl Circuit
{
    /// Create a new circuit with the given sample rate and number of nodes.
    pub fn new(sample_rate : f64, num_nodes : usize) -> Self
    {
        let matrix_size = num_nodes * num_nodes;
        Self
        {
            components : Vec::new(),
            num_nodes,
            y_static : vec![0.0; matrix_size].into_boxed_slice(),
            y_work : vec![0.0; matrix_size].into_boxed_slice(),
            j : vec![0.0; num_nodes].into_boxed_slice(),
            nodes : vec![0.0; num_nodes].into_boxed_slice(),
            dt : 1.0 / sample_rate
        }
    }

    /// Get the number of nodes in the circuit.
    pub fn get_nodes(&self) -> usize { self.num_nodes }

    /// Get the number of devices (components) in the circuit.
    pub fn get_devices(&self) -> usize { self.components.len() }

    /// Add a component to the circuit.
    pub fn add_component(&mut self, component : Box<dyn Component + Send + Sync>)
    {
        self.components.push(component);
    }

    /// Preprocess: Builds the static Y matrix.
    /// Call this ONCE before audio processing starts.
    pub fn preprocess(&mut self, impedance : f64)
    {
        let n = self.num_nodes;

        // Clear matrix
        self.y_static.fill(0.0);

        for comp in &self.components
        {
            let (n1, n2) = comp.nodes();
            let g = comp.get_conductance(self.dt);

            // Stamp Y matrix (0-indexed: Node 1 is index 0)
            if n1 > 0 { self.y_static[(n1 as usize - 1) * n + (n1 as usize - 1)] += g; }
            if n2 > 0 { self.y_static[(n2 as usize - 1) * n + (n2 as usize - 1)] += g; }

            if n1 > 0 && n2 > 0
            {
                self.y_static[(n1 as usize - 1) * n + (n2 as usize - 1)] -= g;
                self.y_static[(n2 as usize - 1) * n + (n1 as usize - 1)] -= g;
            }
        }

        // Add source resistance for Node 1 (Input)
        if n >= 1 { self.y_static[0] += impedance; }
    }

    /// Solve the linear system Y * x = J using Gaussian elimination.
    fn solve_linear_system(&mut self)
    {
        let n = self.num_nodes;

        // Copy static Y to work Y
        self.y_work.copy_from_slice(&self.y_static);

        for i in 0..n
        {
            // Pivot selection
            let mut pivot = i;
            let mut max_val = self.y_work[i * n + i].abs();

            for k in (i + 1)..n
            {
                let val = self.y_work[k * n + i].abs();
                if val > max_val { max_val = val; pivot = k; }
            }

            // Swap rows
            if pivot != i
            {
                for col in i..n
                {
                    self.y_work.swap(i * n + col, pivot * n + col);
                }
                self.j.swap(i, pivot);
            }

            // Eliminate
            let pivot_val = self.y_work[i * n + i];
            if pivot_val.abs() < 1e-9 { continue; }

            for k in (i + 1)..n
            {
                let factor = self.y_work[k * n + i] / pivot_val;
                for j in i..n
                {
                    self.y_work[k * n + j] -= factor * self.y_work[i * n + j];
                }
                self.j[k] -= factor * self.j[i];
            }
        }

        // Back substitution
        for i in (0..n).rev()
        {
            let mut sum = 0.0;
            for j in (i + 1)..n { sum += self.y_work[i * n + j] * self.nodes[j]; }
            self.nodes[i] = (self.j[i] - sum) / self.y_work[i * n + i];
        }
    }

    /// Process a single sample through the circuit.
    /// - `input_voltage`: Input voltage at node 1
    /// - `probe_node`: Node to read output voltage from (1-indexed)
    pub fn process(&mut self, input_voltage : f64, probe_node : usize) -> f64
    {
        let n = self.num_nodes;

        // Reset J vector
        self.j.fill(0.0);

        // Add input source (Norton equivalent at Node 1)
        let g_source = 1.0 / 0.1;
        self.j[0] += input_voltage * g_source;

        // Accumulate dynamic currents from components
        for comp in &self.components
        {
            let is = comp.get_current_source(self.dt);
            if is == 0.0 { continue; }

            let (n1, n2) = comp.nodes();
            if n1 > 0 { self.j[n1 as usize - 1] -= is; }
            if n2 > 0 { self.j[n2 as usize - 1] += is; }
        }

        // Solve for voltages
        self.solve_linear_system();

        // Update component states
        for comp in &mut self.components
        {
            let (n1, n2) = comp.nodes();
            let v1 = if n1 == 0 { 0.0 } else { self.nodes[n1 as usize - 1] };
            let v2 = if n2 == 0 { 0.0 } else { self.nodes[n2 as usize - 1] };
            comp.update_state(v1, v2, self.dt);
        }

        if probe_node == 0 || probe_node > n { return 0.0; }
        self.nodes[probe_node - 1]
    }
}

// ==========================================
// Dynamics Processing
// ==========================================

/// Dynamic range compressor with soft knee and envelope detection.
///
/// Reduces the dynamic range of audio by attenuating signals above a threshold.
/// Features smooth attack/release envelope following and optional soft knee
/// for more transparent compression.
///
/// # Parameters
/// - `threshold` - Level (dB) above which compression begins
/// - `ratio` - Compression ratio (e.g., 4.0 means 4:1 compression)
/// - `attack` - Time (ms) to reach full compression
/// - `release` - Time (ms) to return to unity gain
/// - `makeup` - Output gain (dB) to compensate for level reduction
/// - `knee` - Soft knee width (dB), 0 = hard knee
///
/// # Envelope Detection
/// Uses a one-pole lowpass filter with separate attack/release coefficients
/// for smooth gain reduction that follows the input signal envelope.
pub struct Compression
{
    /// Threshold in dB (signals above this are compressed).
    pub threshold : f64,
    /// Compression ratio (e.g., 4.0 for 4:1).
    pub ratio : f64,
    /// Attack time in milliseconds.
    pub attack : f64,
    /// Release time in milliseconds.
    pub release : f64,
    /// Makeup gain in dB.
    pub makeup : f64,
    /// Soft knee width in dB (0 = hard knee).
    pub knee : f64,
    envelope : f64,
    attack_coeff : f64,
    release_coeff : f64
}
impl Compression
{
    /// Create a new compressor with default parameters.
    ///
    /// Defaults: -20dB threshold, 4:1 ratio, 10ms attack, 100ms release.
    pub fn new(sample_rate : f64) -> Self
    {
        let mut comp = Self
        {
            threshold : -20.0,
            ratio : 4.0,
            attack : 10.0,
            release : 100.0,
            makeup : 0.0,
            knee : 0.0,
            envelope : 0.0,
            attack_coeff : 0.0,
            release_coeff : 0.0
        };
        comp.update_coefficients(sample_rate);
        comp
    }

    /// Update attack/release coefficients when parameters or sample rate change.
    pub fn update_coefficients(&mut self, sample_rate : f64)
    {
        self.attack_coeff = (-1.0 / (self.attack * 0.001 * sample_rate)).exp();
        self.release_coeff = (-1.0 / (self.release * 0.001 * sample_rate)).exp();
    }

    /// Compute gain reduction in dB for a given input level in dB.
    #[inline]
    fn compute_gain(&self, input_db : f64) -> f64
    {
        if self.knee > 0.0
        {
            let half_knee = self.knee * 0.5;
            let lower = self.threshold - half_knee;
            let upper = self.threshold + half_knee;

            if input_db <= lower { 0.0 }
            else if input_db >= upper
            {
                (self.threshold + (input_db - self.threshold) / self.ratio) - input_db
            }
            else
            {
                // Soft knee region
                let x = input_db - lower;
                let slope = 1.0 / self.ratio - 1.0;
                slope * x * x / (2.0 * self.knee)
            }
        }
        else
        {
            // Hard knee
            if input_db <= self.threshold { 0.0 }
            else { (self.threshold + (input_db - self.threshold) / self.ratio) - input_db }
        }
    }

    /// Process a buffer of samples.
    pub fn run(&mut self, input : &Buffer<f64>, output : &Buffer<f64>)
    {
        let input_guard = input.read();
        let mut output_guard = output.write();
        let makeup_linear = db_to_ratio(self.makeup);

        no_denormals(||
        {
            for index in 0..input_guard.len().min(output_guard.len())
            {
                let input_abs = input_guard[index].abs();
                let input_db = if input_abs > 1e-10 { 20.0 * input_abs.log10() } else { -200.0 };

                // Compute target gain reduction
                let target_gr = self.compute_gain(input_db);

                // Envelope follower (attack/release)
                let coeff = if target_gr < self.envelope { self.attack_coeff } else { self.release_coeff };
                self.envelope = target_gr + coeff * (self.envelope - target_gr);

                // Apply gain
                let gain = db_to_ratio(self.envelope) * makeup_linear;
                output_guard[index] = input_guard[index] * gain;
            }
        });
    }
}

/// Brickwall limiter with instant attack and smooth release.
///
/// Prevents output from exceeding a ceiling level by applying instant
/// gain reduction when needed, with smooth release back to unity gain.
/// No lookahead is used, making this suitable for real-time applications.
///
/// # Parameters
/// - `gain` - Input gain (dB) applied before limiting
/// - `ceiling` - Maximum output level (dB)
/// - `release` - Time (ms) to return to unity gain after limiting
///
/// # Behavior
/// - **Instant attack**: Gain reduction is applied immediately when needed
/// - **Smooth release**: One-pole filter smoothly returns to unity gain
pub struct Limit
{
    /// Input gain in dB (applied before limiting).
    pub gain : f64,
    /// Output ceiling in dB (maximum output level).
    pub ceiling : f64,
    /// Release time in milliseconds.
    pub release : f64,
    envelope : f64,
    release_coeff : f64
}
impl Limit
{
    /// Create a new limiter with default parameters.
    ///
    /// Defaults: 0dB gain, 0dB ceiling, 100ms release.
    pub fn new(sample_rate : f64) -> Self
    {
        let mut lim = Self
        {
            gain : 0.0,
            ceiling : 0.0,
            release : 100.0,
            envelope : 1.0,
            release_coeff : 0.0
        };
        lim.update_coefficients(sample_rate);
        lim
    }

    /// Update release coefficient when parameters or sample rate change.
    pub fn update_coefficients(&mut self, sample_rate : f64)
    {
        self.release_coeff = (-1.0 / (self.release * 0.001 * sample_rate)).exp();
    }

    /// Process a single sample.
    #[inline]
    pub fn process(&mut self, input : f64) -> f64
    {
        let gain_linear = db_to_ratio(self.gain);
        let ceiling_linear = db_to_ratio(self.ceiling);

        let amplified = input * gain_linear;
        let abs_sample = amplified.abs();

        // Compute required gain reduction
        let target = if abs_sample > ceiling_linear
        {
            ceiling_linear / abs_sample
        }
        else { 1.0 };

        // Instant attack, smooth release
        if target < self.envelope
        {
            self.envelope = target;
        }
        else
        {
            self.envelope = target + self.release_coeff * (self.envelope - target);
        }

        amplified * self.envelope
    }

    /// Process a buffer of samples.
    pub fn run(&mut self, input : &Buffer<f64>, output : &Buffer<f64>)
    {
        let input_guard = input.read();
        let mut output_guard = output.write();

        no_denormals(||
        {
            for index in 0..input_guard.len().min(output_guard.len())
            {
                output_guard[index] = self.process(input_guard[index]);
            }
        });
    }
}

// ==========================================
// Time-Based Effects
// ==========================================

/// Feedback delay line with wet/dry mix.
///
/// A simple delay effect using a circular buffer for the delay line.
/// Supports feedback for echo/repeat effects and wet/dry mixing.
///
/// # Parameters
/// - `time` - Delay time in milliseconds
/// - `feedback` - Amount of delayed signal fed back (0.0 to 1.0)
/// - `mix` - Wet/dry balance (0.0 = dry only, 1.0 = wet only)
///
/// # Thread Safety
/// Uses a thread-safe circular buffer internally, allowing the delay
/// to be used in multi-threaded audio processing contexts.
///
/// # Note
/// Feedback values >= 1.0 will cause infinite buildup. Use values < 1.0
/// for stable operation.
pub struct Delay
{
    time : f64,
    sample_rate : f64,
    /// Feedback amount (0.0 to 1.0, values >= 1.0 cause buildup).
    pub feedback : f64,
    /// Wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    pub mix : f64,
    buffer : CircularBuffer<f64>
}
impl Delay
{
    /// Create a new delay with the specified time and sample rate.
    ///
    /// # Arguments
    /// * `time` - Delay time in milliseconds
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(time : f64, sample_rate : f64) -> Self
    {
        let delay_samples = ((time * 0.001 * sample_rate) as usize).max(1);
        Self
        {
            time,
            sample_rate,
            feedback : 0.5,
            mix : 0.5,
            buffer : CircularBuffer::new(delay_samples).unwrap()
        }
    }

    /// Get the current delay time in ms.
    pub fn get_time(&self) -> f64 { self.time }

    /// Set the delay time in ms.
    pub fn set_time(&mut self, time : f64)
    {
        self.time = time;
        let delay_samples = ((time * 0.001 * self.sample_rate) as usize).max(1);
        self.buffer.resize(delay_samples).unwrap();
    }

    /// Set the sample rate and update buffer size accordingly.
    pub fn set_sample_rate(&mut self, sample_rate : f64)
    {
        self.sample_rate = sample_rate;
        let delay_samples = ((self.time * 0.001 * sample_rate) as usize).max(1);
        self.buffer.resize(delay_samples).unwrap();
    }

    /// Process a single sample (acquires buffer lock internally).
    #[inline]
    pub fn process(&self, input : f64) -> f64
    {
        let mut guard = self.buffer.write();
        let delayed = guard.next();
        guard.push(input + delayed * self.feedback);
        input * (1.0 - self.mix) + delayed * self.mix
    }

    /// Process a buffer of samples.
    pub fn run(&self, input : &Buffer<f64>, output : &Buffer<f64>)
    {
        let input_guard = input.read();
        let mut output_guard = output.write();
        let mut buffer_guard = self.buffer.write();

        no_denormals(||
        {
            for index in 0..input_guard.len().min(output_guard.len())
            {
                let delayed = buffer_guard.next();
                buffer_guard.push(input_guard[index] + delayed * self.feedback);
                output_guard[index] = input_guard[index] * (1.0 - self.mix) + delayed * self.mix;
            }
        });
    }
}