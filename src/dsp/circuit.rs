/// Trait for circuit components used in MNA simulation.
///
/// Components connect two nodes and contribute to the circuit's
/// admittance matrix (Y) and current source vector (J).
///
/// # Node Indexing
/// - Node 0 is ground (reference)
/// - Nodes 1+ are circuit nodes (1-indexed in the circuit, 0-indexed in arrays)
pub trait Component {
    /// Get the node indices this component connects (node_a, node_b).
    /// Node 0 represents ground.
    fn nodes(&self) -> (i32, i32);

    /// Return the equivalent conductance for the static Y matrix.
    /// Called once during preprocessing.
    fn get_conductance(&self, dt: f32) -> f32;

    /// Return the equivalent current source for the dynamic J vector.
    /// Called every sample for reactive components.
    fn get_current_source(&self, dt: f32) -> f32;

    /// Update internal state after solving node voltages.
    /// Called every sample for reactive components to store history.
    fn update_state(&mut self, v_a: f32, v_b: f32, dt: f32);
}

/// Resistor component (memoryless).
///
/// A linear resistor with constant conductance G = 1/R.
/// Contributes only to the static Y matrix.
pub struct Resistor {
    node_a: i32,
    node_b: i32,
    conductance: f32,
}
impl Resistor {
    /// Create a new resistor between two nodes.
    ///
    /// # Arguments
    /// * `n1`, `n2` - Node indices (0 = ground)
    /// * `resistance` - Resistance in Ohms
    pub fn new(n1: i32, n2: i32, resistance: f32) -> Self {
        Self {
            node_a: n1,
            node_b: n2,
            conductance: 1.0 / resistance,
        }
    }
}
impl Component for Resistor {
    fn nodes(&self) -> (i32, i32) {
        (self.node_a, self.node_b)
    }
    fn get_conductance(&self, _dt: f32) -> f32 {
        self.conductance
    }
    fn get_current_source(&self, _dt: f32) -> f32 {
        0.0
    }
    fn update_state(&mut self, _v_a: f32, _v_b: f32, _dt: f32) {}
}

/// Capacitor component with voltage memory.
///
/// Uses backward Euler companion model:
/// - Equivalent conductance: G = C / dt
/// - Equivalent current source: J = G * V_previous
pub struct Capacitor {
    node_a: i32,
    node_b: i32,
    capacitance: f32,
    prev_voltage: f32,
}
impl Capacitor {
    /// Create a new capacitor between two nodes.
    ///
    /// # Arguments
    /// * `n1`, `n2` - Node indices (0 = ground)
    /// * `capacitance` - Capacitance in Farads
    pub fn new(n1: i32, n2: i32, capacitance: f32) -> Self {
        Self {
            node_a: n1,
            node_b: n2,
            capacitance,
            prev_voltage: 0.0,
        }
    }
}
impl Component for Capacitor {
    fn nodes(&self) -> (i32, i32) {
        (self.node_a, self.node_b)
    }
    fn get_conductance(&self, dt: f32) -> f32 {
        self.capacitance / dt
    }
    fn get_current_source(&self, dt: f32) -> f32 {
        (self.capacitance / dt) * self.prev_voltage
    }
    fn update_state(&mut self, v_a: f32, v_b: f32, _dt: f32) {
        self.prev_voltage = v_a - v_b;
    }
}

/// Inductor component with current memory.
///
/// Uses backward Euler companion model:
/// - Equivalent conductance: G = dt / L
/// - Equivalent current source: J = -I_previous
pub struct Inductor {
    node_a: i32,
    node_b: i32,
    inductance: f32,
    prev_current: f32,
}
impl Inductor {
    /// Create a new inductor between two nodes.
    ///
    /// # Arguments
    /// * `n1`, `n2` - Node indices (0 = ground)
    /// * `inductance` - Inductance in Henrys
    pub fn new(n1: i32, n2: i32, inductance: f32) -> Self {
        Self {
            node_a: n1,
            node_b: n2,
            inductance,
            prev_current: 0.0,
        }
    }
}
impl Component for Inductor {
    fn nodes(&self) -> (i32, i32) {
        (self.node_a, self.node_b)
    }
    fn get_conductance(&self, dt: f32) -> f32 {
        dt / self.inductance
    }
    fn get_current_source(&self, _dt: f32) -> f32 {
        -self.prev_current
    }

    fn update_state(&mut self, v_a: f32, v_b: f32, dt: f32) {
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
pub struct Circuit {
    components: Vec<Box<dyn Component + Send + Sync>>,
    num_nodes: usize,
    y_static: Box<[f32]>,
    y_work: Box<[f32]>,
    j: Box<[f32]>,
    nodes: Box<[f32]>,
    dt: f32,
}

impl Circuit {
    /// Create a new circuit with the given sample rate and number of nodes.
    pub fn new(sample_rate: f32, num_nodes: usize) -> Self {
        let matrix_size = num_nodes * num_nodes;
        Self {
            components: Vec::new(),
            num_nodes,
            y_static: vec![0.0; matrix_size].into_boxed_slice(),
            y_work: vec![0.0; matrix_size].into_boxed_slice(),
            j: vec![0.0; num_nodes].into_boxed_slice(),
            nodes: vec![0.0; num_nodes].into_boxed_slice(),
            dt: 1.0 / sample_rate,
        }
    }

    /// Get the number of nodes in the circuit.
    pub fn get_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get the number of devices (components) in the circuit.
    pub fn get_devices(&self) -> usize {
        self.components.len()
    }

    /// Add a component to the circuit.
    pub fn add_component(&mut self, component: Box<dyn Component + Send + Sync>) {
        self.components.push(component);
    }

    /// Preprocess: Builds the static Y matrix.
    /// Call this ONCE before audio processing starts.
    pub fn preprocess(&mut self, impedance: f32) {
        let n = self.num_nodes;

        // Clear matrix
        self.y_static.fill(0.0);

        for comp in &self.components {
            let (n1, n2) = comp.nodes();
            let g = comp.get_conductance(self.dt);

            // Stamp Y matrix (0-indexed: Node 1 is index 0)
            if n1 > 0 {
                self.y_static[(n1 as usize - 1) * n + (n1 as usize - 1)] += g;
            }
            if n2 > 0 {
                self.y_static[(n2 as usize - 1) * n + (n2 as usize - 1)] += g;
            }

            if n1 > 0 && n2 > 0 {
                self.y_static[(n1 as usize - 1) * n + (n2 as usize - 1)] -= g;
                self.y_static[(n2 as usize - 1) * n + (n1 as usize - 1)] -= g;
            }
        }

        // Add source resistance for Node 1 (Input)
        if n >= 1 {
            self.y_static[0] += impedance;
        }
    }

    /// Solve the linear system Y * x = J using Gaussian elimination.
    fn solve_linear_system(&mut self) {
        let n = self.num_nodes;

        // Copy static Y to work Y
        self.y_work.copy_from_slice(&self.y_static);

        for i in 0..n {
            // Pivot selection
            let mut pivot = i;
            let mut max_val = self.y_work[i * n + i].abs();

            for k in (i + 1)..n {
                let val = self.y_work[k * n + i].abs();
                if val > max_val {
                    max_val = val;
                    pivot = k;
                }
            }

            // Swap rows
            if pivot != i {
                for col in i..n {
                    self.y_work.swap(i * n + col, pivot * n + col);
                }
                self.j.swap(i, pivot);
            }

            // Eliminate. Threshold is well above f32::EPSILON (~1.19e-7) so a
            // near-singular pivot is caught before division amplifies its
            // rounding error, rather than silently producing a huge result.
            let pivot_val = self.y_work[i * n + i];
            if pivot_val.abs() < 1e-6 {
                continue;
            }

            for k in (i + 1)..n {
                let factor = self.y_work[k * n + i] / pivot_val;
                for j in i..n {
                    self.y_work[k * n + j] -= factor * self.y_work[i * n + j];
                }
                self.j[k] -= factor * self.j[i];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += self.y_work[i * n + j] * self.nodes[j];
            }
            self.nodes[i] = (self.j[i] - sum) / self.y_work[i * n + i];
        }
    }

    /// Process a single sample through the circuit.
    /// - `input_voltage`: Input voltage at node 1
    /// - `probe_node`: Node to read output voltage from (1-indexed)
    pub fn process(&mut self, input_voltage: f32, probe_node: usize) -> f32 {
        let n = self.num_nodes;

        // Reset J vector
        self.j.fill(0.0);

        // Add input source (Norton equivalent at Node 1)
        let g_source = 1.0 / 0.1;
        self.j[0] += input_voltage * g_source;

        // Accumulate dynamic currents from components
        for comp in &self.components {
            let is = comp.get_current_source(self.dt);
            if is == 0.0 {
                continue;
            }

            let (n1, n2) = comp.nodes();
            if n1 > 0 {
                self.j[n1 as usize - 1] -= is;
            }
            if n2 > 0 {
                self.j[n2 as usize - 1] += is;
            }
        }

        // Solve for voltages
        self.solve_linear_system();

        // Update component states
        for comp in &mut self.components {
            let (n1, n2) = comp.nodes();
            let v1 = if n1 == 0 {
                0.0
            } else {
                self.nodes[n1 as usize - 1]
            };
            let v2 = if n2 == 0 {
                0.0
            } else {
                self.nodes[n2 as usize - 1]
            };
            comp.update_state(v1, v2, self.dt);
        }

        if probe_node == 0 || probe_node > n {
            return 0.0;
        }
        self.nodes[probe_node - 1]
    }
}
