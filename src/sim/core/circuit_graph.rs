use super::buffer_pool::BufferPool;
use super::node::NodeId;
use crate::sim::components::CircuitComponent;

/// Edge in the circuit graph representing signal flow.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Upstream node this edge reads from.
    pub source: NodeId,
    /// Downstream node this edge feeds into.
    pub dest: NodeId,
}

/// A node wrapping a circuit component.
pub struct Node {
    /// This node's identifier within the owning [`CircuitGraph`].
    pub id: NodeId,
    /// The wrapped component that does the actual signal processing.
    pub component: Box<dyn CircuitComponent>,
}

/// Zero Delay Feedback cluster — a group of nodes in a feedback loop
/// that must be solved together using Newton iteration.
#[derive(Clone, Debug)]
pub struct ZdfCluster {
    /// Node IDs participating in this feedback cluster.
    pub nodes: Vec<NodeId>,
    /// Maximum Newton iterations per sample when solving this cluster.
    pub solver_max_iter: u32,
}

/// Compiled graph ready for block processing.
pub struct CompiledGraph {
    /// Topologically sorted execution order (excludes ZDF cluster internals).
    pub execution_order: Vec<NodeId>,
    /// ZDF feedback clusters requiring iterative solving.
    pub zdf_clusters: Vec<ZdfCluster>,
    /// Buffer assignment: node `i` uses buffer `buffer_assignments[i]`.
    pub buffer_assignments: Vec<usize>,
}

/// Directed graph connecting circuit components.
///
/// Builder pattern: `add_node()` → `connect()` → `compile()` → `process_block()`.
pub struct CircuitGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    compiled: Option<CompiledGraph>,
    buffer_pool: BufferPool,
    block_size: usize,
    sample_rate: f32,
}

impl CircuitGraph {
    /// Create an empty graph for the given block size and sample rate.
    pub fn new(block_size: usize, sample_rate: f32) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            compiled: None,
            buffer_pool: BufferPool::new(0, block_size),
            block_size,
            sample_rate,
        }
    }

    /// Add a component node. Returns its NodeId.
    pub fn add_node(&mut self, mut component: Box<dyn CircuitComponent>) -> NodeId {
        let id = self.nodes.len();
        component.prepare(self.sample_rate);
        self.nodes.push(Node { id, component });
        self.compiled = None; // invalidate
        id
    }

    /// Connect source node's output to dest node's input.
    pub fn connect(&mut self, source: NodeId, dest: NodeId) {
        assert!(source < self.nodes.len(), "invalid source node");
        assert!(dest < self.nodes.len(), "invalid dest node");
        self.edges.push(Edge { source, dest });
        self.compiled = None;
    }

    /// Compile the graph: topological sort, ZDF detection, buffer assignment.
    pub fn compile(&mut self) -> Result<(), GraphError> {
        let n = self.nodes.len();
        if n == 0 {
            self.compiled = Some(CompiledGraph {
                execution_order: Vec::new(),
                zdf_clusters: Vec::new(),
                buffer_assignments: Vec::new(),
            });
            return Ok(());
        }

        // Build adjacency list and in-degree
        let mut adj: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        let mut in_degree = vec![0u32; n];

        for edge in &self.edges {
            adj[edge.source].push(edge.dest);
            in_degree[edge.dest] += 1;
        }

        // Detect cycles using DFS
        let cycles = self.detect_cycles(&adj);

        // Topological sort (Kahn's algorithm), treating cycle nodes as ZDF clusters
        let cycle_nodes: std::collections::HashSet<NodeId> =
            cycles.iter().flat_map(|c| c.iter().copied()).collect();

        let mut queue: Vec<NodeId> = Vec::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 && !cycle_nodes.contains(&i) {
                queue.push(i);
            }
        }

        let mut execution_order = Vec::with_capacity(n);
        let mut visited = vec![false; n];

        // Process queue
        let mut head = 0;
        while head < queue.len() {
            let node = queue[head];
            head += 1;
            execution_order.push(node);
            visited[node] = true;

            for &next in &adj[node] {
                if !cycle_nodes.contains(&next) {
                    in_degree[next] -= 1;
                    if in_degree[next] == 0 {
                        queue.push(next);
                    }
                }
            }
        }

        // Add cycle nodes as ZDF clusters
        let zdf_clusters: Vec<ZdfCluster> = cycles
            .into_iter()
            .map(|nodes| ZdfCluster {
                nodes,
                solver_max_iter: 4,
            })
            .collect();

        // Insert ZDF cluster nodes into execution order at appropriate positions
        for cluster in &zdf_clusters {
            for &node_id in &cluster.nodes {
                if !visited[node_id] {
                    execution_order.push(node_id);
                    visited[node_id] = true;
                }
            }
        }

        // Any remaining unvisited nodes (disconnected)
        for (i, &v) in visited.iter().enumerate() {
            if !v {
                execution_order.push(i);
            }
        }

        // Buffer assignments: each node gets its own buffer
        let buffer_assignments: Vec<usize> = (0..n).collect();

        // Allocate buffer pool
        self.buffer_pool = BufferPool::new(n + 1, self.block_size); // +1 for temp

        self.compiled = Some(CompiledGraph {
            execution_order,
            zdf_clusters,
            buffer_assignments,
        });

        Ok(())
    }

    /// Detect cycles in the graph using DFS.
    fn detect_cycles(&self, adj: &[Vec<NodeId>]) -> Vec<Vec<NodeId>> {
        let n = adj.len();
        let mut visited = vec![0u8; n]; // 0=unvisited, 1=in-stack, 2=done
        let mut stack = Vec::new();
        let mut cycles = Vec::new();

        for start in 0..n {
            if visited[start] == 0 {
                self.dfs_cycle(start, adj, &mut visited, &mut stack, &mut cycles);
            }
        }

        cycles
    }

    fn dfs_cycle(
        &self,
        node: NodeId,
        adj: &[Vec<NodeId>],
        visited: &mut [u8],
        stack: &mut Vec<NodeId>,
        cycles: &mut Vec<Vec<NodeId>>,
    ) {
        visited[node] = 1;
        stack.push(node);

        for &next in &adj[node] {
            if visited[next] == 1 {
                // Found a cycle — collect nodes from `next` to end of stack
                if let Some(pos) = stack.iter().position(|&n| n == next) {
                    cycles.push(stack[pos..].to_vec());
                }
            } else if visited[next] == 0 {
                self.dfs_cycle(next, adj, visited, stack, cycles);
            }
        }

        stack.pop();
        visited[node] = 2;
    }

    /// Process a block of audio through the compiled graph.
    ///
    /// `input` is fed to the first node, `output` is read from the last node.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let compiled = match &self.compiled {
            Some(c) => c,
            None => {
                // Auto-compile if not yet compiled
                if self.compile().is_err() {
                    output.iter_mut().for_each(|s| *s = 0.0);
                    return;
                }
                self.compiled.as_ref().unwrap()
            }
        };

        let len = input.len().min(output.len()).min(self.block_size);

        if compiled.execution_order.is_empty() {
            output[..len].copy_from_slice(&input[..len]);
            return;
        }

        // Copy input into first node's input buffer
        let first_node = compiled.execution_order[0];
        let first_buf = compiled.buffer_assignments[first_node];
        self.buffer_pool.get_mut(first_buf).as_mut_slice()[..len].copy_from_slice(&input[..len]);

        // Process nodes in execution order
        let execution_order = compiled.execution_order.clone();
        let buffer_assignments = compiled.buffer_assignments.clone();

        for (order_idx, &node_id) in execution_order.iter().enumerate() {
            let buf_idx = buffer_assignments[node_id];

            // Determine input: from predecessor's buffer or the initial input buffer
            let temp_buf_idx = self.buffer_pool.len() - 1; // temp buffer

            // Find predecessor
            let pred = self
                .edges
                .iter()
                .find(|e| e.dest == node_id)
                .map(|e| e.source);

            if let Some(pred_id) = pred {
                let pred_buf = buffer_assignments[pred_id];
                // Copy predecessor output to temp
                let src_data: Vec<f32> = self.buffer_pool.get(pred_buf).as_slice()[..len].to_vec();
                self.buffer_pool.get_mut(temp_buf_idx).as_mut_slice()[..len]
                    .copy_from_slice(&src_data);
            } else if order_idx == 0 {
                // First node: input already copied above
                let src_data: Vec<f32> = self.buffer_pool.get(buf_idx).as_slice()[..len].to_vec();
                self.buffer_pool.get_mut(temp_buf_idx).as_mut_slice()[..len]
                    .copy_from_slice(&src_data);
            } else {
                // No predecessor, zero input
                self.buffer_pool.get_mut(temp_buf_idx).as_mut_slice()[..len]
                    .iter_mut()
                    .for_each(|s| *s = 0.0);
            }

            // Process: read from temp, write to node's buffer
            let input_data: Vec<f32> =
                self.buffer_pool.get(temp_buf_idx).as_slice()[..len].to_vec();
            let node = &mut self.nodes[node_id];
            node.component.process_block(
                &input_data,
                &mut self.buffer_pool.get_mut(buf_idx).as_mut_slice()[..len],
            );
        }

        // Process ZDF clusters (iterative solving)
        let zdf_clusters = compiled.zdf_clusters.clone();
        for cluster in &zdf_clusters {
            for _ in 0..cluster.solver_max_iter {
                for &node_id in &cluster.nodes {
                    let buf_idx = buffer_assignments[node_id];
                    let pred = self
                        .edges
                        .iter()
                        .find(|e| e.dest == node_id)
                        .map(|e| e.source);

                    let input_data = if let Some(pred_id) = pred {
                        let pred_buf = buffer_assignments[pred_id];
                        self.buffer_pool.get(pred_buf).as_slice()[..len].to_vec()
                    } else {
                        vec![0.0; len]
                    };

                    let node = &mut self.nodes[node_id];
                    node.component.process_block(
                        &input_data,
                        &mut self.buffer_pool.get_mut(buf_idx).as_mut_slice()[..len],
                    );
                }
            }
        }

        // Copy last node's output
        let last_node = *execution_order.last().unwrap();
        let last_buf = buffer_assignments[last_node];
        output[..len].copy_from_slice(&self.buffer_pool.get(last_buf).as_slice()[..len]);
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph has been compiled.
    pub fn is_compiled(&self) -> bool {
        self.compiled.is_some()
    }
}

/// Errors that can occur while building or compiling a [`CircuitGraph`].
#[derive(Debug)]
pub enum GraphError {
    /// The graph contains a feedback cycle that wasn't resolved into a
    /// [`ZdfCluster`] (should not occur via the normal `compile()` path,
    /// which detects and clusters cycles automatically).
    CycleDetected,
    /// An edge or lookup referenced a node ID that doesn't exist in the graph.
    InvalidNode(NodeId),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::CycleDetected => write!(f, "cycle detected in circuit graph"),
            GraphError::InvalidNode(id) => write!(f, "invalid node id: {id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple pass-through component for testing.
    struct PassThrough {
        gain: f32,
    }

    impl CircuitComponent for PassThrough {
        fn prepare(&mut self, _sample_rate: f32) {}
        fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
            for i in 0..input.len().min(output.len()) {
                output[i] = input[i] * self.gain;
            }
        }
        fn update_parameters(&mut self) {}
    }

    #[test]
    fn test_single_node() {
        let mut graph = CircuitGraph::new(64, 44100.0);
        let _n = graph.add_node(Box::new(PassThrough { gain: 2.0 }));
        graph.compile().unwrap();

        let input = [1.0f32; 64];
        let mut output = [0.0f32; 64];
        graph.process_block(&input, &mut output);

        for &s in &output {
            assert!((s - 2.0).abs() < 1e-6, "expected 2.0, got {s}");
        }
    }

    #[test]
    fn test_chain() {
        let mut graph = CircuitGraph::new(32, 44100.0);
        let n0 = graph.add_node(Box::new(PassThrough { gain: 2.0 }));
        let n1 = graph.add_node(Box::new(PassThrough { gain: 3.0 }));
        graph.connect(n0, n1);
        graph.compile().unwrap();

        let input = [1.0f32; 32];
        let mut output = [0.0f32; 32];
        graph.process_block(&input, &mut output);

        for &s in &output {
            assert!((s - 6.0).abs() < 1e-5, "expected 6.0, got {s}");
        }
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = CircuitGraph::new(16, 44100.0);
        let n0 = graph.add_node(Box::new(PassThrough { gain: 1.0 }));
        let n1 = graph.add_node(Box::new(PassThrough { gain: 1.0 }));
        let n2 = graph.add_node(Box::new(PassThrough { gain: 1.0 }));
        graph.connect(n0, n1);
        graph.connect(n1, n2);
        graph.compile().unwrap();

        let compiled = graph.compiled.as_ref().unwrap();
        // n0 must come before n1, n1 before n2
        let pos = |id: NodeId| {
            compiled
                .execution_order
                .iter()
                .position(|&n| n == id)
                .unwrap()
        };
        assert!(pos(n0) < pos(n1));
        assert!(pos(n1) < pos(n2));
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = CircuitGraph::new(16, 44100.0);
        let n0 = graph.add_node(Box::new(PassThrough { gain: 1.0 }));
        let n1 = graph.add_node(Box::new(PassThrough { gain: 1.0 }));
        graph.connect(n0, n1);
        graph.connect(n1, n0); // creates cycle

        graph.compile().unwrap();
        let compiled = graph.compiled.as_ref().unwrap();
        assert!(
            !compiled.zdf_clusters.is_empty(),
            "should detect ZDF cluster"
        );
    }

    #[test]
    fn test_empty_graph() {
        let mut graph = CircuitGraph::new(16, 44100.0);
        graph.compile().unwrap();

        let input = [1.0f32; 16];
        let mut output = [0.0f32; 16];
        graph.process_block(&input, &mut output);

        // Empty graph: input passes through
        for i in 0..16 {
            assert_eq!(output[i], input[i]);
        }
    }
}
