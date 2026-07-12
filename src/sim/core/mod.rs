//! Core infrastructure: Newton solver, circuit graph, signal buffers.
//!
//! - [`NewtonSolver`] -- generic Newton-Raphson with voltage clamping and step limiting
//! - [`circuit_graph::CircuitGraph`] -- directed processing graph with topological sort
//!   and zero-delay-feedback cluster detection
//! - [`node::SignalBuffer`] -- preallocated aligned audio buffers
//! - [`buffer_pool::BufferPool`] -- allocation-free buffer management

/// Allocation-free pool of [`node::SignalBuffer`]s.
pub mod buffer_pool;
/// Processing graph with topological sort and ZDF cluster detection.
pub mod circuit_graph;
/// Fixed-capacity, cache-aligned signal buffers ([`node::SignalBuffer`]).
pub mod node;
/// Generic Newton-Raphson solver ([`solver::NewtonSolver`]).
pub mod solver;

pub use node::{AlignedBuffer, NodeId, SignalBuffer};
pub use solver::NewtonSolver;
