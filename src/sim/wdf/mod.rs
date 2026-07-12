//! Wave Digital Filter (WDF) framework.
//!
//! Provides the building blocks for modeling analog circuits using
//! wave digital filter theory. Components are composed into binary trees
//! via series and parallel adaptors, with nonlinear elements solved at
//! the tree root.
//!
//! # Architecture
//!
//! A WDF tree is built bottom-up:
//! 1. Leaf nodes: passive one-port elements (resistor, capacitor, inductor)
//! 2. Internal nodes: series or parallel adaptors joining two subtrees
//! 3. Root: a nonlinear element or voltage/current source
//!
//! Processing follows two passes per sample:
//! - **Bottom-up (reflected):** each node computes its reflected wave from children
//! - **Top-down (incident):** the root solves for the incident wave and propagates down

pub mod adaptors;
pub mod components;
pub mod ports;

pub use adaptors::{ParallelAdaptor, SeriesAdaptor};
pub use components::{WdfCapacitor, WdfComponent, WdfIdealVoltageSource, WdfInductor, WdfResistor};
pub use ports::{DiodePairPort, DiodePort, NonlinearPort};
