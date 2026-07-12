//! Circuit component models.
//!
//! Every component implements the [`CircuitComponent`] trait for block-based
//! audio processing. Available models:
//!
//! | Module             | Components                                    |
//! |--------------------|-----------------------------------------------|
//! | [`tubes`]          | `TriodeStage`, `PentodeStage` (Koren eqs)     |
//! | [`diode`]          | `DiodeClipper`, `AntiParallelDiodeClipper`    |
//! | [`transistor`]     | `BjtStage`, `MosfetStage`                     |
//! | [`opamp`]          | `OpAmpStage` (gain + slew + rail clamp)       |
//! | [`potentiometers`] | `Potentiometer` (linear/log/reverse-log)      |
//! | [`switches`]       | `Switch` (smoothed conductance)               |
//! | [`passive`]        | `Resistor`, `Capacitor`, `Inductor` wrappers  |
//! | [`rlc`]            | RLC filter topologies via WDF                 |

pub mod diode;
pub mod opamp;
pub mod passive;
pub mod potentiometers;
pub mod rlc;
pub mod switches;
pub mod transistor;
pub mod tubes;

/// Core trait implemented by all circuit components.
///
/// Components process audio in blocks and support real-time parameter updates.
pub trait CircuitComponent {
    /// Initialize/reset internal state for the given sample rate.
    fn prepare(&mut self, sample_rate: f32);

    /// Process a block of audio samples.
    fn process_block(&mut self, input: &[f32], output: &mut [f32]);

    /// Apply any pending parameter changes (smoothing, etc.).
    fn update_parameters(&mut self);
}
