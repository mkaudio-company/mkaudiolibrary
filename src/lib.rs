#![feature(new_uninit)]

//! Modular audio processing library including MKAU plugin format based on Rust.
//! buffer : includes buffer, push buffer, and circular buffer.
//! simulation : includes convolution and saturation function for audio processing.
//! processor : includes MKAU plugin format.
//!
//! # License
//! The library is offered under GPLv3.0 license for non-commercial use.
//! If you want to use mkaudiolibrary for closed source project, please email to minjaekim@mkaudio.company for agreement and support.

/// includes push buffer and circular buffer.
pub mod buffer;
/// includes convolution and saturation function for audio processing.
pub mod simulation;
/// includes MKAU plugin format.
pub mod processor;