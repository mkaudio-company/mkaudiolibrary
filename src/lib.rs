#![feature(integer_sign_cast)]

//! Modular audio processing library including MKAU plugin format based on Rust.
//! buffer : includes buffer, push buffer, and circular buffer.
//! dsp : includes convolution, saturation, compression, limiter, and delay for audio processing.
//! processor : includes MKAU plugin format.

//! # License
//! The library is offered under GPLv3.0 license for non-commercial use.
//! If you want to use mkaudiolibrary for closed source project, please email to minjaekim@mkaudio.company for agreement and support.

/// includes push buffer and circular buffer.
pub mod buffer;
/// includes convolution and saturation function for audio processing.
pub mod dsp;
/// includes MKAU plugin format.
pub mod processor;
/// includes audio file parsing library.
pub mod audiofile;