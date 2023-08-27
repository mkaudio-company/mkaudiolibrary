#![feature(new_uninit)]

//! Modular audio processing library including MKAU plugin format based on Rust.
//! buffer : includes push buffer and circular buffer.
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

/// Int24 type.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct I24
{
    bit0 : u8,
    bit1 : u8,
    bit2 : u8
}
impl I24
{
    /// Int24 data in i32 format to I24
    #[inline]
    pub fn new(input : i32) -> Self
    {
        return Self
        {
            bit0: (input >> 16) as u8,
            bit1: (input >> 8) as u8,
            bit2: input as u8,
        }
    }
    /// Buffer slice of I24 from file buffer.
    #[inline]
    pub fn from_slice(input : & [u8]) -> buffer::Buffer<Self>
    {
        let mut buffer = buffer::Buffer::new(input.len() / 3);
        for i in 0..buffer.len()
        {
            buffer[i] = Self
            {
                bit0: input[i * 3 + 2],
                bit1: input[i * 3 + 1],
                bit2: input[i * 3],
            }
        }
        return buffer;
    }
    /// Raw I24 data into i32
    #[inline]
    pub fn raw_i32(& self) -> i32
    {
        return (self.bit0 as i32) << 16 + (self.bit1 as i32) << 8 + (self.bit2 as i32);
    }
    #[inline]
    pub fn from_i32(input : i32) -> Self
    {
        let data = input / 256;
        return Self::new(data);
    }
    /// Convert I24 data into i32
    #[inline]
    pub fn to_i32(& self) -> i32
    {
        return ((self.bit0 as i32) << 16 + (self.bit1 as i32) << 8 + (self.bit2 as i32)) * 256;
    }
}