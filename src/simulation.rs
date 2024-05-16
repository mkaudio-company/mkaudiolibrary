use std::alloc::LayoutError;

use crate::buffer::{Buffer, PushBuffer};

///Buffer and window for convolution. Buffer stores data for continuation. Generic T must be either f32 or f64.
pub struct Convolution<T>
{
    buffer : PushBuffer<T>,
    window : Buffer<T>
}
impl<T : std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + Copy + Default> Convolution<T>
{
    pub fn new(data : &mut [T]) -> Result<Self, LayoutError>
    {
        let mut window = Self
        {
            buffer : PushBuffer::<T>::new(data.len())?,
            window : Buffer::from_raw(data.as_mut_ptr(), data.len()),
        };
        window.buffer.set_index(window.buffer.len());
        Ok(window)
    }
    ///Convolve input data into window, then returns into output.
    pub fn run(& mut self, input : &Buffer<T>, output : &mut Buffer<T>)
    {
        let mut data = T::default();
        for index in 0..input.len()
        {
            self.buffer.push(input[index]);
            (0..self.window.len()).for_each(|index| data = data + self.buffer[index] * self.window[index]);
            output[index] = data;
        }
    }
}

///Set saturation character for one side. Generic T must be either f32 or f64.
pub struct Saturation<T>
{
    ths : T,
    lim : T,
    gap : T,
    rad_pow : T,
    org : T
}
impl Saturation<f32>
{
    ///New saturation data.
    pub fn new(ths : f32, lim : f32) -> Self
    {
        let gap = lim - ths;
        let side = ((gap * 2.0).powi(2) + gap.powi(2)).sqrt();
        let ang = 180.0 - 2.0 * (gap * 2.0 / side).asin();
        let rad = side / (2.0 * (ang/2.0).sin());
        let rad_pow = rad.powi(2);
        let org = lim - rad;

        Self { ths, lim, gap, rad_pow, org }
    }
    ///Process each data for non-linear behavior.
    pub fn run(input : & Buffer<f32>, output : & mut Buffer<f32>, upper : Saturation<f32>, lower : Saturation<f32>)
    {
        for index in 0..input.len()
        {
            if input[index] > upper.lim + upper.gap { output[index] = upper.lim; }
            else if input[index] > upper.ths { output[index] =  upper.org + (upper.rad_pow - (upper.lim - input[index]).powi(2)).sqrt(); }
            else if input[index] < lower.lim - lower.gap { output[index] = upper.lim; }
            else if input[index] < lower.ths { output[index] = lower.org - (lower.rad_pow - (lower.lim - input[index]).powi(2)).sqrt(); }
            else { output[index] = input[index]; }
        }
    }
}
impl Saturation<f64>
{
    ///New saturation data.
    pub fn new(ths : f64, lim : f64) -> Self
    {
        let gap = lim - ths;
        let side = ((gap * 2.0).powi(2) + gap.powi(2)).sqrt();
        let ang = 180.0 - 2.0 * (gap * 2.0 / side).asin();
        let rad = side / (2.0 * (ang/2.0).sin());
        let rad_pow = rad.powi(2);
        let org = lim - rad;

        Self { ths, lim, gap, rad_pow, org }
    }
    ///Process each data for non-linear behavior.
    pub fn run(input : & Buffer<f64>, output : & mut Buffer<f64>, upper : Saturation<f64>, lower : Saturation<f64>)
    {
        for index in 0..input.len()
        {
            if input[index] > upper.lim + upper.gap { output[index] = upper.lim; }
            else if input[index] > upper.ths { output[index] =  upper.org + (upper.rad_pow - (upper.lim - input[index]).powi(2)).sqrt(); }
            else if input[index] < lower.lim - lower.gap { output[index] = upper.lim; }
            else if input[index] < lower.ths { output[index] = lower.org - (lower.rad_pow - (lower.lim - input[index]).powi(2)).sqrt(); }
            else { output[index] = input[index]; }
        }
    }
}