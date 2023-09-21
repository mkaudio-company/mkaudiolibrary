use std::alloc::LayoutError;

use crate::buffer::{Buffer, PushBuffer};

///Buffer and window for convolution. Buffer stores data for continuation. Generic T must be either f32 or f64.
pub struct Convolution<T>
{
    buffer : PushBuffer<T>,
    window : Box<[T]>
}
impl Convolution<f32>
{
    #[inline]
    pub fn new(data : & [f32]) -> Result<Self, LayoutError>
    {
        let mut window = Self
        {
            buffer : PushBuffer::<f32>::new(data.len())?,
            window : Box::<[f32]>::from(data),
        };
        window.buffer.set_index(window.buffer.len());
        Ok(window)
    }
    ///Convolve input data into window, then returns into output.
    #[inline]
    pub fn run(& mut self, input : & Buffer<f32>, output : & mut Buffer<f32>)
    {
        let mut data = 0.0;
        for x in 0..input.len()
        {
            self.buffer.push(input[x]);
            (0..self.window.len()).for_each(|x| data += self.buffer[x] * self.window[x]);
            output[x] = data;
        }
    }
}
impl Convolution<f64>
{
    #[inline]
    pub fn new(data : & [f64]) -> Result<Self, LayoutError>
    {
        let mut window = Self
        {
            buffer : PushBuffer::<f64>::new(data.len())?,
            window : Box::<[f64]>::from(data),
        };
        window.buffer.set_index(window.buffer.len());
        Ok(window)
    }
    ///Convolve input data into window, then returns into output.
    #[inline]
    pub fn run(& mut self, input : & Buffer<f64>, output : & mut Buffer<f64>)
    {
        let mut data = 0.0;
        for x in 0..input.len()
        {
            self.buffer.push(input[x]);
            (0..self.window.len()).for_each(|x| data += self.buffer[x] * self.window[x]);
            output[x] = data;
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
    #[inline]
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
    #[inline]
    pub fn run(input : & Buffer<f32>, output : & mut Buffer<f32>, upper : Saturation<f32>, lower : Saturation<f32>)
    {
        for i in 0..input.len()
        {
            if input[i] > upper.lim + upper.gap { output[i] = upper.lim; }
            else if input[i] > upper.ths { output[i] =  upper.org + (upper.rad_pow - (upper.lim - input[i]).powi(2)).sqrt(); }
            else if input[i] < lower.lim - lower.gap { output[i] = upper.lim; }
            else if input[i] < lower.ths { output[i] = lower.org - (lower.rad_pow - (lower.lim - input[i]).powi(2)).sqrt(); }
            else { output[i] = input[i]; }
        }
    }
}
impl Saturation<f64>
{
    ///New saturation data.
    #[inline]
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
    #[inline]
    pub fn run(input : & Buffer<f64>, output : & mut Buffer<f64>, upper : Saturation<f64>, lower : Saturation<f64>)
    {
        for i in 0..input.len()
        {
            if input[i] > upper.lim + upper.gap { output[i] = upper.lim; }
            else if input[i] > upper.ths { output[i] =  upper.org + (upper.rad_pow - (upper.lim - input[i]).powi(2)).sqrt(); }
            else if input[i] < lower.lim - lower.gap { output[i] = upper.lim; }
            else if input[i] < lower.ths { output[i] = lower.org - (lower.rad_pow - (lower.lim - input[i]).powi(2)).sqrt(); }
            else { output[i] = input[i]; }
        }
    }
}