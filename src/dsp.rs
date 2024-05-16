use std::alloc::LayoutError;
use no_denormals::*;

use crate::buffer::*;

/// Convert ratio to dB.
#[inline]
pub fn ratio_to_db(ratio : f64) -> f64 { ratio.log10() / 20.0 }

/// Convert dB to ratio.
#[inline]
pub fn db_to_ratio(db : f64) -> f64 { 10.0f64.powf(db / 20.0) }

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
        no_denormals(||
        {
            for index in 0..input.len()
            {
                self.buffer.push(input[index]);
                (0..self.window.len()).for_each(|index| data = data + self.buffer[index] * self.window[index]);
                output[index] = data;
            }
        });
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
    pub fn run(input : & Buffer<f32>, output : & mut Buffer<f32>, upper : Self, lower : Self)
    {
        no_denormals(||
        {
            for index in 0..input.len()
            {
                if input[index] > upper.lim + upper.gap { output[index] = upper.lim; }
                else if input[index] > upper.ths { output[index] =  upper.org + (upper.rad_pow - (upper.lim - input[index]).powi(2)).sqrt(); }
                else if input[index] < lower.lim - lower.gap { output[index] = upper.lim; }
                else if input[index] < lower.ths { output[index] = lower.org - (lower.rad_pow - (lower.lim - input[index]).powi(2)).sqrt(); }
                else { output[index] = input[index]; }
            }
        });
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
    pub fn run(input : & Buffer<f64>, output : & mut Buffer<f64>, upper : Self, lower : Self)
    {
        no_denormals(||
        {
            for index in 0..input.len()
            {
                if input[index] > upper.lim + upper.gap { output[index] = upper.lim; }
                else if input[index] > upper.ths { output[index] =  upper.org + (upper.rad_pow - (upper.lim - input[index]).powi(2)).sqrt(); }
                else if input[index] < lower.lim - lower.gap { output[index] = upper.lim; }
                else if input[index] < lower.ths { output[index] = lower.org - (lower.rad_pow - (lower.lim - input[index]).powi(2)).sqrt(); }
                else { output[index] = input[index]; }
            }
        });
    }
}

#[derive(Default)]
pub struct Compression
{
    pub threshold : f64,// Threshold in dB.
    pub ratio : f64,    // Ratio of the compression.
    pub attack : f64,   // Attack in ms.
    pub release : f64,  // Release in ms.
    pub makeup : f64,   // Makeup Gain in dB.
    buffer : f64
}
impl Compression
{
    pub fn run(&mut self, input : &Buffer<f64>, output : &mut Buffer<f64>, buffer_size : usize, sample_rate : f64)
    {
        if input.len() != buffer_size || output.len() != buffer_size { return }
        let real_gain = db_to_ratio(self.makeup);
        let real_threshold = db_to_ratio(self.threshold);

        no_denormals(||
        {
            for index in 0..buffer_size
            {
                if input[index] > real_threshold { self.buffer -= ratio_to_db((input[index] - real_threshold) / (self.ratio * (sample_rate / (self.attack * 1000.0)))); }
                output[index] = input[index] * real_gain * db_to_ratio(self.buffer);
                if self.buffer < 0.0 { self.buffer += self.buffer * self.release * 1000.0 / sample_rate; }
            }
        });
    }
}

#[derive(Default)]
pub struct Limit
{
    pub gain : f64,     // Gain in dB.
    pub ceiling : f64,  // Ceiling in dB.
    pub release : f64,  // Release time in ms.
    buffer : f64
}
impl Limit
{
    pub fn run(&mut self, input : &Buffer<f64>, output : &mut Buffer<f64>, buffer_size : usize, sample_rate : f64)
    {
        if input.len() != buffer_size || output.len() != buffer_size { return }
        let real_gain = db_to_ratio(self.gain);
        let real_ceiling = db_to_ratio(self.ceiling);

        no_denormals(||
        {
            for index in 0..buffer_size
            {
                if input[index] * real_gain > real_ceiling { self.buffer -= ratio_to_db(input[index] * real_gain - real_ceiling); }
                output[index] = input[index] * real_gain * db_to_ratio(self.buffer);
                if self.buffer < 0.0 { self.buffer += self.buffer * self.release * 1000.0 / sample_rate; }
            }
        });
    }
}

pub struct Delay
{
    time : f64,                 // Delay time in ms.
    pub feedback : f64,         // Feedback in percent.
    pub mix : f64,              // Mix in percent.
    buffer : CircularBuffer<f64>// Buffer for delay
}
impl Delay
{
    pub fn new(time : f64, sample_rate : f64) -> Self
    {
        Self { time, feedback : 50.0, mix : 50.0, buffer : CircularBuffer::new((time / (1000.0 * sample_rate)) as usize).unwrap() }
    }
    pub fn get_time(&self) -> f64 { self.time }
    pub fn set_time(&mut self, time : f64, sample_rate : f64)
    {
        self.time = time;
        self.buffer.resize((time * 1000.0 / sample_rate) as usize).unwrap();
    }
    pub fn run(&mut self, input : &Buffer<f64>, output : &mut Buffer<f64>, buffer_size : usize)
    {
        if input.len() != buffer_size || output.len() != buffer_size { return }
        no_denormals(||
        {
            for index in 0..buffer_size
            {
                let data = self.buffer.next();
                output[index] = input[index] + data * self.mix / 100.0;
                self.buffer.push(data * self.feedback / 100.0 + input[index]);
            }
        });
    }
}