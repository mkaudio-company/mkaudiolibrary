use no_denormals::*;

/// Feedback delay line with wet/dry mix.
///
/// A simple delay effect using a power-of-two ring buffer for the delay
/// line (so the read/write cursors wrap with a cheap bitmask rather than a
/// modulo). Supports feedback for echo/repeat effects and wet/dry mixing.
///
/// # Parameters
/// - `time` - Delay time in milliseconds
/// - `feedback` - Amount of delayed signal fed back (0.0 to 1.0)
/// - `mix` - Wet/dry balance (0.0 = dry only, 1.0 = wet only)
///
/// This processor owns its delay line and scratch buffer outright (no
/// internal locking) - use `&mut Delay` per audio-processing thread rather
/// than sharing one instance across threads.
///
/// # Note
/// The actual delay is `time` rounded up to the next power of two in
/// samples (matching the ring buffer's capacity), so it may be slightly
/// longer than requested. Feedback values >= 1.0 will cause infinite
/// buildup; use values < 1.0 for stable operation.
pub struct Delay {
    time: f32,
    sample_rate: f32,
    /// Feedback amount (0.0 to 1.0, values >= 1.0 cause buildup).
    pub feedback: f32,
    /// Wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    pub mix: f32,
    ring: Vec<f32>,
    read: usize,
    write: usize,
    mask: usize,
    // Pre-allocated wet-signal scratch space, reused across calls to
    // `run()` so the audio thread never allocates once warmed up.
    scratch: Vec<f32>,
}
impl Delay {
    /// Create a new delay with the specified time and sample rate.
    ///
    /// # Arguments
    /// * `time` - Delay time in milliseconds
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(time: f32, sample_rate: f32) -> Self {
        let capacity = Self::capacity_for(time, sample_rate);
        Self {
            time,
            sample_rate,
            feedback: 0.5,
            mix: 0.5,
            ring: vec![0.0; capacity],
            read: 0,
            write: 0,
            mask: capacity - 1,
            scratch: Vec::new(),
        }
    }

    fn capacity_for(time: f32, sample_rate: f32) -> usize {
        ((time * 0.001 * sample_rate) as usize)
            .max(1)
            .next_power_of_two()
    }

    fn resize(&mut self, capacity: usize) {
        self.ring = vec![0.0; capacity];
        self.read = 0;
        self.write = 0;
        self.mask = capacity - 1;
    }

    /// Get the current delay time in ms.
    pub fn get_time(&self) -> f32 {
        self.time
    }

    /// Set the delay time in ms.
    pub fn set_time(&mut self, time: f32) {
        self.time = time;
        self.resize(Self::capacity_for(time, self.sample_rate));
    }

    /// Set the sample rate and update buffer size accordingly.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.resize(Self::capacity_for(self.time, sample_rate));
    }

    #[inline]
    fn step(&mut self, input: f32) -> f32 {
        let delayed = self.ring[self.read];
        self.read = (self.read + 1) & self.mask;
        self.ring[self.write] = input + delayed * self.feedback;
        self.write = (self.write + 1) & self.mask;
        delayed
    }

    /// Process a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.step(input);
        input * (1.0 - self.mix) + delayed * self.mix
    }

    /// Process a buffer of samples.
    ///
    /// Advances the (inherently sequential) delay line first into a scratch
    /// wet-signal buffer, then computes the wet/dry mix for the whole block
    /// in one vectorized pass (SIMD-accelerated with the `simd` feature).
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        if self.scratch.len() < len {
            self.scratch.resize(len, 0.0);
        }

        unsafe {
            no_denormals(|| {
                // Not a needless_range_loop: `self.step()` needs `&mut self`
                // as a whole, so `self.scratch` can't be iterated mutably
                // (via iter_mut/zip) at the same time.
                #[allow(clippy::needless_range_loop)]
                for index in 0..len {
                    self.scratch[index] = self.step(input[index]);
                }

                crate::simd::mix_scalar(
                    &mut output[..len],
                    &input[..len],
                    &self.scratch[..len],
                    self.mix,
                );
            });
        }
    }
}
