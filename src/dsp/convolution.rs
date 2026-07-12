use no_denormals::*;

/// FIR convolution processor with impulse response.
///
/// Performs discrete convolution of input signal with a kernel (impulse response).
/// Keeps a sliding window of the most recent `kernel.len()` input samples so
/// convolution is seamless across `process`/`run` calls (no reset at buffer
/// boundaries).
///
/// This processor owns its history buffer outright (no internal locking) -
/// use `&mut Convolution` per audio-processing thread rather than sharing
/// one instance across threads.
pub struct Convolution {
    // Sliding window of the most recent `kernel.len()` input samples,
    // oldest first. Shifted left by one and appended to on each `push`.
    history: Vec<f32>,
    kernel: Box<[f32]>,
}
impl Convolution {
    /// Create a new convolution processor with the given impulse response.
    pub fn new(kernel: &[f32]) -> Self {
        Self {
            history: vec![0.0; kernel.len()],
            kernel: kernel.to_vec().into_boxed_slice(),
        }
    }

    /// Get the length of the impulse response.
    pub fn kernel_len(&self) -> usize {
        self.kernel.len()
    }

    #[inline]
    fn push(&mut self, value: f32) {
        let len = self.history.len();
        if len == 0 {
            return;
        }
        self.history.copy_within(1..len, 0);
        self.history[len - 1] = value;
    }

    /// Process a single sample through convolution.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        self.push(input);
        crate::simd::dot(&self.history, &self.kernel)
    }

    /// Convolve input buffer with impulse response, writing to output.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            no_denormals(|| {
                for index in 0..input.len().min(output.len()) {
                    self.push(input[index]);
                    output[index] = crate::simd::dot(&self.history, &self.kernel);
                }
            });
        }
    }
}
