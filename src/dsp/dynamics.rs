use no_denormals::*;

use crate::dsp::{db_to_ratio, ratio_to_db};

/// Dynamic range compressor with soft knee and envelope detection.
///
/// Reduces the dynamic range of audio by attenuating signals above a threshold.
/// Features smooth attack/release envelope following and optional soft knee
/// for more transparent compression.
///
/// # Parameters
/// - `threshold` - Level (dB) above which compression begins
/// - `ratio` - Compression ratio (e.g., 4.0 means 4:1 compression)
/// - `attack` - Time (ms) to reach full compression
/// - `release` - Time (ms) to return to unity gain
/// - `makeup` - Output gain (dB) to compensate for level reduction
/// - `knee` - Soft knee width (dB), 0 = hard knee
///
/// # Envelope Detection
/// Uses a one-pole lowpass filter with separate attack/release coefficients
/// for smooth gain reduction that follows the input signal envelope.
pub struct Compression {
    /// Threshold in dB (signals above this are compressed).
    pub threshold: f32,
    /// Compression ratio (e.g., 4.0 for 4:1).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack: f32,
    /// Release time in milliseconds.
    pub release: f32,
    /// Makeup gain in dB.
    pub makeup: f32,
    /// Soft knee width in dB (0 = hard knee).
    pub knee: f32,
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
    // Pre-allocated gain envelope scratch space, reused across calls to
    // `run()` so the audio thread never allocates once warmed up.
    gain_scratch: Vec<f32>,
}
impl Compression {
    /// Create a new compressor with default parameters.
    ///
    /// Defaults: -20dB threshold, 4:1 ratio, 10ms attack, 100ms release.
    pub fn new(sample_rate: f32) -> Self {
        let mut comp = Self {
            threshold: -20.0,
            ratio: 4.0,
            attack: 10.0,
            release: 100.0,
            makeup: 0.0,
            knee: 0.0,
            envelope: 0.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            gain_scratch: Vec::new(),
        };
        comp.update_coefficients(sample_rate);
        comp
    }

    /// Pre-allocate the internal gain envelope scratch buffer for a given
    /// block size, avoiding any allocation the first time `run()` is called
    /// with that size. Call this from `prepare_to_play` when wrapping this
    /// in a `Processor`.
    pub fn set_block_size(&mut self, block_size: usize) {
        if self.gain_scratch.len() < block_size {
            self.gain_scratch.resize(block_size, 0.0);
        }
    }

    /// Update attack/release coefficients when parameters or sample rate change.
    pub fn update_coefficients(&mut self, sample_rate: f32) {
        self.attack_coeff = (-1.0 / (self.attack * 0.001 * sample_rate)).exp();
        self.release_coeff = (-1.0 / (self.release * 0.001 * sample_rate)).exp();
    }

    /// Compute gain reduction in dB for a given input level in dB.
    #[inline]
    fn compute_gain(&self, input_db: f32) -> f32 {
        if self.knee > 0.0 {
            let half_knee = self.knee * 0.5;
            let lower = self.threshold - half_knee;
            let upper = self.threshold + half_knee;

            if input_db <= lower {
                0.0
            } else if input_db >= upper {
                (self.threshold + (input_db - self.threshold) / self.ratio) - input_db
            } else {
                // Soft knee region
                let x = input_db - lower;
                let slope = 1.0 / self.ratio - 1.0;
                slope * x * x / (2.0 * self.knee)
            }
        } else {
            // Hard knee
            if input_db <= self.threshold {
                0.0
            } else {
                (self.threshold + (input_db - self.threshold) / self.ratio) - input_db
            }
        }
    }

    /// Process a buffer of samples.
    ///
    /// Runs the (inherently sequential) attack/release envelope follower
    /// first into a scratch gain buffer, then applies the gain to the whole
    /// block in one vectorized pass (SIMD-accelerated with the `simd` feature).
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        let makeup_linear = db_to_ratio(self.makeup);

        self.set_block_size(len);

        unsafe {
            no_denormals(|| {
                for (index, &x) in input[..len].iter().enumerate() {
                    let input_abs = x.abs();
                    let input_db = if input_abs > 1e-10 {
                        20.0 * input_abs.log10()
                    } else {
                        -200.0
                    };

                    // Compute target gain reduction
                    let target_gr = self.compute_gain(input_db);

                    // Envelope follower (attack/release)
                    let coeff = if target_gr < self.envelope {
                        self.attack_coeff
                    } else {
                        self.release_coeff
                    };
                    self.envelope = target_gr + coeff * (self.envelope - target_gr);

                    self.gain_scratch[index] = db_to_ratio(self.envelope) * makeup_linear;
                }

                crate::simd::mul_elementwise(
                    &mut output[..len],
                    &input[..len],
                    &self.gain_scratch[..len],
                );
            });
        }
    }
}

/// Brickwall limiter with instant attack and smooth release.
///
/// Prevents output from exceeding a ceiling level by applying instant
/// gain reduction when needed, with smooth release back to unity gain.
/// No lookahead is used, making this suitable for real-time applications.
///
/// # Parameters
/// - `gain` - Input gain (dB) applied before limiting
/// - `ceiling` - Maximum output level (dB)
/// - `release` - Time (ms) to return to unity gain after limiting
///
/// # Behavior
/// - **Instant attack**: Gain reduction is applied immediately when needed
/// - **Smooth release**: One-pole filter smoothly returns to unity gain
pub struct Limit {
    /// Input gain in dB (applied before limiting).
    pub gain: f32,
    /// Output ceiling in dB (maximum output level).
    pub ceiling: f32,
    /// Release time in milliseconds.
    pub release: f32,
    envelope: f32,
    release_coeff: f32,
    // Pre-allocated gain envelope scratch space, reused across calls to
    // `run()` so the audio thread never allocates once warmed up.
    gain_scratch: Vec<f32>,
}
impl Limit {
    /// Create a new limiter with default parameters.
    ///
    /// Defaults: 0dB gain, 0dB ceiling, 100ms release.
    pub fn new(sample_rate: f32) -> Self {
        let mut lim = Self {
            gain: 0.0,
            ceiling: 0.0,
            release: 100.0,
            envelope: 1.0,
            release_coeff: 0.0,
            gain_scratch: Vec::new(),
        };
        lim.update_coefficients(sample_rate);
        lim
    }

    /// Update release coefficient when parameters or sample rate change.
    pub fn update_coefficients(&mut self, sample_rate: f32) {
        self.release_coeff = (-1.0 / (self.release * 0.001 * sample_rate)).exp();
    }

    /// Pre-allocate the internal gain envelope scratch buffer for a given
    /// block size, avoiding any allocation the first time `run()` is called
    /// with that size. Call this from `prepare_to_play` when wrapping this
    /// in a `Processor`.
    pub fn set_block_size(&mut self, block_size: usize) {
        if self.gain_scratch.len() < block_size {
            self.gain_scratch.resize(block_size, 0.0);
        }
    }

    /// Process a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let gain_linear = db_to_ratio(self.gain);
        let ceiling_linear = db_to_ratio(self.ceiling);

        let amplified = input * gain_linear;
        let abs_sample = amplified.abs();

        // Compute required gain reduction
        let target = if abs_sample > ceiling_linear {
            ceiling_linear / abs_sample
        } else {
            1.0
        };

        // Instant attack, smooth release
        if target < self.envelope {
            self.envelope = target;
        } else {
            self.envelope = target + self.release_coeff * (self.envelope - target);
        }

        amplified * self.envelope
    }

    /// Process a buffer of samples.
    ///
    /// Runs the (inherently sequential) instant-attack/smooth-release
    /// envelope follower first into a scratch gain buffer, then applies the
    /// gain to the whole block in one vectorized pass (SIMD-accelerated
    /// with the `simd` feature).
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        let gain_linear = db_to_ratio(self.gain);
        let ceiling_linear = db_to_ratio(self.ceiling);

        self.set_block_size(len);

        unsafe {
            no_denormals(|| {
                for (index, &x) in input[..len].iter().enumerate() {
                    let amplified = x * gain_linear;
                    let abs_sample = amplified.abs();

                    let target = if abs_sample > ceiling_linear {
                        ceiling_linear / abs_sample
                    } else {
                        1.0
                    };

                    if target < self.envelope {
                        self.envelope = target;
                    } else {
                        self.envelope = target + self.release_coeff * (self.envelope - target);
                    }

                    self.gain_scratch[index] = gain_linear * self.envelope;
                }

                crate::simd::mul_elementwise(
                    &mut output[..len],
                    &input[..len],
                    &self.gain_scratch[..len],
                );
            });
        }
    }
}

/// Downward-expanding noise gate with hysteresis and hold time.
///
/// Attenuates the signal toward `range` dB whenever its level drops below
/// `threshold`, so it silences noise between notes without chopping off
/// sustain. A `hold` period keeps the gate open for a fixed time after the
/// level first drops below threshold, before the release ramp begins, which
/// avoids audible "chattering" on signals that hover near the threshold.
///
/// # Parameters
/// - `threshold` - Level (dB) below which the gate begins to close
/// - `attack` - Time (ms) to fully open once the level rises above threshold
/// - `hold` - Time (ms) the gate stays fully open after level drops below threshold
/// - `release` - Time (ms) to close from fully open to `range`
/// - `range` - Attenuation (dB, negative) applied when fully closed
pub struct Gate {
    /// Threshold in dB (signals below this cause the gate to close).
    pub threshold: f32,
    /// Attack time in milliseconds (opening).
    pub attack: f32,
    /// Hold time in milliseconds (fully open before release begins).
    pub hold: f32,
    /// Release time in milliseconds (closing).
    pub release: f32,
    /// Attenuation in dB applied when fully closed (e.g. -60.0).
    pub range: f32,
    envelope: f32,
    hold_remaining_samples: usize,
    attack_coeff: f32,
    release_coeff: f32,
    sample_rate: f32,
    // Pre-allocated gain envelope scratch space, reused across calls to
    // `run()` so the audio thread never allocates once warmed up.
    gain_scratch: Vec<f32>,
}
impl Gate {
    /// Create a new noise gate with default parameters.
    ///
    /// Defaults: -40dB threshold, 1ms attack, 50ms hold, 100ms release, -60dB range.
    pub fn new(sample_rate: f32) -> Self {
        let mut gate = Self {
            threshold: -40.0,
            attack: 1.0,
            hold: 50.0,
            release: 100.0,
            range: -60.0,
            envelope: 0.0,
            hold_remaining_samples: 0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            sample_rate,
            gain_scratch: Vec::new(),
        };
        gate.update_coefficients(sample_rate);
        gate
    }

    /// Update attack/release coefficients when parameters or sample rate change.
    pub fn update_coefficients(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.attack_coeff = (-1.0 / (self.attack.max(0.001) * 0.001 * sample_rate)).exp();
        self.release_coeff = (-1.0 / (self.release.max(0.001) * 0.001 * sample_rate)).exp();
    }

    /// Pre-allocate the internal gain envelope scratch buffer for a given
    /// block size, avoiding any allocation the first time `run()` is called
    /// with that size.
    pub fn set_block_size(&mut self, block_size: usize) {
        if self.gain_scratch.len() < block_size {
            self.gain_scratch.resize(block_size, 0.0);
        }
    }

    /// Process a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let input_abs = input.abs();
        let input_db = if input_abs > 1e-10 {
            ratio_to_db(input_abs)
        } else {
            -200.0
        };

        let target_db = if input_db >= self.threshold {
            self.hold_remaining_samples = (self.hold * 0.001 * self.sample_rate) as usize;
            0.0
        } else if self.hold_remaining_samples > 0 {
            self.hold_remaining_samples -= 1;
            0.0
        } else {
            self.range
        };

        let target_linear = db_to_ratio(target_db);
        let coeff = if target_linear > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = target_linear + coeff * (self.envelope - target_linear);

        input * self.envelope
    }

    /// Process a buffer of samples.
    ///
    /// Runs the (inherently sequential) attack/hold/release envelope
    /// follower first into a scratch gain buffer, then applies the gain to
    /// the whole block in one vectorized pass (SIMD-accelerated with the
    /// `simd` feature).
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());

        self.set_block_size(len);

        unsafe {
            no_denormals(|| {
                for (index, &x) in input[..len].iter().enumerate() {
                    let input_abs = x.abs();
                    let input_db = if input_abs > 1e-10 {
                        ratio_to_db(input_abs)
                    } else {
                        -200.0
                    };

                    let target_db = if input_db >= self.threshold {
                        self.hold_remaining_samples =
                            (self.hold * 0.001 * self.sample_rate) as usize;
                        0.0
                    } else if self.hold_remaining_samples > 0 {
                        self.hold_remaining_samples -= 1;
                        0.0
                    } else {
                        self.range
                    };

                    let target_linear = db_to_ratio(target_db);
                    let coeff = if target_linear > self.envelope {
                        self.attack_coeff
                    } else {
                        self.release_coeff
                    };
                    self.envelope = target_linear + coeff * (self.envelope - target_linear);

                    self.gain_scratch[index] = self.envelope;
                }

                crate::simd::mul_elementwise(
                    &mut output[..len],
                    &input[..len],
                    &self.gain_scratch[..len],
                );
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_opens_above_threshold() {
        let mut gate = Gate::new(44100.0);
        gate.threshold = -20.0;
        gate.attack = 0.1;

        // Loud signal should converge to (near) unity gain.
        let mut last = 0.0;
        for _ in 0..4410 {
            last = gate.process(0.5);
        }
        assert!(
            (last.abs() / 0.5) > 0.9,
            "gate should be mostly open for a loud signal, got gain {}",
            last / 0.5
        );
    }

    #[test]
    fn test_gate_closes_below_threshold_after_hold() {
        let mut gate = Gate::new(44100.0);
        gate.threshold = -20.0;
        gate.hold = 1.0;
        gate.release = 5.0;
        gate.range = -60.0;

        // Quiet signal, well past hold+release, should converge toward `range`.
        let mut last = 0.0;
        for _ in 0..44100 {
            last = gate.process(0.0001);
        }
        let gain_db = ratio_to_db((last / 0.0001).abs());
        assert!(
            gain_db < -50.0,
            "gate should be mostly closed for a quiet signal, got {gain_db} dB"
        );
    }
}
