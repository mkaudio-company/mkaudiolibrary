use no_denormals::*;

/// Asymmetric logarithmic saturation model for analog-style harmonic generation.
///
/// This waveshaper uses a logarithmic curve with independent parameters for
/// positive and negative signal excursions, enabling asymmetric distortion
/// characteristics similar to tube amplifiers and tape saturation.
///
/// # Parameters
///
/// - **Alpha (α)** - Drive/knee control: Higher values create sharper knee transitions
/// - **Beta (β)** - Compression/gain: Controls output amplitude and compression amount
/// - **Delta (δ)** - DC bias: Shifts the crossover point between positive/negative curves
/// - **Gamma (γ)** - Polarity: Boolean to flip output polarity
///
/// # Transfer Function
///
/// For input `x` and bias `δ`:
/// - When `x >= δ`: `output = β+ * log2(1 + α+ * (x - δ)) / log2(1 + α+)`
/// - When `x < δ`:  `output = -β- * log2(1 + α- * (δ - x)) / log2(1 + α-)`
///
/// # Alternatives
///
/// This model is memoryless and cheap, but purely a curve-fit. For a
/// physically-modeled alternative driven by an actual vacuum tube circuit
/// (Koren triode equations solved via Newton-Raphson over a WDF network),
/// see [`TubeSaturation`] (requires the `sim` feature).
pub struct Saturation {
    drive_alpha_plus: f32,
    drive_alpha_minus: f32,
    compression_beta_plus: f32,
    compression_beta_minus: f32,
    bias_delta: f32,
    flip_polarity: bool,
    norm_factor_plus: f32,
    norm_factor_minus: f32,
}
impl Saturation {
    /// Create a new saturation processor with asymmetric parameters.
    ///
    /// # Arguments
    /// * `alpha_plus` - Drive/knee for positive signal (higher = sharper knee)
    /// * `alpha_minus` - Drive/knee for negative signal
    /// * `beta_plus` - Output gain/compression for positive signal
    /// * `beta_minus` - Output gain/compression for negative signal
    /// * `delta_bias` - DC bias offset (shifts crossover point)
    /// * `flip` - Invert output polarity if true
    pub fn new(
        alpha_plus: f32,
        alpha_minus: f32,
        beta_plus: f32,
        beta_minus: f32,
        delta_bias: f32,
        flip: bool,
    ) -> Self {
        let drive_alpha_plus = alpha_plus.max(1e-4);
        let drive_alpha_minus = alpha_minus.max(1e-4);

        let norm_factor_plus = 1.0 / (1.0 + drive_alpha_plus).log2();
        let norm_factor_minus = 1.0 / (1.0 + drive_alpha_minus).log2();

        Self {
            drive_alpha_plus,
            drive_alpha_minus,
            compression_beta_plus: beta_plus,
            compression_beta_minus: beta_minus,
            bias_delta: delta_bias,
            flip_polarity: flip,
            norm_factor_plus,
            norm_factor_minus,
        }
    }

    /// Process a single sample through the saturation curve.
    #[inline]
    pub fn process(&self, input_sample: f32) -> f32 {
        let output_value = if input_sample >= self.bias_delta {
            // Processing positive side (x >= δ)
            let relative_input = input_sample - self.bias_delta;
            let log_out = (1.0 + self.drive_alpha_plus * relative_input).log2();
            self.compression_beta_plus * (log_out * self.norm_factor_plus)
        } else {
            // Processing negative side (x < δ)
            let relative_input = self.bias_delta - input_sample;
            let log_out = (1.0 + self.drive_alpha_minus * relative_input).log2();
            -self.compression_beta_minus * (log_out * self.norm_factor_minus)
        };

        if self.flip_polarity {
            -output_value
        } else {
            output_value
        }
    }

    /// Process a buffer of samples.
    pub fn run(&self, input: &[f32], output: &mut [f32]) {
        unsafe {
            no_denormals(|| {
                for index in 0..input.len().min(output.len()) {
                    output[index] = self.process(input[index]);
                }
            });
        }
    }
}

/// Physically-modeled tube saturation driven by [`crate::sim`]'s circuit
/// simulation (requires the `sim` feature).
///
/// Unlike [`Saturation`]'s memoryless log-curve waveshaper, `TubeSaturation`
/// runs an actual 12AX7 triode stage: a Wave Digital Filter plate/cathode
/// network combined with a Newton-Raphson solve of the Koren triode
/// equations each sample. This produces frequency-dependent, level-dependent
/// harmonic behavior (including the bias-point-driven asymmetry real tube
/// stages exhibit) that a static curve cannot reproduce, at the cost of
/// materially more CPU per sample.
///
/// `sim`'s components and the rest of `dsp` both operate in `f32`, so this
/// wrapper is a direct pass-through to [`crate::sim::components::tubes::TriodeStage`]
/// with no per-sample conversion.
///
/// # Example
///
/// ```ignore
/// use mkaudiolibrary::dsp::TubeSaturation;
///
/// let mut tube = TubeSaturation::new(44100.0);
/// let wet = tube.process(0.5);
/// ```
#[cfg(feature = "sim")]
pub struct TubeSaturation {
    stage: crate::sim::components::tubes::TriodeStage,
}

#[cfg(feature = "sim")]
impl TubeSaturation {
    /// Create a tube saturation stage using default 12AX7 parameters.
    pub fn new(sample_rate: f32) -> Self {
        Self::with_params(
            sample_rate,
            crate::sim::components::tubes::PARAMS_12AX7.clone(),
        )
    }

    /// Create a tube saturation stage with custom triode parameters
    /// (see [`crate::sim::components::tubes::TriodeParams`]).
    pub fn with_params(
        sample_rate: f32,
        params: crate::sim::components::tubes::TriodeParams,
    ) -> Self {
        use crate::sim::components::CircuitComponent;
        let mut stage = crate::sim::components::tubes::TriodeStage::new(params);
        stage.prepare(sample_rate);
        Self { stage }
    }

    /// Re-initialize internal state for a new sample rate.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        use crate::sim::components::CircuitComponent;
        self.stage.prepare(sample_rate);
    }

    /// Process a single sample through the triode stage.
    #[inline]
    pub fn process(&mut self, input_sample: f32) -> f32 {
        use crate::sim::components::CircuitComponent;
        let input = [input_sample];
        let mut output = [0.0f32];
        self.stage.process_block(&input, &mut output);
        output[0]
    }

    /// Process a buffer of samples.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        use crate::sim::components::CircuitComponent;
        let len = input.len().min(output.len());
        unsafe {
            no_denormals(|| {
                self.stage.process_block(&input[..len], &mut output[..len]);
            });
        }
    }
}
