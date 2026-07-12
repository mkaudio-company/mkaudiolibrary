[![](https://img.shields.io/crates/v/mkaudiolibrary.svg)](https://crates.io/crates/mkaudiolibrary)
[![](https://img.shields.io/crates/l/mkaudiolibrary.svg)](https://crates.io/crates/mkaudiolibrary)
[![](https://docs.rs/mkaudiolibrary/badge.svg)](https://docs.rs/mkaudiolibrary/)

# mkaudiolibrary

A Rust library for real-time audio signal processing, featuring analog modeling through numeric functions and circuit simulation via Modified Nodal Analysis (MNA).

Every audio-sample-carrying API in this crate operates on plain `f32` - matching VST3/AU/MKAP plugin hosting's native sample format - passed as `&[f32]`/`&mut [f32]` slices or the unlocked `Buffer<f32>` wrapper. None of this crate's own processors share buffers across threads internally, so nothing pays for locking with no reader on the other end; wrap a buffer yourself (`Arc<Mutex<_>>`, a lock-free ring buffer, etc.) if you need to hand it to another thread.

## Features

- **Analog modeling** - asymmetric log-curve saturation, and (via the `sim` feature) physically-modeled vacuum tube saturation
- **Circuit simulation** - real-time MNA solver for reactive circuits (`dsp::Circuit`), plus a full tube/diode/transistor + Wave Digital Filter circuit modeling toolkit (`sim` feature, merged in from [libmksim](https://github.com/mkaudio-company/libmksim))
- **DSP primitives** - convolution, IIR (biquad/Butterworth) and FIR (windowed-sinc) filtering, compression/limiting/gating, delay, integer oversampling, and FFT-based sample-rate conversion, with pre-allocated scratch buffers so steady-state processing never allocates
- **SIMD acceleration** (optional `simd` feature) for hot per-sample loops: AVX2+FMA/SSE2 on `x86_64`, NEON on `aarch64`, scalar fallback otherwise
- **Time-frequency analysis** (`tf` module) - DFT, FFT (radix-2 + Bluestein for arbitrary lengths), DCT, STFT/multi-resolution STFT, CWT, CQT, and mel spectrograms
- **Audio file I/O** for WAV, BWF, and AIFF formats with Buffer integration
- **Plugin hosting** (`host` module) for MKAP (native), VST3, and AUv2 (macOS) - load and run third-party plugins through one `HostedPlugin` trait
- **MKAP plugin system** for building your own modular processing chains
- **Real-time streaming** via an RTAudio-style API (optional `realtime` feature) with real hardware-clocked backends: CoreAudio (macOS), WASAPI (Windows), ALSA (Linux)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mkaudiolibrary = "2.0.0"
```

For real-time audio streaming (real CoreAudio/WASAPI/ALSA backends), enable the `realtime` feature:

```toml
[dependencies]
mkaudiolibrary = { version = "2.0.0", features = ["realtime"] }
```

For SIMD-accelerated DSP/TF hot paths:

```toml
[dependencies]
mkaudiolibrary = { version = "2.0.0", features = ["simd"] }
```

For physically-modeled analog circuit simulation (tubes, diodes, transistors, WDF networks):

```toml
[dependencies]
mkaudiolibrary = { version = "2.0.0", features = ["sim"] }
# or with SIMD backends for sim's internal math: "sim-avx2" / "sim-avx512" / "sim-neon"
```

For hosting third-party VST3 plugins:

```toml
[dependencies]
mkaudiolibrary = { version = "2.0.0", features = ["vst3"] }
```

For hosting Audio Units (macOS only):

```toml
[dependencies]
mkaudiolibrary = { version = "2.0.0", features = ["au"] }
```

For MIDI support with mkmidilibrary integration:

```toml
[dependencies]
mkaudiolibrary = { version = "2.0.0", features = ["midi"] }
```

For plugin GUI support with [mkapk](https://github.com/mkaudio-company/mkapk) integration (`mkapk-core` + `mkapk-host`):

```toml
[dependencies]
mkaudiolibrary = { version = "2.0.0", features = ["gui"] }
```

## Quick Start

```rust
use mkaudiolibrary::audiofile::{AudioFile, FileFormat};
use mkaudiolibrary::dsp::Compression;
use mkaudiolibrary::buffer::Buffer;

// Load an audio file
let mut audio = AudioFile::default();
audio.load("input.wav");

println!("Loaded: {} channels, {} samples, {}Hz",
    audio.num_channel(),
    audio.num_sample(),
    audio.sample_rate()
);

// Convert to buffers for processing
let mut buffers = audio.to_buffers();

// Apply compression to each channel
let mut comp = Compression::new(audio.sample_rate());
comp.threshold = -12.0;
comp.ratio = 4.0;

for buffer in &mut buffers {
    let mut output = Buffer::new(buffer.len());
    comp.run(buffer, &mut output);
    // ... use processed output
}

// Save result
audio.save("output.wav", FileFormat::Wav);
```

## Modules

### buffer

Plain (unlocked) audio sample containers, single-owner - use `&mut` per audio-processing thread rather than sharing one instance across threads:

| Type | Description | Use Case |
|------|-------------|----------|
| `Buffer<T>` | Resizable, owned block of samples (`Box<[T]>` wrapper) | Owning your own sample storage (`AudioIO`/`MidiIO` themselves just borrow `&[T]`/`&mut [T]`, not `Buffer`) |
| `PushBuffer<T>` | FIFO buffer that shifts samples on push | FIR filters, convolution |
| `CircularBuffer<T>` | Ring buffer with power-of-2 sizing | Delay lines, lookahead buffers |

All three implement `Deref`/`DerefMut<Target = [T]>`, so they can be indexed or passed anywhere a slice is expected.

```rust
use mkaudiolibrary::buffer::Buffer;

let mut buffer = Buffer::<f32>::new(1024);
for i in 0..1024 {
    buffer[i] = (i as f32 / 1024.0).sin();
}
println!("First sample: {}", buffer[0]);
```

### dsp

Audio processing components organized by category. `run()` methods take `&[f32]` input and `&mut [f32]` output directly.

#### Utility Functions

```rust
use mkaudiolibrary::dsp::{ratio_to_db, db_to_ratio};

let db = ratio_to_db(2.0);      // ~6.02 dB
let ratio = db_to_ratio(-6.0);  // ~0.5
```

#### Saturation (Analog Modeling)

Asymmetric logarithmic saturation model for analog-style harmonic generation:

- **Alpha parameters** - Independent drive/knee control for positive and negative signals
- **Beta parameters** - Separate compression/gain characteristics per polarity
- **Delta parameter** - DC bias offset for curve positioning
- **Gamma parameter** - Boolean polarity inversion

```rust
use mkaudiolibrary::dsp::Saturation;
use mkaudiolibrary::buffer::Buffer;

let sat = Saturation::new(
    10.0, 10.0,   // alpha_plus, alpha_minus (drive)
    1.0, 1.0,     // beta_plus, beta_minus (compression)
    0.0,          // delta (bias)
    false         // flip polarity
);

// Process single sample
let output = sat.process(0.8);

// Process buffer
let input = Buffer::from_slice(&[0.0, 0.5, 1.0, -0.5, -1.0]);
let mut output = Buffer::new(5);
sat.run(&input, &mut output);
```

For a physically-modeled alternative driven by an actual vacuum tube circuit (requires the `sim` feature):

```rust
#[cfg(feature = "sim")]
{
    use mkaudiolibrary::dsp::TubeSaturation;
    let mut tube = TubeSaturation::new(44100.0);
    let wet = tube.process(0.5);
}
```

#### Circuit Simulation (Modified Nodal Analysis)

Sample-by-sample circuit analysis for real-time filtering:

- **Component library** - Resistors, capacitors, and inductors with companion model discretization
- **Gaussian elimination** solver with partial pivoting
- **Single preprocessing step** builds the static admittance matrix
- **Per-sample updates** for reactive element state

```rust
use mkaudiolibrary::dsp::{Circuit, Resistor, Capacitor};

// Create an RC lowpass filter: fc = 1/(2πRC) ≈ 159Hz
let mut circuit = Circuit::new(44100.0, 2);
circuit.add_component(Box::new(Resistor::new(1, 2, 1000.0)));   // 1kΩ
circuit.add_component(Box::new(Capacitor::new(2, 0, 1e-6)));    // 1µF

// Build Y matrix (call once before processing)
circuit.preprocess(10.0);

// Process samples
let output = circuit.process(1.0, 2);  // Input voltage, probe node 2
```

#### IIR/FIR Filtering

RBJ biquad sections (with Butterworth cascade helper) and windowed-sinc FIR design:

```rust
use mkaudiolibrary::dsp::iir::{Biquad, BiquadType, IirFilter};
use mkaudiolibrary::dsp::fir::FirFilter;

// Single biquad section
let mut lowpass = Biquad::new(BiquadType::LowPass, 44100.0, 1000.0, 0.707, 0.0);
let y = lowpass.process(0.5);

// 4th-order Butterworth lowpass (two cascaded biquads)
let mut butterworth = IirFilter::butterworth(BiquadType::LowPass, 4, 44100.0, 1000.0);
let y2 = butterworth.process(0.5);

// Windowed-sinc FIR lowpass, 101 taps
let mut fir_lp = FirFilter::lowpass(101, 44100.0, 1000.0);
let y3 = fir_lp.process(0.5);
```

#### Dynamics Processing

```rust
use mkaudiolibrary::dsp::{Compression, Limit, Gate};
use mkaudiolibrary::buffer::Buffer;

// Compressor with soft knee
let mut compressor = Compression::new(44100.0);
compressor.threshold = -20.0;  // dB
compressor.ratio = 4.0;        // 4:1
compressor.attack = 10.0;      // ms
compressor.release = 100.0;    // ms
compressor.makeup = 6.0;       // dB
compressor.knee = 6.0;         // dB (soft knee width)

// Brickwall limiter
let mut limiter = Limit::new(44100.0);
limiter.gain = 0.0;            // dB input gain
limiter.ceiling = -0.1;        // dB output ceiling
limiter.release = 100.0;       // ms

// Downward-expanding noise gate with hold time
let mut gate = Gate::new(44100.0);
gate.threshold = -40.0;        // dB
gate.hold = 50.0;              // ms
gate.range = -60.0;            // dB attenuation when fully closed

// Process buffers
let input = Buffer::new(1024);
let mut output = Buffer::new(1024);
compressor.run(&input, &mut output);
```

#### Sample-Rate Conversion

```rust
use mkaudiolibrary::dsp::{Oversampler, Saturation, resample};

// Integer oversampling around a nonlinear stage (reduces aliasing)
let mut os = Oversampler::new(4, 44100.0);
let sat = Saturation::new(10.0, 10.0, 1.0, 1.0, 0.0, false);
let wet = os.process(1.0, |x| sat.process(x));

// Whole-buffer FFT-based resampling (offline/analysis use)
let input = vec![0.0f32; 1000];
let output = resample(&input, 44100.0, 48000.0);
```

#### Time-Based Effects

```rust
use mkaudiolibrary::dsp::{Convolution, Delay};
use mkaudiolibrary::buffer::Buffer;

// Convolution with impulse response
let impulse_response = vec![1.0, 0.5, 0.25, 0.125];
let mut conv = Convolution::new(&impulse_response);

let input = Buffer::new(1024);
let mut output = Buffer::new(1024);
conv.run(&input, &mut output);

// Feedback delay
let mut delay = Delay::new(250.0, 44100.0);  // 250ms delay
delay.feedback = 0.5;  // 50% feedback
delay.mix = 0.5;       // 50% wet
```

### audiofile

Load and save audio files with automatic format detection:

```rust
use mkaudiolibrary::audiofile::{AudioFile, FileFormat};

// Load audio file (WAV or AIFF auto-detected)
let mut audio = AudioFile::default();
audio.load("song.wav");

// Inspect file properties
println!("Format: {:?}", audio.format());
println!("Channels: {}", audio.num_channel());
println!("Samples: {}", audio.num_sample());
println!("Sample rate: {} Hz", audio.sample_rate());
println!("Bit depth: {}", audio.bit_depth());
println!("Duration: {:.2} seconds", audio.length());

// Direct channel access
if let Some(left) = audio.channel(0) {
    let peak = left.iter().fold(0.0_f32, |max, &s| max.max(s.abs()));
    println!("Peak level: {:.4}", peak);
}

// Modify samples
if let Some(left) = audio.channel_mut(0) {
    for sample in left.iter_mut() {
        *sample *= 0.5;  // Apply -6dB gain
    }
}

// Save in different format
audio.set_bit_depth(24);
audio.save("output.aiff", FileFormat::Aiff);
```

#### Buffer Integration

```rust
use mkaudiolibrary::audiofile::AudioFile;

let mut audio = AudioFile::default();
audio.load("input.wav");

// Convert to buffers for processing
let buffers = audio.to_buffers();

// ... process each channel ...

// Copy results back
audio.from_buffers(&buffers);
```

#### BWF (Broadcast Wave Format)

Support for professional broadcast metadata, markers, and tempo information:

```rust
use mkaudiolibrary::audiofile::{AudioFile, BextChunk, Marker};

let mut audio = AudioFile::default();
audio.load("broadcast.wav");

// Access BWF metadata
if let Some(bext) = audio.bext() {
    println!("Description: {}", bext.description);
    println!("Originator: {}", bext.originator);
    println!("Timecode: {} samples", bext.time_reference);
}

// Work with markers
for marker in audio.markers() {
    println!("Marker '{}' at sample {}", marker.label, marker.position);
}

// Add markers
audio.add_marker(Marker::new(44100, "Verse 1"));
audio.add_marker(Marker::new(88200, "Chorus"));

// Set tempo for DAW integration
audio.set_tempo(120.0);
audio.set_tempo_with_time_sig(120.0, 4, 4);

// Set tempo at a specific sample position
audio.set_tempo_at(140.0, 44100 * 30);  // 140 BPM starting at 30 seconds

// Set BWF metadata
let mut bext = BextChunk::with_description("Recording session", "Studio A");
bext.set_datetime("2025-01-15", "14:30:00");
audio.set_bext(bext);

// Save as BWF (includes bext chunk)
audio.save_bwf("output_bwf.wav");
```

### processor

MKAU plugin format for modular audio processing chains. `AudioIO` is a
thin, non-owning view: it borrows per-channel slices from storage the
*caller* owns for the life of the stream, rather than allocating its own
(matching how VST3/CoreAudio hand a plugin pointers into host-owned
memory). `input`/`sidechain_in`/`sidechain_out` are `Option` since a
generator plugin may have no input and sidechain busses are often absent;
`output` is always present.

```rust
use mkaudiolibrary::processor::{Processor, AudioIO, load};

// Load a plugin
let plugin = load("/path/to/plugins", "myplugin").expect("Failed to load");
println!("Loaded: {}", plugin.name());

// Prepare for playback
plugin.prepare_to_play(512, 44100);

// Own the actual sample storage for the stream's lifetime...
let input_storage = vec![vec![0.0f32; 512]; 2];
let mut output_storage = vec![vec![0.0f32; 512]; 2];

// ...and borrow an AudioIO view into it for each block.
let input: Vec<&[f32]> = input_storage.iter().map(Vec::as_slice).collect();
let mut output: Vec<&mut [f32]> = output_storage.iter_mut().map(Vec::as_mut_slice).collect();
let mut audio = AudioIO::new(Some(&input), &mut output, None, None);

// Process audio
plugin.run(&mut audio);
```

#### Creating Plugins

```rust
use mkaudiolibrary::processor::{Processor, AudioIO};

struct GainPlugin {
    gain: f32,
}

impl Processor for GainPlugin {
    fn init(&mut self) {}
    fn name(&self) -> String { String::from("Gain") }
    fn get_parameter(&self, _index: usize) -> f32 { self.gain }
    fn set_parameter(&mut self, _index: usize, value: f32) { self.gain = value; }
    fn get_parameter_name(&self, _index: usize) -> String { String::from("Gain") }
    fn prepare_to_play(&mut self, _buffer_size: usize, _sample_rate: usize) {}

    // `editor()` defaults to `None` (no GUI) - only override it for a plugin
    // with a `PluginEditor` (see the GUI example above).

    fn run(&self, audio: &mut AudioIO) {
        let Some(input) = audio.input else { return };
        for ch in 0..input.len().min(audio.output.len()) {
            let (input, output) = (input[ch], &mut audio.output[ch]);
            for i in 0..input.len().min(output.len()) {
                output[i] = input[i] * self.gain;
            }
        }
    }

    #[cfg(feature = "midi")]
    fn run_with_midi(&self, audio: &mut AudioIO, _midi: &mut MidiIO) {
        self.run(audio);
    }
}

// Export as dynamic library
mkaudiolibrary::declare_plugin!(GainPlugin, GainPlugin::new);
```

#### MIDI Processing (requires `midi` feature)

`MidiIO` follows the same non-owning pattern: `input` is always present,
`output` is `Option` (a plugin that only consumes MIDI has nowhere to write
outgoing messages).

```rust
#[cfg(feature = "midi")]
use mkaudiolibrary::processor::{Processor, AudioIO, MidiIO, MidiMessage};

// Process with MIDI
#[cfg(feature = "midi")]
let mut midi_input = vec![None; 512];
#[cfg(feature = "midi")]
let mut midi_output = vec![None; 512];
#[cfg(feature = "midi")]
let mut midi = MidiIO::new(&mut midi_input, Some(&mut midi_output));

// Add MIDI input messages
#[cfg(feature = "midi")]
{ midi.input[0] = Some(MidiMessage::note_on(0, 60, 100)); }

// Run processor with MIDI
#[cfg(feature = "midi")]
plugin.run_with_midi(&mut audio, &mut midi);

// Check MIDI output
#[cfg(feature = "midi")]
if let Some(output) = midi.output.as_deref() {
    for msg in output.iter().flatten() {
        println!("MIDI out: {:?}", msg);
    }
}
```

### host (Plugin Hosting)

Load and run third-party plugins - MKAP (always available), VST3 (`vst3` feature), and AUv2 (`au` feature, macOS only) - through one `HostedPlugin` trait:

```rust
use mkaudiolibrary::host::{scan_vst3, load};
use mkaudiolibrary::processor::AudioIO;
use std::path::Path;

// Scan a directory for VST3 plugins
let found = scan_vst3(Path::new("/Library/Audio/Plug-Ins/VST3"));
for d in &found {
    println!("[{}] {} - {}", d.format, d.vendor, d.name);
}

// Load and run one
let mut plugin = load(&found[0]).expect("failed to load");
plugin.prepare(48000, 512).expect("prepare failed");
plugin.set_active(true).expect("activate failed");

println!("{} in / {} out, {} parameters", plugin.num_inputs(), plugin.num_outputs(), plugin.num_parameters());
for i in 0..plugin.num_parameters() {
    println!("  [{}] {} = {:.3}", i, plugin.parameter_name(i), plugin.get_parameter(i));
}

let input_storage = vec![vec![0.0f32; 512]; plugin.num_inputs()];
let mut output_storage = vec![vec![0.0f32; 512]; plugin.num_outputs()];
let input: Vec<&[f32]> = input_storage.iter().map(Vec::as_slice).collect();
let mut output: Vec<&mut [f32]> = output_storage.iter_mut().map(Vec::as_mut_slice).collect();
let mut audio = AudioIO::new(Some(&input), &mut output, None, None);
plugin.process(&mut audio);
```

The VST3 backend talks directly to a plugin's `IComponent`/`IAudioProcessor`/`IEditController` COM-style interfaces using hand-written vtables matching Steinberg's public ABI - no vendored SDK or C++ toolchain required. Both the VST3 and AUv2 backends negotiate 32-bit float first (this library's own native sample format, and what every VST3 plugin and most third-party AUs support), falling back to 64-bit float with a per-block conversion scratch buffer only for the plugins that require it.

### sim (Analog Circuit Simulation, optional feature)

Physically-modeled vacuum tubes, diodes, transistors, op-amps, potentiometers, switches, and passive/RLC filters, merged in from [libmksim](https://github.com/mkaudio-company/libmksim). Linear passive networks use Wave Digital Filters (series/parallel adaptor trees); nonlinear devices use local Newton-Raphson solvers over the Koren/Shockley/Ebers-Moll/square-law equations. Enable with the `sim` feature; `sim-avx2`/`sim-avx512`/`sim-neon` additionally enable SIMD backends for its internal fast-math.

```rust
#[cfg(feature = "sim")]
{
    use mkaudiolibrary::sim::components::tubes::{TriodeStage, PARAMS_12AX7};
    use mkaudiolibrary::sim::components::CircuitComponent;

    let mut triode = TriodeStage::new(PARAMS_12AX7.clone());
    triode.prepare(44100.0);

    let input = [0.5f32; 64];
    let mut output = [0.0f32; 64];
    triode.process_block(&input, &mut output);
}
```

[`dsp::TubeSaturation`](#saturation-analog-modeling) builds on this module to provide a physically-modeled alternative to `dsp::Saturation`.

### tf (Time-Frequency Analysis)

DFT, FFT, DCT, STFT/multi-resolution STFT, CWT, CQT, and mel spectrograms. Hot inner loops go through the same SIMD dot-product primitives as `dsp`.

```rust
use mkaudiolibrary::tf::{fft, stft, mel};

// FFT of a real signal (any length - radix-2 or Bluestein's algorithm as needed)
let spectrum = fft::rfft(&signal);

// STFT
let stft_config = stft::StftConfig::new(1024, 256, stft::WindowFunction::Hann);
let frames = stft::stft(&signal, &stft_config);

// Mel spectrogram
let mel_config = mel::MelConfig { sample_rate: 44100.0, num_mels: 80, min_freq: 20.0, max_freq: 20000.0 };
let mel_frames = mel::mel_spectrogram(&signal, &stft_config, &mel_config);
```

For Cohen's class of *bilinear* time-frequency distributions (Wigner-Ville, Choi-Williams, Rihaczek, ...), see the sibling [`bilinear_tf`](https://crates.io/crates/bilinear_tf) crate - `tf` covers the standard linear transforms.

### realtime (Optional Feature)

Real-time audio streaming I/O inspired by the C++ RTAudio library, with real hardware-clocked backends (not a simulated/dummy stream) per platform. Enable with the `realtime` feature.

#### Supported Backends

| Platform | Backend | API |
|----------|---------|-----|
| macOS | CoreAudio | `Api::CoreAudio` |
| Windows | WASAPI | `Api::Wasapi` |
| Linux | ALSA | `Api::Alsa` |

#### Basic Usage

```rust
use mkaudiolibrary::realtime::{Realtime, StreamParameters, AudioCallback};

// Create audio interface (auto-detects best API)
let mut audio = Realtime::new(None).unwrap();

// List available devices
for id in audio.get_device_ids() {
    if let Ok(info) = audio.get_device_info(id) {
        println!("{}: {} (in:{}, out:{})",
            info.id, info.name,
            info.input_channels, info.output_channels);
    }
}

// Define audio callback
let callback: AudioCallback = Box::new(|output, input, frames, _time, _status| {
    // Simple pass-through with gain
    for i in 0..output.len() {
        output[i] = input.get(i).copied().unwrap_or(0.0) * 0.5;
    }
    0  // Return 0 to continue, 1 to stop, 2 to abort
});

// Configure output stream
let output_params = StreamParameters {
    device_id: audio.get_default_output_device(),
    num_channels: 2,
    first_channel: 0,
};

// Configure input stream
let input_params = StreamParameters {
    device_id: audio.get_default_input_device(),
    num_channels: 2,
    first_channel: 0,
};

// Open duplex stream
audio.open_stream(
    Some(&output_params),
    Some(&input_params),
    44100,  // Sample rate
    256,    // Buffer frames
    callback,
    None,   // Optional stream options
).unwrap();

// Start streaming
audio.start_stream().unwrap();

// ... do work ...

// Stop and cleanup
audio.stop_stream().unwrap();
audio.close_stream();
```

#### Stereo Processing Helper

```rust
use mkaudiolibrary::realtime::{stereo_callback, Realtime, StreamParameters};

// Create callback that works with separate L/R channels
let callback = stereo_callback(|left_in, right_in, left_out, right_out, frames| {
    for i in 0..frames {
        // Simple stereo processing
        left_out[i] = left_in[i] * 0.8;
        right_out[i] = right_in[i] * 0.8;
    }
});

let mut audio = Realtime::new(None).unwrap();
// ... configure and open stream with callback ...
```

#### Buffer Integration

```rust
use mkaudiolibrary::realtime::{deinterleave, interleave, AudioCallback};
use mkaudiolibrary::buffer::Buffer;
use mkaudiolibrary::dsp::Compression;
use std::sync::{Arc, Mutex};

// Shared DSP processor
let compressor = Arc::new(Mutex::new(Compression::new(44100.0)));
let comp_clone = compressor.clone();

let callback: AudioCallback = Box::new(move |output, input, frames, _, _| {
    // Deinterleave to separate channel buffers
    let mut input_buffers = deinterleave(input, 2, frames);

    // Process each channel
    let mut comp = comp_clone.lock().unwrap();
    let output_buffers: Vec<Buffer<f32>> = input_buffers.iter_mut()
        .map(|buf| {
            let mut out = Buffer::new(frames);
            comp.run(buf, &mut out);
            out
        })
        .collect();

    // Interleave back to output
    interleave(&output_buffers, output, frames);
    0
});
```

## Supported Audio Formats

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| WAV | `.wav` | Yes | Yes | PCM, IEEE Float |
| BWF | `.wav` | Yes | Yes | WAV with bext chunk, markers, tempo |
| AIFF | `.aiff`, `.aif` | Yes | Yes | Uncompressed, AIFC |

Supported bit depths: 8, 16, 24, 32-bit

## Changelog

### 2.1.0
- **`f32` throughout**: `dsp`, `processor::AudioIO`, `audiofile`, `buffer`, and the plugin hosting backends now operate on `f32` (previously `f64`), matching VST3/AU/MKAP's native sample format and `sim`'s own circuit models - removes a redundant conversion at every plugin-hosting and `sim` boundary
- **Unlocked buffers**: `Buffer`/`PushBuffer`/`CircularBuffer` dropped their internal `Arc<RwLock<...>>` - they're plain owned containers now (`Deref`/`DerefMut<Target = [T]>`, no `.read()`/`.write()` guards), since nothing in this crate's own processing graph shared them across threads without its own synchronization
- **`dsp` API**: `run()` methods now take `&[f32]`/`&mut [f32]` directly instead of `&Buffer<f64>`
- **New analog circuit simulation**: `sim` module (feature-gated) merged in from [libmksim](https://github.com/mkaudio-company/libmksim) - vacuum tubes, diodes, transistors, op-amps, potentiometers, switches, and passive/RLC filters via Wave Digital Filters and Newton-Raphson solvers; `dsp::TubeSaturation` builds on it for a physically-modeled saturation alternative
- **Expanded `dsp`**: new `dsp::iir` (RBJ biquad + Butterworth cascades), `dsp::fir` (windowed-sinc design), `dsp::Gate` (noise gate), `dsp::Oversampler` (integer oversampling), and `dsp::resampling` (FFT-based whole-buffer resampling)
- **SIMD independence**: `crate::simd` split into per-backend files (`scalar`/`x86_64`/`aarch64`), with `dot`/`mul_elementwise`/`mix_scalar` as the single canonical `f32` primitives shared by `dsp` and `tf` (previously duplicated under `_f32`-suffixed names during the `f64` -> `f32` transition); `sim`'s own `f32`-lane trait-based SIMD abstraction now lives at `crate::simd::generic`, independent of the `simd` feature
- VST3/AUv2 hosting now prefers 32-bit float negotiation (matching this library's native format) with a 64-bit float fallback, inverted from the previous `f64`-native preference
- **`gui` feature**: now backed by [mkapk](https://github.com/mkaudio-company/mkapk)'s `mkapk-core`/`mkapk-host` (a plugin-editor GUI framework: widget tree, geometry, paint commands, and `PluginEditor`/`EditorHost` parent-window-embedding traits) instead of `mkgraphic` - a better fit for hosted plugin editors than a general-purpose windowing crate, and pure Rust with no platform-specific dependencies, so it's now verified on Windows/Linux too, not just macOS. `Processor::get_view()`/`get_view_mut()`/`get_preferred_size()` were replaced by a single `editor() -> Option<&mut dyn PluginEditor>` (default `None`)
- **`AudioIO`/`MidiIO` are non-owning views now**: both gained a lifetime parameter and their fields became borrowed slices (`&[&[f32]]`/`&mut [&mut [f32]]`, `&mut [Option<MidiMessage>]`) instead of owned `Vec<Buffer<f32>>`/`Box<[Option<MidiMessage>]>` - the caller owns the actual sample/message storage for the stream's lifetime and borrows a view into it per block, avoiding the allocate-and-copy `AudioIO::new(channel_count, buffer_size)` used to do. `AudioIO::input`/`sidechain_in`/`sidechain_out` and `MidiIO::output` are `Option` (generator plugins have no input, sidechain busses are often absent, and not every plugin produces MIDI output); `AudioIO::output` and `MidiIO::input` are always present. `AudioIO::set_channel`/`resize` and `MidiIO::resize` were removed along with the owned storage they resized

### 2.0.0
- **Real realtime backends**: `realtime` module restructured into per-platform backends (mirroring `mkmidilibrary`'s design) - CoreAudio (AUHAL), WASAPI (event-driven `IAudioClient`), and ALSA (blocking `snd_pcm`) now drive real, hardware-clocked audio I/O instead of a simulated dummy stream
- **Plugin hosting**: new `host` module with a unified `HostedPlugin` trait - MKAP (always available), VST3 (`vst3` feature, hand-written COM/vtable FFI, no vendored SDK needed), and AUv2 (`au` feature, macOS)
- **Time-frequency analysis**: new `tf` module - DFT, FFT (radix-2 + Bluestein), DCT, STFT/multi-resolution STFT, CWT, CQT, mel spectrograms
- **SIMD**: new `simd` feature with AVX2+FMA/SSE2 (`x86_64`) and NEON (`aarch64`) dot-product/elementwise-multiply/mix primitives, used by `dsp` and `tf`'s hot loops
- **No-allocation steady state**: `Compression`, `Limit`, and `Delay` now use pre-allocated scratch buffers instead of allocating per `run()` call
- Dependency updates: `libloading` 0.9, `no_denormals` 0.3 (now `unsafe fn`), `windows` 0.62, `alsa` 0.9 (pinned to match `mkmidilibrary`'s native-link constraint), `mkmidilibrary` 0.2, `mkgraphic` 0.4
- `Processor` trait gained a `num_parameters()` method (default `0`) so hosts can enumerate MKAP plugin parameters

### 1.4.0
- **GUI support**: New `gui` feature with mkgraphic integration for plugin UI
- Replaced `open_window()`/`close_window()` with `get_view()`, `get_view_mut()`, and `get_preferred_size()` methods
- Re-exports `View`, `Window`, `WindowBuilder`, `Extent`, `Point` from mkgraphic

### 1.3.0
- **Buffer-based Processor I/O**: Changed `Processor::run()` to use `AudioIO` struct with `Buffer` types
- **MIDI support**: New `midi` feature with `MidiIO` struct and `run_with_midi()` method
- **mkmidilibrary integration**: Re-exports `MidiMessage` from mkmidilibrary for MIDI event handling
- `AudioIO` provides `new()` and `set_channel()` constructors for flexible channel configurations

### 1.2.0
- BWF (Broadcast Wave Format) support with `bext` chunk for broadcast metadata
- Markers/cue points with labels via `cue` and `LIST` chunks
- Tempo information via `acid` chunk for DAW integration

### 1.1.0
- Added `realtime` feature with cross-platform audio streaming I/O
- `Realtime` struct providing callback-based audio input/output
- Platform backends: CoreAudio (macOS), WASAPI (Windows), ALSA (Linux)
- Helper functions for buffer interleaving/deinterleaving
- `stereo_callback` wrapper for simplified stereo processing

### 1.0.0
- Major update with thread-safe buffers using `RwLock`
- New saturation model with asymmetric numeric modeling
- Circuit simulation with MNA solver
- Refined compression and limiting with proper envelope detection
- Enhanced audiofile module with Buffer integration
- Comprehensive documentation for all modules

### 0.3.0
- Reconstructed sized buffer, used slice instead of buffer for plugins

### 0.2.x
- Added audiofile module
- Lock/unlock mechanisms for data safety
- Updated processor loader and documentation

### 0.1.x
- Initial development versions
- Buffer implementations with reference counting
- Basic DSP components

## License

MIT License
