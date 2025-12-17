[![](https://img.shields.io/crates/v/mkaudiolibrary.svg)](https://crates.io/crates/mkaudiolibrary)
[![](https://img.shields.io/crates/l/mkaudiolibrary.svg)](https://crates.io/crates/mkaudiolibrary)
[![](https://docs.rs/mkaudiolibrary/badge.svg)](https://docs.rs/mkaudiolibrary/)

# mkaudiolibrary

A Rust library for real-time audio signal processing, featuring analog modeling through numeric functions and circuit simulation via Modified Nodal Analysis (MNA).

## Features

- **Thread-safe buffers** with `RwLock`-based concurrent access
- **Analog modeling** for tube/tape-style saturation with asymmetric parameters
- **Circuit simulation** with real-time transient analysis using MNA
- **DSP primitives** including convolution, compression, limiting, and delay
- **Audio file I/O** for WAV and AIFF formats with Buffer integration
- **Plugin system** via MKAU format for modular processing chains
- **Real-time streaming** via RTAudio-style API (optional `realtime` feature)
- **Zero-copy design** with boxed slices for minimal allocation overhead

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mkaudiolibrary = "1.0"
```

For real-time audio streaming, enable the `realtime` feature:

```toml
[dependencies]
mkaudiolibrary = { version = "1.0", features = ["realtime"] }
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

// Convert to thread-safe buffers for processing
let buffers = audio.to_buffers();

// Apply compression to each channel
let mut comp = Compression::new(audio.sample_rate() as f64);
comp.threshold = -12.0;
comp.ratio = 4.0;

for buffer in &buffers {
    let output = Buffer::new(buffer.len());
    comp.run(buffer, &output);
    // ... copy processed output back
}

// Save result
audio.save("output.wav", FileFormat::Wav);
```

## Modules

### buffer

Thread-safe audio buffers designed for concurrent multi-threaded access:

| Type | Description | Use Case |
|------|-------------|----------|
| `Buffer<T>` | General-purpose buffer with read/write locking | Sample storage, inter-thread communication |
| `PushBuffer<T>` | FIFO buffer that shifts samples on push | FIR filters, convolution |
| `CircularBuffer<T>` | Ring buffer with power-of-2 sizing | Delay lines, lookahead buffers |

All buffers use `Arc<RwLock<...>>` internally, allowing multiple concurrent readers or exclusive write access.

```rust
use mkaudiolibrary::buffer::Buffer;
use std::thread;

let buffer = Buffer::<f64>::new(1024);
let buffer_clone = buffer.clone();  // Shares underlying data

// Writer thread
let writer = thread::spawn(move || {
    let mut guard = buffer_clone.write();
    for i in 0..1024 {
        guard[i] = (i as f64 / 1024.0).sin();
    }
});

writer.join().unwrap();

// Reader thread
let guard = buffer.read();
println!("First sample: {}", guard[0]);
```

### dsp

Audio processing components organized by category:

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
let output = Buffer::new(5);
sat.run(&input, &output);
```

#### Circuit Simulation (Modified Nodal Analysis)

Sample-by-sample circuit analysis for real-time filtering:

- **Component library** - Resistors, capacitors, and inductors with companion model discretization
- **Gaussian elimination** solver with partial pivoting
- **Single preprocessing step** builds the static admittance matrix
- **Per-sample updates** for reactive element state

```rust
use mkaudiolibrary::dsp::{Circuit, Resistor, Capacitor, Inductor};

// Create an RC lowpass filter: fc = 1/(2πRC) ≈ 159Hz
let mut circuit = Circuit::new(44100.0, 2);
circuit.add_component(Box::new(Resistor::new(1, 2, 1000.0)));   // 1kΩ
circuit.add_component(Box::new(Capacitor::new(2, 0, 1e-6)));    // 1µF

// Build Y matrix (call once before processing)
circuit.preprocess(10.0);

// Process samples
for input_sample in audio_input {
    let output = circuit.process(input_sample, 2);  // Input voltage, probe node 2
}
```

#### Dynamics Processing

```rust
use mkaudiolibrary::dsp::{Compression, Limit};
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

// Process buffers
let input = Buffer::new(1024);
let output = Buffer::new(1024);
compressor.run(&input, &output);
```

#### Time-Based Effects

```rust
use mkaudiolibrary::dsp::{Convolution, Delay};
use mkaudiolibrary::buffer::Buffer;

// Convolution with impulse response
let impulse_response = vec![1.0, 0.5, 0.25, 0.125];
let conv = Convolution::new(&impulse_response).unwrap();

let input = Buffer::new(1024);
let output = Buffer::new(1024);
conv.run(&input, &output);

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
    let peak = left.iter().fold(0.0_f64, |max, &s| max.max(s.abs()));
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

Seamlessly integrate with thread-safe buffers for DSP processing:

```rust
use mkaudiolibrary::audiofile::AudioFile;
use mkaudiolibrary::buffer::Buffer;

let mut audio = AudioFile::default();
audio.load("input.wav");

// Convert to thread-safe buffers
let buffers = audio.to_buffers();

// Process each channel in parallel (example)
// ... parallel processing ...

// Copy results back
audio.from_buffers(&buffers);
```

### processor

MKAU plugin format for modular audio processing chains:

```rust
use mkaudiolibrary::processor::{Processor, load};

// Load a plugin
let plugin = load("/path/to/plugins", "myplugin").expect("Failed to load");
println!("Loaded: {}", plugin.name());

// Prepare for playback
plugin.prepare_to_play(512, 44100);

// Process audio
let input: Vec<&[f64]> = vec![&left_in, &right_in];
let mut output: Vec<&mut [f64]> = vec![&mut left_out, &mut right_out];
plugin.run(&input, &[], &mut output, &mut []);
```

#### Creating Plugins

```rust
use mkaudiolibrary::processor::Processor;
use mkaudiolibrary::buffer::Buffer;

struct GainPlugin {
    gain: f64,
}

impl Processor for GainPlugin {
    fn init(&mut self) {}
    fn name(&self) -> String { String::from("Gain") }
    fn get_parameter(&self, _index: usize) -> f64 { self.gain }
    fn set_parameter(&mut self, _index: usize, value: f64) { self.gain = value; }
    fn get_parameter_name(&self, _index: usize) -> String { String::from("Gain") }
    fn open_window(&self) {}
    fn close_window(&self) {}
    fn prepare_to_play(&mut self, _buffer_size: usize, _sample_rate: usize) {}

    fn run(&self, input: &[&[f64]], _sidechain_in: &[&[f64]],
           output: &mut [&mut [f64]], _sidechain_out: &mut [&mut [f64]]) {
        for ch in 0..input.len() {
            for i in 0..input[ch].len() {
                output[ch][i] = input[ch][i] * self.gain;
            }
        }
    }
}

// Export as dynamic library
mkaudiolibrary::declare_plugin!(GainPlugin, GainPlugin::new);
```

### realtime (Optional Feature)

Real-time audio streaming I/O inspired by the C++ RTAudio library. Enable with the `realtime` feature.

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
    let input_buffers = deinterleave(input, 2, frames);

    // Process each channel
    let mut comp = comp_clone.lock().unwrap();
    let output_buffers: Vec<Buffer<f64>> = input_buffers.iter()
        .map(|buf| {
            let out = Buffer::new(frames);
            comp.run(buf, &out);
            out
        })
        .collect();

    // Interleave back to output
    interleave(&output_buffers, output, frames);
    0
});
```

## Thread Safety

All buffer types implement `Send + Sync` and use interior mutability with `RwLock`:

| Operation | Behavior |
|-----------|----------|
| `buffer.read()` | Returns `BufferReadGuard`, multiple allowed |
| `buffer.write()` | Returns `BufferWriteGuard`, exclusive access |
| `buffer.clone()` | Creates new handle to same data (like `Arc`) |

Guards automatically release locks when dropped (RAII pattern).

## Supported Audio Formats

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| WAV | `.wav` | Yes | Yes | PCM, IEEE Float |
| AIFF | `.aiff`, `.aif` | Yes | Yes | Uncompressed, AIFC |

Supported bit depths: 8, 16, 24, 32-bit

## Changelog

### 1.1.0
- Added `realtime` feature with cross-platform audio streaming I/O
- `Realtime` struct providing callback-based audio input/output
- Platform backends: CoreAudio (macOS), WASAPI (Windows), ALSA (Linux)
- Helper functions for buffer interleaving/deinterleaving
- `stereo_callback` wrapper for simplified stereo processing
- Seamless integration with thread-safe `Buffer` types

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

This library is dual-licensed:

- **GPL-3.0** for open source projects
- **Commercial license** available for closed source usage

For commercial licensing inquiries, contact: minjaekim@mkaudio.company
