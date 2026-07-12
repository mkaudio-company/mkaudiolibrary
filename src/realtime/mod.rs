//! Real-time audio streaming I/O inspired by RTAudio.
//!
//! This module provides cross-platform real-time audio input/output capabilities,
//! translated from the C++ RTAudio library design. It integrates with the library's
//! thread-safe `Buffer` types for seamless DSP processing pipelines.
//!
//! # Supported Backends
//!
//! | Platform | Backend | API |
//! |----------|---------|-----|
//! | macOS | CoreAudio (AUHAL) | `Api::CoreAudio` |
//! | Windows | WASAPI | `Api::Wasapi` |
//! | Linux | ALSA | `Api::Alsa` |
//!
//! Each backend drives a real, callback-based hardware audio stream (no
//! simulated timing) via a small internal `Backend` trait, mirroring
//! upstream RtAudio's `RtApi` base class and its per-platform subclasses
//! (`RtApiCore`, `RtApiWasapi`, `RtApiAlsa`).
//!
//! # Audio Format
//!
//! Audio samples are represented as normalized `f32` values in the range -1.0 to 1.0,
//! consistent with the rest of the library. Format conversion happens automatically
//! at the hardware interface.
//!
//! # Callback Model
//!
//! Audio processing uses a callback function that receives input samples and fills
//! output samples. The callback runs in a high-priority audio thread owned by the
//! platform's audio driver (not a library-managed sleep loop).
//!
//! ```ignore
//! use mkaudiolibrary::realtime::{Realtime, StreamParameters, AudioCallback};
//!
//! // Define callback
//! let callback: AudioCallback = Box::new(|output, input, frames, time, status| {
//!     // Simple pass-through
//!     for i in 0..frames {
//!         output[i] = input[i];
//!     }
//!     0  // Return 0 to continue, 1 to stop, 2 to abort
//! });
//!
//! // Create audio interface
//! let mut audio = Realtime::new(None).unwrap();
//!
//! // Configure streams
//! let output_params = StreamParameters {
//!     device_id: audio.get_default_output_device(),
//!     num_channels: 2,
//!     first_channel: 0,
//! };
//!
//! let input_params = StreamParameters {
//!     device_id: audio.get_default_input_device(),
//!     num_channels: 2,
//!     first_channel: 0,
//! };
//!
//! // Open stream
//! audio.open_stream(
//!     Some(&output_params),
//!     Some(&input_params),
//!     44100,
//!     256,
//!     callback,
//!     None,
//! ).unwrap();
//!
//! // Start streaming
//! audio.start_stream().unwrap();
//! ```
//!
//! # Buffer Integration
//!
//! Use with thread-safe `Buffer` types for DSP processing:
//!
//! ```ignore
//! use mkaudiolibrary::realtime::{Realtime, StreamParameters, AudioCallback};
//! use mkaudiolibrary::buffer::Buffer;
//! use mkaudiolibrary::dsp::Compression;
//! use std::sync::Arc;
//!
//! // Shared state for DSP
//! let compressor = Arc::new(std::sync::Mutex::new(Compression::new(44100.0)));
//! let comp_clone = compressor.clone();
//!
//! let callback: AudioCallback = Box::new(move |output, input, frames, _, _| {
//!     let mut comp = comp_clone.lock().unwrap();
//!     for i in 0..frames {
//!         output[i] = comp.process(input[i]);
//!     }
//!     0
//! });
//! ```

mod dummy_impl;

#[cfg(target_os = "macos")]
mod coreaudio_impl;

#[cfg(target_os = "windows")]
mod wasapi_impl;

#[cfg(target_os = "linux")]
mod alsa_impl;

use std::fmt;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use crate::buffer::Buffer;

// ==========================================
// Enums - Translated from RTAudio
// ==========================================

/// Audio API backend specifier.
///
/// Translated from `RtAudio::Api` enum in the C++ RTAudio library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Api {
    /// Search for a working compiled API (default).
    #[default]
    Unspecified,
    /// macOS CoreAudio API.
    CoreAudio,
    /// Linux ALSA API.
    Alsa,
    /// Linux PulseAudio API.
    Pulse,
    /// Linux OSS API.
    Oss,
    /// Jack Audio Connection Kit.
    Jack,
    /// Windows WASAPI API.
    Wasapi,
    /// Windows ASIO API.
    Asio,
    /// Windows DirectSound API.
    DirectSound,
    /// Dummy API for testing (no audio).
    Dummy,
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Api::Unspecified => "Unspecified",
            Api::CoreAudio => "CoreAudio",
            Api::Alsa => "ALSA",
            Api::Pulse => "PulseAudio",
            Api::Oss => "OSS",
            Api::Jack => "Jack",
            Api::Wasapi => "WASAPI",
            Api::Asio => "ASIO",
            Api::DirectSound => "DirectSound",
            Api::Dummy => "Dummy",
        };
        write!(f, "{}", name)
    }
}

/// Audio sample format specifier.
///
/// Translated from `RtAudioFormat` flags in the C++ RTAudio library.
/// Note: This library normalizes all formats to f32 internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SampleFormat {
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 24-bit signed integer (packed in 3 bytes).
    Int24,
    /// 32-bit signed integer.
    Int32,
    /// 32-bit floating point normalized between ±1.0.
    Float32,
    /// 64-bit floating point normalized between ±1.0.
    #[default]
    Float64,
}

impl SampleFormat {
    /// Get the size in bytes for this format.
    pub fn byte_size(&self) -> usize {
        match self {
            SampleFormat::Int8 => 1,
            SampleFormat::Int16 => 2,
            SampleFormat::Int24 => 3,
            SampleFormat::Int32 => 4,
            SampleFormat::Float32 => 4,
            SampleFormat::Float64 => 8,
        }
    }
}

/// Stream configuration flags.
///
/// Translated from `RtAudioStreamFlags` in the C++ RTAudio library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StreamFlags {
    /// Use non-interleaved buffers (default: interleaved).
    pub noninterleaved: bool,
    /// Attempt to minimize latency.
    pub minimize_latency: bool,
    /// Attempt to grab device for exclusive use.
    pub hog_device: bool,
    /// Try to select realtime scheduling for callback thread.
    pub schedule_realtime: bool,
}

/// Stream status indicators passed to callback.
///
/// Translated from `RtAudioStreamStatus` in the C++ RTAudio library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StreamStatus {
    /// Input data was discarded due to overflow.
    pub input_overflow: bool,
    /// Output buffer ran empty (underflow).
    pub output_underflow: bool,
}

// ==========================================
// Structures - Translated from RTAudio
// ==========================================

/// Stream parameter specification for input or output.
///
/// Translated from `RtAudio::StreamParameters` in the C++ RTAudio library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamParameters {
    /// Device identifier (0 = first device).
    pub device_id: usize,
    /// Number of channels to open.
    pub num_channels: usize,
    /// First channel index on the device.
    pub first_channel: usize,
}

impl Default for StreamParameters {
    fn default() -> Self {
        Self {
            device_id: 0,
            num_channels: 2,
            first_channel: 0,
        }
    }
}

/// Optional stream configuration options.
///
/// Translated from `RtAudio::StreamOptions` in the C++ RTAudio library.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Stream configuration flags.
    pub flags: StreamFlags,
    /// Number of buffers for the stream (0 = auto).
    pub number_of_buffers: usize,
    /// Stream name (for JACK, etc.).
    pub stream_name: String,
    /// Scheduling priority (1-99, 0 = default).
    pub priority: i32,
}

/// Information about an audio device.
///
/// Translated from `RtAudio::DeviceInfo` in the C++ RTAudio library.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device identifier.
    pub id: usize,
    /// Character string for device name.
    pub name: String,
    /// Maximum output channels supported.
    pub output_channels: usize,
    /// Maximum input channels supported.
    pub input_channels: usize,
    /// Maximum simultaneous input/output channels.
    pub duplex_channels: usize,
    /// Whether this is the default output device.
    pub is_default_output: bool,
    /// Whether this is the default input device.
    pub is_default_input: bool,
    /// Supported sample rates.
    pub sample_rates: Vec<usize>,
    /// Preferred sample rate.
    pub preferred_sample_rate: usize,
    /// Native sample formats supported.
    pub native_formats: Vec<SampleFormat>,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            output_channels: 0,
            input_channels: 0,
            duplex_channels: 0,
            is_default_output: false,
            is_default_input: false,
            sample_rates: vec![44100, 48000, 96000],
            preferred_sample_rate: 44100,
            native_formats: vec![SampleFormat::Float32],
        }
    }
}

// ==========================================
// Error Types
// ==========================================

/// Realtime audio error types.
///
/// Translated from `RtAudioErrorType` in the C++ RTAudio library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MKAudioError {
    /// A non-critical error.
    Warning(String),
    /// A non-critical error which might be useful for debugging.
    DebugWarning(String),
    /// The default, unspecified error type.
    Unspecified(String),
    /// No devices found on system.
    NoDevicesFound,
    /// An invalid device ID was specified.
    InvalidDevice(String),
    /// A device in use was unexpectedly disconnected.
    DeviceDisconnect(String),
    /// An error occurred during memory allocation.
    MemoryError(String),
    /// An invalid parameter was specified to a function.
    InvalidParameter(String),
    /// The function was called incorrectly.
    InvalidUse(String),
    /// A system driver error occurred.
    DriverError(String),
    /// A system error occurred.
    SystemError(String),
    /// A thread error occurred.
    ThreadError(String),
}

impl std::fmt::Display for MKAudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MKAudioError::Warning(s) => write!(f, "Warning: {}", s),
            MKAudioError::DebugWarning(s) => write!(f, "Debug: {}", s),
            MKAudioError::Unspecified(s) => write!(f, "Error: {}", s),
            MKAudioError::NoDevicesFound => write!(f, "No audio devices found"),
            MKAudioError::InvalidDevice(s) => write!(f, "Invalid device: {}", s),
            MKAudioError::DeviceDisconnect(s) => write!(f, "Device disconnected: {}", s),
            MKAudioError::MemoryError(s) => write!(f, "Memory error: {}", s),
            MKAudioError::InvalidParameter(s) => write!(f, "Invalid parameter: {}", s),
            MKAudioError::InvalidUse(s) => write!(f, "Invalid use: {}", s),
            MKAudioError::DriverError(s) => write!(f, "Driver error: {}", s),
            MKAudioError::SystemError(s) => write!(f, "System error: {}", s),
            MKAudioError::ThreadError(s) => write!(f, "Thread error: {}", s),
        }
    }
}

impl std::error::Error for MKAudioError {}

/// Result type for Realtime audio operations.
pub type MKAudioResult<T> = Result<T, MKAudioError>;

// ==========================================
// Callback Type
// ==========================================

/// Audio callback function type.
///
/// Translated from `RtAudioCallback` in the C++ RTAudio library.
///
/// # Arguments
/// * `output` - Output buffer to fill (interleaved samples), empty for an input-only stream
/// * `input` - Input buffer to read (interleaved samples), empty for an output-only stream
/// * `frames` - Number of frames (samples per channel)
/// * `stream_time` - Stream time in seconds since start
/// * `status` - Stream status flags (overflow/underflow)
///
/// # Returns
/// * `0` - Continue streaming
/// * `1` - Stop stream and drain output
/// * `2` - Abort stream immediately
///
/// # Realtime Safety
/// This closure runs on the platform's real-time audio thread. Avoid
/// allocating, blocking locks, or anything else that could stall it.
pub type AudioCallback = Box<dyn FnMut(&mut [f32], &[f32], usize, f32, StreamStatus) -> i32 + Send>;

// ==========================================
// Backend trait - one real implementation per platform
// ==========================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamState {
    Closed,
    Stopped,
    Running,
}

/// Internal per-platform audio I/O backend, analogous to upstream RtAudio's
/// `RtApi` abstract base class. Each platform module (`coreaudio_impl`,
/// `wasapi_impl`, `alsa_impl`, `dummy_impl`) provides one implementation
/// that drives a real hardware-clocked callback thread — there is no
/// library-managed sleep loop simulating timing.
pub(crate) trait Backend: Send {
    fn device_ids(&self) -> Vec<usize>;
    fn device_info(&self, device_id: usize) -> MKAudioResult<DeviceInfo>;
    fn default_output_device(&self) -> usize;
    fn default_input_device(&self) -> usize;

    /// Open (and allocate, but not yet start) a stream. Returns the actual
    /// buffer size negotiated with the device.
    #[allow(clippy::too_many_arguments)]
    fn open_stream(
        &mut self,
        output_params: Option<&StreamParameters>,
        input_params: Option<&StreamParameters>,
        sample_rate: usize,
        buffer_frames: usize,
        callback: AudioCallback,
        options: &StreamOptions,
    ) -> MKAudioResult<usize>;

    fn start(&mut self) -> MKAudioResult<()>;
    fn stop(&mut self) -> MKAudioResult<()>;
    fn close(&mut self);
    fn is_running(&self) -> bool;
    fn stream_time(&self) -> f32;
    fn latency_samples(&self) -> usize;
}

fn make_backend(api: Api) -> Box<dyn Backend> {
    match api {
        #[cfg(target_os = "macos")]
        Api::CoreAudio => Box::new(coreaudio_impl::CoreAudioBackend::new()),

        #[cfg(target_os = "windows")]
        Api::Wasapi => Box::new(wasapi_impl::WasapiBackend::new()),

        #[cfg(target_os = "linux")]
        Api::Alsa | Api::Pulse => Box::new(alsa_impl::AlsaBackend::new()),

        _ => Box::new(dummy_impl::DummyBackend::new()),
    }
}

// ==========================================
// Realtime Main Class
// ==========================================

/// Real-time audio I/O class.
///
/// Provides a common API for real-time audio input/output across multiple
/// platforms. This is a direct translation of the C++ RTAudio class API,
/// backed by a real per-platform audio driver (an internal `Backend` trait
/// implementation - `CoreAudioBackend`, `WasapiBackend`, or `AlsaBackend`).
///
/// # Thread Safety
///
/// The audio callback runs in a separate high-priority thread owned by the
/// platform's audio driver. Use thread-safe types (like `Arc<Mutex<T>>` or
/// the library's `Buffer` types) to share state between the callback and
/// the main thread.
///
/// # Example
///
/// ```ignore
/// use mkaudiolibrary::realtime::{Realtime, Api};
///
/// // Create with default API
/// let audio = Realtime::new(None).unwrap();
///
/// // List available devices
/// for id in audio.get_device_ids() {
///     if let Ok(info) = audio.get_device_info(id) {
///         println!("{}: {} (in:{}, out:{})",
///             info.id, info.name,
///             info.input_channels, info.output_channels);
///     }
/// }
/// ```
pub struct Realtime {
    api: Api,
    backend: Box<dyn Backend>,
    show_warnings: bool,
}

impl Realtime {
    /// Create a new Realtime instance.
    ///
    /// # Arguments
    /// * `api` - Desired audio API (None for auto-detection)
    ///
    /// # Returns
    /// `Ok(Realtime)` on success, or an error if no suitable API is found.
    ///
    /// # Example
    /// ```ignore
    /// use mkaudiolibrary::realtime::{Realtime, Api};
    ///
    /// // Auto-detect best API
    /// let audio = Realtime::new(None).unwrap();
    ///
    /// // Or specify an API
    /// let audio = Realtime::new(Some(Api::CoreAudio)).unwrap();
    /// ```
    pub fn new(api: Option<Api>) -> MKAudioResult<Self> {
        let selected_api = api.unwrap_or_else(Self::detect_api);

        Ok(Self {
            api: selected_api,
            backend: make_backend(selected_api),
            show_warnings: true,
        })
    }

    /// Get the current audio API in use.
    pub fn get_current_api(&self) -> Api {
        self.api
    }

    /// Get list of compiled APIs available on this system.
    pub fn get_compiled_apis() -> Vec<Api> {
        let mut apis = vec![Api::Dummy];

        #[cfg(target_os = "macos")]
        apis.push(Api::CoreAudio);

        #[cfg(target_os = "windows")]
        apis.push(Api::Wasapi);

        #[cfg(target_os = "linux")]
        apis.push(Api::Alsa);

        apis
    }

    /// Detect the best available API for this platform.
    fn detect_api() -> Api {
        #[cfg(target_os = "macos")]
        return Api::CoreAudio;

        #[cfg(target_os = "windows")]
        return Api::Wasapi;

        #[cfg(target_os = "linux")]
        return Api::Alsa;

        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        return Api::Dummy;
    }

    /// Get the number of audio devices available.
    pub fn get_device_count(&self) -> usize {
        self.backend.device_ids().len()
    }

    /// Get a list of audio device identifiers.
    pub fn get_device_ids(&self) -> Vec<usize> {
        self.backend.device_ids()
    }

    /// Get a list of audio device names.
    pub fn get_device_names(&self) -> Vec<String> {
        self.backend
            .device_ids()
            .iter()
            .filter_map(|&id| self.backend.device_info(id).ok())
            .map(|info| info.name)
            .collect()
    }

    /// Get information about a specific device.
    ///
    /// # Arguments
    /// * `device_id` - Device identifier from `get_device_ids()`
    pub fn get_device_info(&self, device_id: usize) -> MKAudioResult<DeviceInfo> {
        self.backend.device_info(device_id)
    }

    /// Get the default output device ID.
    pub fn get_default_output_device(&self) -> usize {
        self.backend.default_output_device()
    }

    /// Get the default input device ID.
    pub fn get_default_input_device(&self) -> usize {
        self.backend.default_input_device()
    }

    /// Open an audio stream.
    ///
    /// # Arguments
    /// * `output_params` - Output stream parameters (None for input-only)
    /// * `input_params` - Input stream parameters (None for output-only)
    /// * `sample_rate` - Desired sample rate in Hz
    /// * `buffer_frames` - Desired buffer size in frames (may be adjusted)
    /// * `callback` - Audio processing callback function
    /// * `options` - Optional stream configuration
    ///
    /// # Returns
    /// The actual buffer size used (may differ from requested).
    pub fn open_stream(
        &mut self,
        output_params: Option<&StreamParameters>,
        input_params: Option<&StreamParameters>,
        sample_rate: usize,
        buffer_frames: usize,
        callback: AudioCallback,
        options: Option<StreamOptions>,
    ) -> MKAudioResult<usize> {
        if output_params.is_none() && input_params.is_none() {
            return Err(MKAudioError::InvalidParameter(
                "At least one of output or input parameters must be specified".into(),
            ));
        }

        let options = options.unwrap_or_default();
        self.backend.open_stream(
            output_params,
            input_params,
            sample_rate,
            buffer_frames,
            callback,
            &options,
        )
    }

    /// Close the audio stream.
    pub fn close_stream(&mut self) {
        self.backend.close();
    }

    /// Start the audio stream.
    pub fn start_stream(&mut self) -> MKAudioResult<()> {
        self.backend.start()
    }

    /// Stop the audio stream.
    pub fn stop_stream(&mut self) -> MKAudioResult<()> {
        self.backend.stop()
    }

    /// Abort the audio stream (immediate stop without draining).
    pub fn abort_stream(&mut self) -> MKAudioResult<()> {
        self.backend.stop()
    }

    /// Check if a stream is running.
    pub fn is_stream_running(&self) -> bool {
        self.backend.is_running()
    }

    /// Get the stream time in seconds.
    pub fn get_stream_time(&self) -> f32 {
        self.backend.stream_time()
    }

    /// Get the stream latency in samples.
    pub fn get_stream_latency(&self) -> usize {
        self.backend.latency_samples()
    }

    /// Enable or disable warning messages.
    pub fn show_warnings(&mut self, show: bool) {
        self.show_warnings = show;
    }
}

impl Drop for Realtime {
    fn drop(&mut self) {
        self.backend.close();
    }
}

// ==========================================
// Buffer Integration Helpers
// ==========================================

/// Convert interleaved samples to separate channel buffers.
///
/// # Arguments
/// * `interleaved` - Interleaved sample data
/// * `channels` - Number of channels
/// * `frames` - Number of frames per channel
///
/// # Returns
/// Vector of `Buffer<f32>`, one per channel.
pub fn deinterleave(interleaved: &[f32], channels: usize, frames: usize) -> Vec<Buffer<f32>> {
    let mut buffers = Vec::with_capacity(channels);
    for ch in 0..channels {
        let mut buffer = Buffer::new(frames);
        for frame in 0..frames {
            buffer[frame] = interleaved[frame * channels + ch];
        }
        buffers.push(buffer);
    }
    buffers
}

/// Convert separate channel buffers to interleaved samples.
///
/// # Arguments
/// * `buffers` - Vector of channel buffers
/// * `interleaved` - Output interleaved buffer to fill
/// * `frames` - Number of frames per channel
pub fn interleave(buffers: &[Buffer<f32>], interleaved: &mut [f32], frames: usize) {
    let channels = buffers.len();
    for (ch, buffer) in buffers.iter().enumerate() {
        for frame in 0..frames {
            interleaved[frame * channels + ch] = buffer[frame];
        }
    }
}

/// Create a stereo callback wrapper that works with separate L/R buffers.
///
/// Simplifies processing when you want to work with individual channel buffers
/// rather than interleaved data.
///
/// # Arguments
/// * `processor` - Function that receives (left_in, right_in, left_out, right_out, frames)
///
/// # Returns
/// An `AudioCallback` suitable for use with `Realtime::open_stream()`.
pub fn stereo_callback<F>(mut processor: F) -> AudioCallback
where
    F: FnMut(&[f32], &[f32], &mut [f32], &mut [f32], usize) + Send + 'static,
{
    Box::new(move |output, input, frames, _time, _status| {
        // Deinterleave input
        let mut left_in = vec![0.0; frames];
        let mut right_in = vec![0.0; frames];
        for i in 0..frames {
            if input.len() >= (i + 1) * 2 {
                left_in[i] = input[i * 2];
                right_in[i] = input[i * 2 + 1];
            }
        }

        // Prepare output buffers
        let mut left_out = vec![0.0; frames];
        let mut right_out = vec![0.0; frames];

        // Process
        processor(&left_in, &right_in, &mut left_out, &mut right_out, frames);

        // Interleave output
        for i in 0..frames {
            if output.len() >= (i + 1) * 2 {
                output[i * 2] = left_out[i];
                output[i * 2 + 1] = right_out[i];
            }
        }

        0
    })
}

/// Shared helper: wrap a `Mutex<Option<AudioCallback>>` invocation, used by
/// every platform backend's real-time thread/proc to run the user callback
/// and interpret its return code consistently.
pub(crate) fn invoke_callback(
    callback: &Arc<Mutex<Option<AudioCallback>>>,
    running: &Arc<AtomicBool>,
    output: &mut [f32],
    input: &[f32],
    frames: usize,
    stream_time: f32,
    status: StreamStatus,
) {
    let mut guard = callback.lock().unwrap();
    if let Some(cb) = guard.as_mut() {
        let result = cb(output, input, frames, stream_time, status);
        if result != 0 {
            running.store(false, Ordering::SeqCst);
        }
    } else {
        output.fill(0.0);
    }
}
