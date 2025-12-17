//! MKAU plugin format for modular audio processing chains.
//!
//! This module provides the `Processor` trait for implementing audio plugins
//! and a loader function for dynamically loading `.mkap` plugin files.
//!
//! ## Example Plugin Implementation
//!
//! ```ignore
//! use mkaudiolibrary::buffer::Buffer;
//! use mkaudiolibrary::processor::{Processor, AudioIO};
//!
//! struct GainPlugin
//! {
//!     parameters : [(String, f64); 1],
//!     internal_buffer : Buffer<f64>
//! }
//!
//! impl GainPlugin
//! {
//!     fn new() -> Self
//!     {
//!         Self
//!         {
//!             parameters : [(String::from("Gain"), 0.5)],
//!             internal_buffer : Buffer::new(1024)
//!         }
//!     }
//! }
//!
//! impl Processor for GainPlugin
//! {
//!     fn init(&mut self) {}
//!     fn name(&self) -> String { String::from("Gain") }
//!     fn get_parameter(&self, index : usize) -> f64 { self.parameters[index].1 }
//!     fn set_parameter(&mut self, index : usize, value : f64) { self.parameters[index].1 = value; }
//!     fn get_parameter_name(&self, index : usize) -> String { self.parameters[index].0.clone() }
//!
//!     #[cfg(feature = "gui")]
//!     fn get_view(&self) -> Option<&View> { None }
//!     #[cfg(feature = "gui")]
//!     fn get_view_mut(&mut self) -> Option<&mut View> { None }
//!
//!     fn prepare_to_play(&mut self, buffer_size : usize, _sample_rate : usize)
//!     {
//!         self.internal_buffer.resize(buffer_size);
//!     }
//!
//!     fn run(&self, audio : &mut AudioIO)
//!     {
//!         let gain = self.parameters[0].1;
//!         let num_channels = audio.input.len().min(audio.output.len());
//!         for channel in 0..num_channels
//!         {
//!             let input_guard = audio.input[channel].read();
//!             let mut output_guard = audio.output[channel].write();
//!             for sample in 0..input_guard.len().min(output_guard.len())
//!             {
//!                 output_guard[sample] = input_guard[sample] * gain;
//!             }
//!         }
//!     }
//!
//!     #[cfg(feature = "midi")]
//!     fn run_with_midi(&self, audio : &mut AudioIO, _midi : &mut MidiIO)
//!     {
//!         self.run(audio);
//!     }
//! }
//!
//! mkaudiolibrary::declare_plugin!(GainPlugin, GainPlugin::new);
//! ```
//!
//! ## MIDI Processing Example (requires `midi` feature)
//!
//! ```ignore
//! #[cfg(feature = "midi")]
//! use mkaudiolibrary::processor::{Processor, AudioIO, MidiIO};
//!
//! #[cfg(feature = "midi")]
//! fn process_with_midi(processor: &dyn Processor, audio: &mut AudioIO, midi: &mut MidiIO)
//! {
//!     // Process incoming MIDI messages
//!     for msg in &midi.input
//!     {
//!         // Handle MIDI messages (note on/off, CC, etc.)
//!     }
//!
//!     // Run audio processing
//!     processor.run(audio);
//!
//!     // Optionally generate MIDI output
//!     // midi.output.push(MidiMessage::NoteOn { channel: 0, key: 60, velocity: 100 });
//! }
//! ```
//!
//! ## GUI Example (requires `gui` feature)
//!
//! ```ignore
//! #[cfg(feature = "gui")]
//! use mkaudiolibrary::processor::{Processor, AudioIO, View, Window, WindowBuilder, Extent};
//!
//! #[cfg(feature = "gui")]
//! fn open_plugin_window(processor: &dyn Processor)
//! {
//!     if let Some(view) = processor.get_view()
//!     {
//!         let size = processor.get_preferred_size();
//!         let window = WindowBuilder::new(processor.name().as_str(), size).build();
//!         window.show();
//!     }
//! }
//! ```
//!
//! ## Loading Plugins
//!
//! ```ignore
//! use mkaudiolibrary::processor::load;
//!
//! // Load a plugin from /path/to/plugins/myplugin.mkap
//! let plugin = load("/path/to/plugins", "myplugin").expect("Failed to load plugin");
//! println!("Loaded: {}", plugin.name());
//! ```

extern crate libloading;
use libloading::{Library, Symbol};

use crate::buffer::Buffer;

#[cfg(feature = "midi")]
pub use mkmidilibrary::midi::MidiMessage;

#[cfg(feature = "gui")]
pub use mkgraphic::prelude::{View, Window, Extent, Point};
#[cfg(feature = "gui")]
pub use mkgraphic::host::WindowBuilder;

/// Audio I/O container for buffer-based processing.
///
/// Provides access to audio input/output buffers and optional sidechain buffers.
/// Each buffer is a `Buffer<f64>` that can be safely shared across threads.
///
/// # Example
/// ```ignore
/// fn process(audio: &mut AudioIO)
/// {
///     for ch in 0..audio.input.len().min(audio.output.len())
///     {
///         let input = audio.input[ch].read();
///         let mut output = audio.output[ch].write();
///         for i in 0..input.len()
///         {
///             output[i] = input[i];
///         }
///     }
/// }
/// ```

pub enum ChannelLayout
{
    Mono,
    Stereo,
    LCR,
    Quad,
    Surround5p1,
    Surround7p1,
    Surround7p1p2,
    Surround7p1p4
}
impl ChannelLayout
{
    pub fn num_channels(&self) -> usize
    {
        match self
        {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::LCR => 3,
            ChannelLayout::Quad =>4,
            ChannelLayout::Surround5p1 => 6,
            ChannelLayout::Surround7p1 => 8,
            ChannelLayout::Surround7p1p2 => 10,
            ChannelLayout::Surround7p1p4 => 12,
        }
    }
}

pub struct AudioIO
{
    /// Input audio buffers (one per channel).
    pub input : Vec<Buffer<f64>>,
    /// Output audio buffers (one per channel).
    pub output : Vec<Buffer<f64>>,
    /// Sidechain input buffers (optional, one per channel).
    pub sidechain_in : Vec<Buffer<f64>>,
    /// Sidechain output buffers (optional, one per channel).
    pub sidechain_out : Vec<Buffer<f64>>
}

impl AudioIO
{
    /// Create a new AudioIO with the specified channel counts and buffer size.
    ///
    /// # Arguments
    /// * `input_channels` - Number of input channels
    /// * `output_channels` - Number of output channels
    /// * `sidechain_in_channels` - Number of sidechain input channels
    /// * `sidechain_out_channels` - Number of sidechain output channels
    /// * `buffer_size` - Size of each buffer in samples
    pub fn new(input_channels : usize, output_channels : usize,
               sidechain_in_channels : usize, sidechain_out_channels : usize,
               buffer_size : usize) -> Self
    {
        Self
        {
            input : (0..input_channels).map(|_| Buffer::new(buffer_size)).collect(),
            output : (0..output_channels).map(|_| Buffer::new(buffer_size)).collect(),
            sidechain_in : (0..sidechain_in_channels).map(|_| Buffer::new(buffer_size)).collect(),
            sidechain_out : (0..sidechain_out_channels).map(|_| Buffer::new(buffer_size)).collect()
        }
    }

    /// Create an AudioIO with layout.
    pub fn set_channel(layout : ChannelLayout, buffer_size : usize) -> Self
    {
        Self::new(layout.num_channels(), layout.num_channels(), 0, 0, buffer_size)
    }

    /// Resize all buffers to a new size.
    pub fn resize(&mut self, buffer_size : usize)
    {
        for buf in &self.input { buf.resize(buffer_size); }
        for buf in &self.output { buf.resize(buffer_size); }
        for buf in &self.sidechain_in { buf.resize(buffer_size); }
        for buf in &self.sidechain_out { buf.resize(buffer_size); }
    }
}

impl Default for AudioIO
{
    fn default() -> Self { Self::set_channel(ChannelLayout::Stereo, 1024) }
}

/// MIDI I/O container for MIDI message processing.
///
/// Provides input and output vectors for MIDI messages. The input contains
/// messages received during the current processing block, and output is
/// for messages to be sent after processing.
///
/// Only available with the `midi` feature enabled.
///
/// # Example
/// ```ignore
/// #[cfg(feature = "midi")]
/// fn process_midi(midi: &mut MidiIO)
/// {
///     for msg in &midi.input
///     {
///         match msg
///         {
///             MidiMessage::NoteOn { channel, key, velocity } =>
///             {
///                 // Handle note on
///             }
///             MidiMessage::ControlChange { channel, controller, value } =>
///             {
///                 // Handle CC
///             }
///             _ => {}
///         }
///     }
///     // Clear input after processing
///     midi.input.clear();
/// }
/// ```
#[cfg(feature = "midi")]
pub struct MidiIO
{
    /// Incoming MIDI messages for the current processing block.
    pub input : Box<[Option<MidiMessage>]>,
    /// Outgoing MIDI messages to be sent after processing.
    pub output : Box<[Option<MidiMessage>]>
}

#[cfg(feature = "midi")]
impl MidiIO
{
    /// Create a new empty MidiIO.
    pub fn new(buffer_size : usize) -> Self
    {
        Self
        {
            input : vec![None; buffer_size].into_boxed_slice(),
            output : vec![None; buffer_size].into_boxed_slice(),
        }
    }
    pub fn resize(&mut self, buffer_size : usize)
    {
        self.input = vec![None; buffer_size].into_boxed_slice();
        self.output = vec![None; buffer_size].into_boxed_slice();
    }
}

#[cfg(feature = "midi")]
impl Default for MidiIO
{
    fn default() -> Self { Self::new(1024) }
}

/// Declare a plugin for dynamic loading.
///
/// This macro generates the `_create` extern function required for
/// loading the plugin as a `.mkap` dynamic library.
///
/// # Arguments
/// * `$plugin_type` - The type implementing `Processor`
/// * `$constructor` - Path to the constructor function (e.g., `MyPlugin::new`)
#[macro_export]
macro_rules! declare_plugin
{
    ($plugin_type:ty, $constructor:path) =>
    {
        #[no_mangle]
        pub extern "C" fn _create() -> *mut dyn Processor
        {
            let constructor : fn() -> $plugin_type = $constructor;
            let object = constructor();
            let boxed : Box<dyn Processor> = Box::new(object);
            Box::into_raw(boxed)
        }
    };
}

/// Audio processor trait for MKAU plugins.
///
/// Implement this trait to create an audio plugin that can be loaded
/// dynamically or used directly in a processing chain.
///
/// ## Audio I/O
/// The `run` method uses `AudioIO` which contains thread-safe `Buffer<f64>` types
/// for input, output, and sidechain channels. Access buffer data using `.read()`
/// for inputs and `.write()` for outputs.
///
/// ## MIDI Support
/// When the `midi` feature is enabled, use `run_with_midi` for processors that
/// need MIDI input/output. The default implementation calls `run` and ignores MIDI.
pub trait Processor
{
    /// Initialize the processor after loading.
    /// Called once when the plugin is first loaded.
    fn init(&mut self);

    /// Get the display name of the processor.
    fn name(&self) -> String;

    /// Get the value of a parameter by index.
    /// Returns a value typically in the range 0.0 to 1.0.
    fn get_parameter(&self, index : usize) -> f64;

    /// Set the value of a parameter by index.
    fn set_parameter(&mut self, index : usize, value : f64);

    /// Get the display name of a parameter by index.
    fn get_parameter_name(&self, index : usize) -> String;

    /// Get the plugin's UI view.
    ///
    /// Only available with the `gui` feature enabled.
    /// Returns the View containing the plugin's UI elements.
    ///
    /// # Example
    /// ```ignore
    /// #[cfg(feature = "gui")]
    /// fn get_view(&self) -> Option<&View>
    /// {
    ///     self.view.as_ref()
    /// }
    /// ```
    #[cfg(feature = "gui")]
    fn get_view(&self) -> Option<&View>;

    /// Get a mutable reference to the plugin's UI view.
    ///
    /// Only available with the `gui` feature enabled.
    #[cfg(feature = "gui")]
    fn get_view_mut(&mut self) -> Option<&mut View>;

    /// Get the preferred window size for the plugin UI.
    ///
    /// Only available with the `gui` feature enabled.
    /// Returns the preferred width and height as an Extent.
    #[cfg(feature = "gui")]
    fn get_preferred_size(&self) -> Extent
    {
        Extent::new(400.0, 300.0)
    }

    /// Prepare the processor for playback.
    /// Called before audio processing begins or when buffer size/sample rate changes.
    ///
    /// # Arguments
    /// * `buffer_size` - Number of samples per processing block
    /// * `sample_rate` - Audio sample rate in Hz
    fn prepare_to_play(&mut self, buffer_size : usize, sample_rate : usize);

    /// Process audio through the plugin.
    ///
    /// # Arguments
    /// * `audio` - Audio I/O container with input/output/sidechain buffers
    ///
    /// # Example
    /// ```ignore
    /// fn run(&self, audio : &mut AudioIO)
    /// {
    ///     for ch in 0..audio.input.len().min(audio.output.len())
    ///     {
    ///         let input = audio.input[ch].read();
    ///         let mut output = audio.output[ch].write();
    ///         for i in 0..input.len()
    ///         {
    ///             output[i] = input[i] * self.gain;
    ///         }
    ///     }
    /// }
    /// ```
    fn run(&self, audio : &mut AudioIO);

    /// Process audio with MIDI input/output.
    ///
    /// Only available with the `midi` feature enabled.
    /// Default implementation calls `run` and ignores MIDI data.
    ///
    /// # Arguments
    /// * `audio` - Audio I/O container with input/output/sidechain buffers
    /// * `midi` - MIDI I/O container with input/output message vectors
    #[cfg(feature = "midi")]
    fn run_with_midi(&self, audio : &mut AudioIO, midi : &mut MidiIO);
}

/// Load a plugin from a `.mkap` dynamic library file.
///
/// # Arguments
/// * `path` - Directory containing the plugin file
/// * `name` - Plugin name (without `.mkap` extension)
///
/// # Returns
/// A boxed `Processor` trait object on success, or a loading error.
///
/// # Safety
/// This function loads and executes code from an external library.
/// Only load plugins from trusted sources.
pub fn load(path : &str, name : &str) -> Result<Box<dyn Processor>, libloading::Error>
{
    unsafe
    {
        let file = format!("{}/{}.mkap", path, name);
        let lib = Library::new(&file)?;
        let constructor : Symbol<unsafe fn() -> *mut dyn Processor> = lib.get(b"_create\0")?;
        let mut plugin = Box::from_raw(constructor());
        plugin.init();
        Ok(plugin)
    }
}