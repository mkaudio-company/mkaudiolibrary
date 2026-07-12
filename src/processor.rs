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
//!     parameters : [(String, f32); 1],
//!     internal_buffer : Buffer<f32>
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
//!     fn get_parameter(&self, index : usize) -> f32 { self.parameters[index].1 }
//!     fn set_parameter(&mut self, index : usize, value : f32) { self.parameters[index].1 = value; }
//!     fn get_parameter_name(&self, index : usize) -> String { self.parameters[index].0.clone() }
//!
//!     #[cfg(feature = "gui")]
//!     fn editor(&mut self) -> Option<&mut dyn PluginEditor> { None }
//!
//!     fn prepare_to_play(&mut self, buffer_size : usize, _sample_rate : usize)
//!     {
//!         self.internal_buffer.resize(buffer_size);
//!     }
//!
//!     fn run(&self, audio : &mut AudioIO)
//!     {
//!         let gain = self.parameters[0].1;
//!         let Some(input) = audio.input else { return };
//!         let num_channels = input.len().min(audio.output.len());
//!         for channel in 0..num_channels
//!         {
//!             let input = input[channel];
//!             let output = &mut audio.output[channel];
//!             for sample in 0..input.len().min(output.len())
//!             {
//!                 output[sample] = input[sample] * gain;
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
//!     for msg in midi.input.iter().flatten()
//!     {
//!         // Handle MIDI messages (note on/off, CC, etc.)
//!     }
//!
//!     // Run audio processing
//!     processor.run(audio);
//!
//!     // Optionally generate MIDI output, if this plugin has any
//!     // if let Some(output) = midi.output.as_deref_mut() {
//!     //     output[0] = Some(MidiMessage::NoteOn { channel: 0, key: 60, velocity: 100 });
//!     // }
//! }
//! ```
//!
//! ## Instrument Plugins
//!
//! A synth/generator has no audio input - only MIDI in and audio out.
//! Override `num_inputs` to declare that, so hosts (VST3/AU/MKAP alike)
//! don't connect an input bus that will never be read; `audio.input` will
//! then be `None` in `run`/`run_with_midi`:
//!
//! ```ignore
//! impl Processor for MySynth
//! {
//!     fn num_inputs(&self) -> usize { 0 }
//!     fn num_outputs(&self) -> usize { 2 }
//!
//!     fn run(&self, audio : &mut AudioIO)
//!     {
//!         debug_assert!(audio.input.is_none());
//!         // ...generate audio into audio.output...
//!     }
//! }
//! ```
//!
//! ## GUI Example (requires `gui` feature)
//!
//! `Processor::editor()` returns a [`PluginEditor`] - a plugin-embedding
//! editor (widget tree, geometry, and parent-window-handle lifecycle from
//! [mkapk](https://github.com/mkaudio-company/mkapk)), not a standalone
//! application window. The host embeds it into its own parent window:
//!
//! ```ignore
//! #[cfg(feature = "gui")]
//! use mkaudiolibrary::processor::{Processor, AudioIO, ParentWindowHandle};
//!
//! #[cfg(feature = "gui")]
//! fn open_plugin_editor(processor: &mut dyn Processor, parent: ParentWindowHandle, host: &dyn mkaudiolibrary::processor::EditorHost)
//! {
//!     if let Some(editor) = processor.editor()
//!     {
//!         let constraints = editor.size_constraints();
//!         editor.open(parent, host);
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

#[cfg(feature = "midi")]
pub use mkmidilibrary::midi::MidiMessage;

#[cfg(feature = "gui")]
pub use mkapk_core::{Point, Pointf, Size, Sizef};
#[cfg(feature = "gui")]
pub use mkapk_host::editor::{EditorHost, ParentWindowHandle, PluginEditor, SizeConstraints};

/// Standard speaker channel layouts - a convenient way to size the sample
/// storage a caller allocates before borrowing an [`AudioIO`] view into it,
/// without spelling out a raw channel count.
///
/// # Example
/// ```ignore
/// let channels = ChannelLayout::Stereo.num_channels();
/// let storage = vec![vec![0.0f32; 512]; channels];
/// ```
pub enum ChannelLayout {
    /// 1 channel.
    Mono,
    /// 2 channels: left, right.
    Stereo,
    /// 3 channels: left, center, right.
    LCR,
    /// 4 channels: front left, front right, rear left, rear right.
    Quad,
    /// 6 channels: 5.1 surround (L, R, C, LFE, rear L, rear R).
    Surround5p1,
    /// 8 channels: 7.1 surround (5.1 plus side L/R).
    Surround7p1,
    /// 10 channels: 7.1.2 surround (7.1 plus two height channels).
    Surround7p1p2,
    /// 12 channels: 7.1.4 surround (7.1 plus four height channels).
    Surround7p1p4,
}
impl ChannelLayout {
    /// Number of discrete audio channels in this layout.
    pub fn num_channels(&self) -> usize {
        match self {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::LCR => 3,
            ChannelLayout::Quad => 4,
            ChannelLayout::Surround5p1 => 6,
            ChannelLayout::Surround7p1 => 8,
            ChannelLayout::Surround7p1p2 => 10,
            ChannelLayout::Surround7p1p4 => 12,
        }
    }
}

/// Audio I/O view for buffer-based processing.
///
/// A thin, non-owning wrapper over per-channel sample slices - input and
/// sidechain-input channels are borrowed immutably, output and
/// sidechain-output channels mutably. `AudioIO` doesn't allocate or own any
/// sample storage itself: the caller (typically a host driving
/// [`Processor::run`] once per audio callback) owns the actual buffers for
/// the life of the stream and constructs a new `AudioIO` borrowing into
/// them for each block, matching how real audio APIs (VST3's
/// `AudioBusBuffers`, CoreAudio's `AudioBufferList`) hand a plugin raw
/// pointers into host-owned memory rather than copying into a buffer the
/// plugin owns.
///
/// `input`, `sidechain_in`, and `sidechain_out` are `Option`: a generator
/// plugin (synth, noise source, ...) may have no audio input at all, and
/// sidechain busses are commonly absent. `output` is always present -
/// every processor produces *some* output, even if it's silence.
///
/// # Example
/// ```ignore
/// fn process(audio: &mut AudioIO)
/// {
///     let Some(input) = audio.input else { return };
///     for ch in 0..input.len().min(audio.output.len())
///     {
///         let (input, output) = (input[ch], &mut audio.output[ch]);
///         for i in 0..input.len().min(output.len())
///         {
///             output[i] = input[i];
///         }
///     }
/// }
///
/// // Building one for a call, from caller-owned storage:
/// let input_storage = vec![vec![0.0f32; 512]; 2];
/// let mut output_storage = vec![vec![0.0f32; 512]; 2];
/// let input: Vec<&[f32]> = input_storage.iter().map(Vec::as_slice).collect();
/// let mut output: Vec<&mut [f32]> = output_storage.iter_mut().map(Vec::as_mut_slice).collect();
/// let mut audio = AudioIO::new(Some(&input), &mut output, None, None);
/// process(&mut audio);
/// ```
pub struct AudioIO<'a> {
    /// Input audio channels, borrowed from caller-owned storage. `None` for
    /// a plugin with no audio input (e.g. an instrument/generator).
    pub input: Option<&'a [&'a [f32]]>,
    /// Output audio channels, borrowed mutably from caller-owned storage.
    pub output: &'a mut [&'a mut [f32]],
    /// Sidechain input channels. `None` when no sidechain bus is connected.
    pub sidechain_in: Option<&'a [&'a [f32]]>,
    /// Sidechain output channels. `None` when no sidechain bus is connected.
    pub sidechain_out: Option<&'a mut [&'a mut [f32]]>,
}

impl<'a> AudioIO<'a> {
    /// Wrap existing per-channel sample slices - no allocation.
    pub fn new(
        input: Option<&'a [&'a [f32]]>,
        output: &'a mut [&'a mut [f32]],
        sidechain_in: Option<&'a [&'a [f32]]>,
        sidechain_out: Option<&'a mut [&'a mut [f32]]>,
    ) -> Self {
        Self {
            input,
            output,
            sidechain_in,
            sidechain_out,
        }
    }
}

impl Default for AudioIO<'_> {
    /// An empty view (no channels). Mainly useful as a placeholder before
    /// the first real block is available.
    fn default() -> Self {
        Self {
            input: None,
            output: &mut [],
            sidechain_in: None,
            sidechain_out: None,
        }
    }
}

/// MIDI I/O container for MIDI message processing.
///
/// Provides input and output vectors for MIDI messages. The input contains
/// messages received during the current processing block, and output is
/// for messages to be sent after processing.
///
/// Only available with the `midi` feature enabled.
///
/// A thin, non-owning wrapper, the same way as [`AudioIO`]: the caller owns
/// the message slots for the stream's lifetime and borrows a `MidiIO` into
/// them for each block. `output` is `Option`: a plugin that only consumes
/// MIDI (e.g. a MIDI-controlled effect with no MIDI generation/thru of its
/// own) has nowhere to write outgoing messages.
///
/// # Example
/// ```ignore
/// #[cfg(feature = "midi")]
/// fn process_midi(midi: &mut MidiIO)
/// {
///     for msg in midi.input.iter().flatten()
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
///     midi.input.fill(None);
/// }
///
/// // Building one for a call, from caller-owned storage:
/// #[cfg(feature = "midi")]
/// let mut input_storage = vec![None; 512];
/// #[cfg(feature = "midi")]
/// let mut output_storage = vec![None; 512];
/// #[cfg(feature = "midi")]
/// let mut midi = MidiIO::new(&mut input_storage, Some(&mut output_storage));
/// ```
#[cfg(feature = "midi")]
pub struct MidiIO<'a> {
    /// Incoming MIDI messages for the current processing block.
    pub input: &'a mut [Option<MidiMessage>],
    /// Outgoing MIDI messages to be sent after processing. `None` for a
    /// plugin with no MIDI output.
    pub output: Option<&'a mut [Option<MidiMessage>]>,
}

#[cfg(feature = "midi")]
impl<'a> MidiIO<'a> {
    /// Wrap existing input/output message slots - no allocation.
    pub fn new(
        input: &'a mut [Option<MidiMessage>],
        output: Option<&'a mut [Option<MidiMessage>]>,
    ) -> Self {
        Self { input, output }
    }
}

#[cfg(feature = "midi")]
impl Default for MidiIO<'_> {
    /// An empty view (no message slots).
    fn default() -> Self {
        Self {
            input: &mut [],
            output: None,
        }
    }
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
macro_rules! declare_plugin {
    ($plugin_type:ty, $constructor:path) => {
        #[no_mangle]
        pub extern "C" fn _create() -> *mut dyn Processor {
            let constructor: fn() -> $plugin_type = $constructor;
            let object = constructor();
            let boxed: Box<dyn Processor> = Box::new(object);
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
/// The `run` method uses `AudioIO`, a thin non-owning view whose
/// input/output/sidechain channels are plain `&[f32]`/`&mut [f32]` slices
/// borrowed from caller-owned storage -- index into them directly
/// (`audio.input[ch]`/`audio.output[ch]`).
///
/// ## MIDI Support
/// When the `midi` feature is enabled, use `run_with_midi` for processors that
/// need MIDI input/output. The default implementation calls `run` and ignores MIDI.
pub trait Processor {
    /// Initialize the processor after loading.
    /// Called once when the plugin is first loaded.
    fn init(&mut self);

    /// Get the display name of the processor.
    fn name(&self) -> String;

    /// Get the number of automatable parameters.
    ///
    /// Defaults to 0. Override this so hosts (including
    /// [`crate::host`]'s unified plugin hosting) can enumerate parameters
    /// without guessing indices.
    fn num_parameters(&self) -> usize {
        0
    }

    /// Get the number of audio input channels this processor expects.
    ///
    /// Defaults to 2 (stereo). Override to declare a different count -
    /// instrument/generator plugins that take no audio input (only MIDI
    /// in, audio out) should return 0 so hosts (including
    /// [`crate::host`]'s unified plugin hosting) know not to allocate or
    /// connect an input bus that will never be read.
    fn num_inputs(&self) -> usize {
        2
    }

    /// Get the number of audio output channels this processor produces.
    ///
    /// Defaults to 2 (stereo). Override to declare a different count.
    fn num_outputs(&self) -> usize {
        2
    }

    /// Get the value of a parameter by index.
    /// Returns a value typically in the range 0.0 to 1.0.
    fn get_parameter(&self, index: usize) -> f32;

    /// Set the value of a parameter by index.
    fn set_parameter(&mut self, index: usize, value: f32);

    /// Get the display name of a parameter by index.
    fn get_parameter_name(&self, index: usize) -> String;

    /// Get the plugin's editor (GUI), if it has one.
    ///
    /// Only available with the `gui` feature enabled. Unlike a standalone
    /// application window, a [`PluginEditor`] is embedded into a host-owned
    /// parent window: the host calls `open(parent, host)` to embed it,
    /// `resize()`/`idle()` during its lifetime, and `close()` to tear it
    /// down (see [`PluginEditor`] and [`EditorHost`]). Its preferred size
    /// comes from `PluginEditor::size_constraints()`, not a separate method
    /// on `Processor`. Returns `None` for plugins with no GUI.
    ///
    /// # Example
    /// ```ignore
    /// #[cfg(feature = "gui")]
    /// fn editor(&mut self) -> Option<&mut dyn PluginEditor>
    /// {
    ///     self.editor.as_deref_mut()
    /// }
    /// ```
    #[cfg(feature = "gui")]
    fn editor(&mut self) -> Option<&mut dyn PluginEditor> {
        None
    }

    /// Prepare the processor for playback.
    /// Called before audio processing begins or when buffer size/sample rate changes.
    ///
    /// # Arguments
    /// * `buffer_size` - Number of samples per processing block
    /// * `sample_rate` - Audio sample rate in Hz
    fn prepare_to_play(&mut self, buffer_size: usize, sample_rate: usize);

    /// Process audio through the plugin.
    ///
    /// # Arguments
    /// * `audio` - Audio I/O container with input/output/sidechain buffers
    ///
    /// # Example
    /// ```ignore
    /// fn run(&self, audio : &mut AudioIO)
    /// {
    ///     let Some(input) = audio.input else { return };
    ///     for ch in 0..input.len().min(audio.output.len())
    ///     {
    ///         let input = input[ch];
    ///         let output = &mut audio.output[ch];
    ///         for i in 0..input.len().min(output.len())
    ///         {
    ///             output[i] = input[i] * self.gain;
    ///         }
    ///     }
    /// }
    /// ```
    fn run(&self, audio: &mut AudioIO<'_>);

    /// Process audio with MIDI input/output.
    ///
    /// Only available with the `midi` feature enabled.
    /// Default implementation calls `run` and ignores MIDI data.
    ///
    /// # Arguments
    /// * `audio` - Audio I/O container with input/output/sidechain buffers
    /// * `midi` - MIDI I/O container with input/output message vectors
    #[cfg(feature = "midi")]
    fn run_with_midi(&self, audio: &mut AudioIO<'_>, midi: &mut MidiIO<'_>);
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
pub fn load(path: &str, name: &str) -> Result<Box<dyn Processor>, libloading::Error> {
    unsafe {
        let file = format!("{}/{}.mkap", path, name);
        let lib = Library::new(&file)?;
        let constructor: Symbol<unsafe fn() -> *mut dyn Processor> = lib.get(b"_create\0")?;
        let mut plugin = Box::from_raw(constructor());
        plugin.init();
        Ok(plugin)
    }
}
