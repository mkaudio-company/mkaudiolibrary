//! MKAU plugin format for modular audio processing chains.
//!
//! This module provides the `Processor` trait for implementing audio plugins
//! and a loader function for dynamically loading `.mkap` plugin files.
//!
//! ## Example Plugin Implementation
//!
//! ```ignore
//! use mkaudiolibrary::buffer::Buffer;
//! use mkaudiolibrary::processor::Processor;
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
//!     fn open_window(&self) {}
//!     fn close_window(&self) {}
//!
//!     fn prepare_to_play(&mut self, buffer_size : usize, _sample_rate : usize)
//!     {
//!         self.internal_buffer.resize(buffer_size);
//!     }
//!
//!     fn run(&self, input : &[&[f64]], _sidechain_in : &[&[f64]],
//!            output : &mut [&mut [f64]], _sidechain_out : &mut [&mut [f64]])
//!     {
//!         let gain = self.parameters[0].1;
//!         for channel in 0..input.len()
//!         {
//!             for sample in 0..input[channel].len()
//!             {
//!                 output[channel][sample] = input[channel][sample] * gain;
//!             }
//!         }
//!     }
//! }
//!
//! mkaudiolibrary::declare_plugin!(GainPlugin, GainPlugin::new);
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

    /// Open the plugin's UI window.
    fn open_window(&self);

    /// Close the plugin's UI window.
    fn close_window(&self);

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
    /// * `input` - Input audio channels (slice of channel slices)
    /// * `sidechain_in` - Sidechain input channels (optional)
    /// * `output` - Output audio channels (mutable)
    /// * `sidechain_out` - Sidechain output channels (optional)
    ///
    /// All buffers must have the same sample count per channel.
    fn run(&self, input : &[&[f64]], sidechain_in : &[&[f64]],
           output : &mut [&mut [f64]], sidechain_out : &mut [&mut [f64]]);
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