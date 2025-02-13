//! ### Use
//! ```
//! use mkaudiolibrary::buffer::*;
//! use mkaudiolibrary::processor::*;
//! 
//! struct UI;
//! 
//! struct Plugin
//! {
//!     ui : UI,
//!     parameters : [(String, f64);1],
//!     internal_buffer : Buffer<f64>
//! }
//! impl Plugin { fn new() -> Self { Self { ui : todo!(), parameters : [(format!("parameter"), 0.5)], internal_buffer : Buffer::new(1024).unwrap() } } }
//! impl Processor for Plugin
//! {
//!     fn init(& mut self) { todo!() }
//!     fn name(& self) -> String { format!("Plugin") }
//!     fn get_parameter(& self, index: usize) -> f64 { self.parameters[index].1 }
//!     fn set_parameter(& mut self, index: usize, value: f64) { self.parameters[index].1 = value; }
//!     fn get_parameter_name(& self, index: usize) -> String { self.parameters[index].0.clone() }
//!     fn open_window(&self) { todo!() }
//!     fn close_window(&self) { todo!() }
//!     fn prepare_to_play(&mut self, buffer_size : usize, sample_rate : usize) { self.internal_buffer.resize(buffer_size).unwrap(); }
//!     fn run(& self, input: &[&[f64]], sidechain_in: &[&[f64]], output: &mut [ &mut [f64]], sidechain_out: &mut [ &mut [f64]])
//!     {
//!         for channel in 0..input.len() { for sample in 0..input[channel].len() { output[channel][sample] = input[channel][sample] * self.parameters[0].1; } }
//!     }
//! }
//! mkaudiolibrary::declare_plugin!(Plugin, Plugin::new);
//! ```

extern crate libloading;
use libloading::{Library, Symbol};

/// Declare plugin.
#[macro_export]
macro_rules! declare_plugin
{
    ($plugin_type:ty, $constructor:path) =>
    {
        #[no_mangle]
        pub extern "C" fn _create() -> * mut dyn Processor
        {
            let constructor : fn() -> $plugin_type = $constructor;
            let object = constructor();
            let boxed : Box<dyn Processor> = Box::new(object);
            return Box::into_raw(boxed);
        }
    };
}
pub trait Processor
{
    ///Initialize processor when loaded.
    fn init(& mut self);
    ///Get name.
    fn name(& self) -> String;
    ///Get the value of the parameter of the index.
    fn get_parameter(& self, index : usize) -> f64;
    ///Set the value of the parameter of the index.
    fn set_parameter(& mut self, index : usize, value : f64);
    ///Get the name of the parameter of the index.
    fn get_parameter_name(& self, index : usize) -> String;
    ///Open the view of the processor.
    fn open_window(&self);
    ///Close the view of the processor.
    fn close_window(&self);
    ///Prepare internal methods for play.
    fn prepare_to_play(&mut self, buffer_size : usize, sample_rate : usize);
    ///Process with the plugin. Optional sidechain I/O. Buffer size of I/O must be same.
    fn run(& self, input: &[&[f64]], sidechain_in : &[&[f64]],
           output: &mut [&mut [f64]], sidechain_out : &mut [&mut [f64]]);
}
///Loads plugin.
pub fn load(path : &str, name : &str) -> Result<Box<dyn Processor>, libloading::Error>
{
    unsafe
    {
        let file = format!("{}/{}.mkap", path, name);
        let lib = Library::new(&file)?;
        let constructor : Symbol<unsafe fn() -> * mut dyn Processor> = lib.get(b"_create\0")?;
        let mut plugin = Box::from_raw(constructor());
        plugin.init();
        Ok(plugin)
    }
}