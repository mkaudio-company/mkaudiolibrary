//! ### Use
//! ```
//! use slint;
//! 
//! use mkaudiolibrary::buffer::*;
//! use mkaudiolibrary::processor::*;
//!
//! //We recommend slint for UI design.
//! slint::slint!
//! {
//!     export component UI
//!     {
//!         todo!()
//!     }
//! }
//! 
//! #[derive(Debug, Default)]
//! struct Plugin
//! {
//!     ui : UI,
//!     parameters : [(String, f64);1],
//!     internal_buffer : Buffer<f64>
//! }
//! impl Plugin { fn new() -> Self { return Plugin { ui : UI::new().unwrap(), parameters : [(parameter.to_string(), 0.5)], internal_buffer : Buffer::new(1024).unwrap() }; } }
//! impl Processor for Plugin
//! {
//!     fn init(& mut self) { self.ui.run().unwrap(); }
//!     fn name(& self) -> String { return "Plugin".to_string(); }
//!     fn get_parameter(& self, index: usize) -> f64 { return self.parameters[index].1; }
//!     fn set_parameter(& mut self, index: usize, value: f64) { self.parameters[index].1 = value; }
//!     fn get_parameter_name(& self, index: usize) -> String { return self.parameters[index].0.clone(); }
//!     fn open_window(&self) { self.ui.show().unwrap(); }
//!     fn close_window(&self) { self.ui.hide().unwrap(); }
//!     fn prepare_to_play(&mut self, buffer_size : usize, sample_rate : usize) { Buffer::resize(buffer_size).unwrap(); }
//!     fn run(& self, input: &Option<&Buffer<f64>>, sidechain_in: &Option<&mut Buffer<f64>>, output: &mut Buffer<f64>, sidechain_out: &mut Option<&mut Buffer<f64>>)
//!     {
//!         if input.is_none()
//!         {
//!             return;
//!         }
//!         let input = input.as_ref();
//!         for sample in 0..input.len() { output[sample] = input[sample] * self.parameters[0].1; }
//!     }
//! }
//! mkaudiolibrary::declare_plugin!(Plugin, Plugin::new());
//! ```

extern crate libloading;

use std::ops::Add;
use libloading::{Library, Symbol};
use crate::buffer::Buffer;

/// Declare plugin.
#[macro_export]
macro_rules! declare_plugin
{
    ($plugin_type:ty, $constructor:path) =>
    {
        #[no_mangle]
        pub extern "C" fn _create() -> * mut dyn Processor
        {
            let constructor : fn() -> plugin_type = constructor;
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
    ///Returns name.
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
    fn run(& self, input: &Option<&Buffer<f64>>, sidechain_in : &Option<&mut Buffer<f64>>,
           output: &mut Buffer<f64>, sidechain_out : &mut Option<&mut Buffer<f64>>);
}
///Loads plugin.
pub fn load(filename : String) -> Result<Box<dyn Processor>, Box<dyn std::error::Error>>
{
    unsafe
        {
            let lib = Library::new(filename.add(".mkap").as_str())?;
            let constructor : Symbol<unsafe extern fn() -> * mut dyn Processor> = lib.get(b"_create\0")?;
            let mut plugin = Box::from_raw(constructor());
            plugin.init();
            Ok(plugin)
        }
}