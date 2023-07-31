//! ### Use
//! ```
//! use mkaudiolibrary::buffer::*;
//! use mkaudiolibrary::processor::*;
//!
//! #[derive(Debug, Default)]
//! struct Plugin
//! {
//!     parameters : [(String, f64);1],
//! }
//! impl Plugin { fn new() -> Self { return Plugin { parameters : [(parameter.to_string(), 0.5)] }; } }
//! impl Processor for Plugin
//! {
//!     fn init(& mut self) {}
//!     fn name(& self) -> String { return "Plugin".to_string(); }
//!     fn get_parameter(& self, index: usize) -> f64 { return self.parameters[index].1; }
//!     fn set_parameter(& mut self, index: usize, value: f64) { self.parameters[index].1 = value; }
//!     fn get_parameter_name(& self, index: usize) -> String { return self.parameters[index].0.clone(); }
//!     fn run(& self, input: & Box<[Buffer<f64>]>>, sidechain_in: & Buffer<f64>, output: & mut Box<[Buffer<f64>]>, sidechain_out: & mut Buffer<f64>)
//!     {
//!         for x in 0..input.len() { for y in 0..input[x].len() { input[x][y] = input[x][y] * self.parameters[0].1; } }
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
    ///Process with the plugin. Optional sidechain I/O. Buffer size of I/O must be same.
    fn run(& self, input: & Option<Box<[Buffer<f64>]>>, sidechain_in : & Buffer<f64>,
           output: & mut Box<[Buffer<f64>]>, sidechain_out : & mut Buffer<f64>);
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