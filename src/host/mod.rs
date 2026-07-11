//! Unified plugin hosting for MKAP, VST3, and AUv2.
//!
//! This module lets `mkaudiolibrary` load and run *third-party* plugins
//! (as opposed to [`crate::processor`], which defines the MKAP format for
//! plugins built *with* this library) through one common `HostedPlugin`
//! trait, regardless of the underlying binary format:
//!
//! - **MKAP** - always available, thin wrapper over [`crate::processor::load`].
//! - **VST3** - `vst3` feature. Cross-platform; talks directly to a plugin's
//!   `IComponent`/`IAudioProcessor` COM-style interfaces (the same ABI any
//!   VST3 host uses), without vendoring Steinberg's SDK headers.
//! - **AUv2** - `au` feature, macOS only. Talks to the system's
//!   AudioComponent registry via CoreAudio/AudioToolbox.
//!
//! ```ignore
//! use mkaudiolibrary::host::{scan_vst3, load};
//! use mkaudiolibrary::processor::AudioIO;
//! use std::path::Path;
//!
//! let found = scan_vst3(Path::new("/Library/Audio/Plug-Ins/VST3"));
//! let mut plugin = load(&found[0])?;
//! plugin.prepare(48000, 512)?;
//! plugin.set_active(true)?;
//!
//! let mut audio = AudioIO::new(2, 2, 0, 0, 512);
//! plugin.process(&mut audio);
//! # Ok::<(), mkaudiolibrary::host::HostError>(())
//! ```

mod mkap;

#[cfg(feature = "vst3")]
pub mod vst3;

#[cfg(all(feature = "au", target_os = "macos"))]
pub mod au;

use std::path::{Path, PathBuf};

use crate::processor::AudioIO;

/// Plugin binary format hosted through [`HostedPlugin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PluginFormat {
    /// This library's own MKAP format (see [`crate::processor`]).
    Mkap,
    /// Steinberg VST3.
    Vst3,
    /// Apple Audio Unit (version 2), macOS only.
    Au,
}

impl std::fmt::Display for PluginFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginFormat::Mkap => write!(f, "MKAP"),
            PluginFormat::Vst3 => write!(f, "VST3"),
            PluginFormat::Au => write!(f, "AUv2"),
        }
    }
}

/// Errors that can occur while scanning for or loading a hosted plugin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostError {
    /// No plugin matching the descriptor could be found.
    NotFound(String),
    /// The plugin binary could not be loaded (missing file, bad bundle, dlopen failure, ...).
    LoadFailed(String),
    /// This build of `mkaudiolibrary` doesn't have the format's feature enabled.
    UnsupportedFormat(String),
    /// The plugin loaded, but failed during its own initialization/activation.
    InitializationFailed(String),
    /// A requested operation isn't supported by this plugin instance.
    Unsupported(String),
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HostError::NotFound(s) => write!(f, "plugin not found: {}", s),
            HostError::LoadFailed(s) => write!(f, "failed to load plugin: {}", s),
            HostError::UnsupportedFormat(s) => write!(f, "unsupported plugin format: {}", s),
            HostError::InitializationFailed(s) => write!(f, "plugin initialization failed: {}", s),
            HostError::Unsupported(s) => write!(f, "unsupported operation: {}", s),
        }
    }
}

impl std::error::Error for HostError {}

/// Result type for plugin hosting operations.
pub type HostResult<T> = Result<T, HostError>;

/// Identity of one discovered plugin, as returned by the `scan_*` functions
/// and consumed by [`load`].
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    /// Which backend this descriptor loads through.
    pub format: PluginFormat,
    /// Display name.
    pub name: String,
    /// Vendor/manufacturer name, when known.
    pub vendor: String,
    /// Filesystem location, for the file-based formats (MKAP, VST3).
    pub path: Option<PathBuf>,
    /// Free-form category string (e.g. "Fx|Dynamics"), when known.
    pub category: String,

    /// AUv2 component identity: (type, subtype, manufacturer) four-char codes.
    #[cfg_attr(not(feature = "au"), allow(dead_code))]
    pub(crate) au_component: Option<(u32, u32, u32)>,
    /// VST3 class ID within the module's factory.
    #[cfg_attr(not(feature = "vst3"), allow(dead_code))]
    pub(crate) vst3_class_id: Option<[u8; 16]>,
}

/// A loaded, ready-to-drive plugin instance, regardless of its underlying format.
///
/// # Realtime Safety
/// [`HostedPlugin::process`] runs on whatever thread the caller drives it
/// from (commonly the [`crate::realtime`] audio callback). Implementations
/// avoid allocating inside `process`; `prepare` is the place for any
/// block-size-dependent setup.
pub trait HostedPlugin: Send {
    /// Display name of the plugin.
    fn name(&self) -> String;
    /// Vendor/manufacturer name.
    fn vendor(&self) -> String;
    /// Number of audio input channels the plugin's main bus expects.
    fn num_inputs(&self) -> usize;
    /// Number of audio output channels the plugin's main bus produces.
    fn num_outputs(&self) -> usize;
    /// Number of automatable parameters.
    fn num_parameters(&self) -> usize;
    /// Display name of a parameter by index.
    fn parameter_name(&self, index: usize) -> String;
    /// Get a parameter's normalized value (0.0 to 1.0).
    fn get_parameter(&self, index: usize) -> f64;
    /// Set a parameter's normalized value (0.0 to 1.0).
    fn set_parameter(&mut self, index: usize, value: f64);
    /// Configure the plugin for a given sample rate and maximum block size.
    /// Must be called (at least once) before [`process`](Self::process).
    fn prepare(&mut self, sample_rate: usize, block_size: usize) -> HostResult<()>;
    /// Activate or deactivate audio processing.
    fn set_active(&mut self, active: bool) -> HostResult<()>;
    /// Process one block of audio in place.
    fn process(&mut self, audio: &mut AudioIO);
}

/// Scan a directory for MKAP plugins (`.mkap` files).
///
/// This lists candidates without loading them; construction/`init()` only
/// happens in [`load`].
pub fn scan_mkap(directory: &Path) -> Vec<PluginDescriptor> {
    mkap::scan(directory)
}

/// Load a plugin from its descriptor.
pub fn load(descriptor: &PluginDescriptor) -> HostResult<Box<dyn HostedPlugin>> {
    match descriptor.format {
        PluginFormat::Mkap => mkap::load(descriptor),

        #[cfg(feature = "vst3")]
        PluginFormat::Vst3 => vst3::load(descriptor),
        #[cfg(not(feature = "vst3"))]
        PluginFormat::Vst3 => Err(HostError::UnsupportedFormat(
            "enable the `vst3` feature to host VST3 plugins".into(),
        )),

        #[cfg(all(feature = "au", target_os = "macos"))]
        PluginFormat::Au => au::load(descriptor),
        #[cfg(not(all(feature = "au", target_os = "macos")))]
        PluginFormat::Au => Err(HostError::UnsupportedFormat(
            "enable the `au` feature (macOS only) to host Audio Units".into(),
        )),
    }
}

/// Scan a directory for VST3 plugins (`.vst3` bundles/files). Requires the `vst3` feature.
#[cfg(feature = "vst3")]
pub fn scan_vst3(directory: &Path) -> Vec<PluginDescriptor> {
    vst3::scan(directory)
}

#[cfg(not(feature = "vst3"))]
/// Scan a directory for VST3 plugins (`.vst3` bundles/files). Requires the `vst3` feature.
pub fn scan_vst3(_directory: &Path) -> Vec<PluginDescriptor> {
    Vec::new()
}

/// Scan the system for installed Audio Units. Requires the `au` feature (macOS only).
#[cfg(all(feature = "au", target_os = "macos"))]
pub fn scan_au() -> Vec<PluginDescriptor> {
    au::scan()
}

#[cfg(not(all(feature = "au", target_os = "macos")))]
/// Scan the system for installed Audio Units. Requires the `au` feature (macOS only).
pub fn scan_au() -> Vec<PluginDescriptor> {
    Vec::new()
}
