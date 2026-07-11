//! MKAP backend for the unified plugin host: a thin adapter from
//! [`crate::processor::Processor`] (this library's native plugin trait) to
//! [`super::HostedPlugin`], so native MKAP plugins can be scanned/loaded
//! through the same API as VST3/AUv2 plugins.

use std::path::Path;

use crate::processor::{AudioIO, Processor};

use super::{HostError, HostResult, HostedPlugin, PluginDescriptor, PluginFormat};

pub(super) fn scan(directory: &Path) -> Vec<PluginDescriptor> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };

    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "mkap"))
        .filter_map(|e| {
            let path = e.path();
            let name = path.file_stem()?.to_str()?.to_string();
            Some(PluginDescriptor {
                format: PluginFormat::Mkap,
                name,
                vendor: String::new(),
                path: Some(path),
                category: String::new(),
                au_component: None,
                vst3_class_id: None,
            })
        })
        .collect()
}

struct MkapHosted {
    inner: Box<dyn Processor>,
}

// `Processor` implementors are expected to be usable from the audio thread
// they're driven from; MKAP plugins are always in-process Rust code loaded
// via `libloading`, so there's no cross-thread COM/ObjC ownership concern
// here (unlike the VST3/AU backends).
unsafe impl Send for MkapHosted {}

impl HostedPlugin for MkapHosted {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn vendor(&self) -> String {
        String::new()
    }
    // The `Processor` trait doesn't carry an intrinsic channel count (the
    // host chooses the `AudioIO` layout it passes to `run`), so this
    // reports the common stereo default rather than a value read from the
    // plugin itself.
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
    fn parameter_name(&self, index: usize) -> String {
        self.inner.get_parameter_name(index)
    }
    fn get_parameter(&self, index: usize) -> f64 {
        self.inner.get_parameter(index)
    }
    fn set_parameter(&mut self, index: usize, value: f64) {
        self.inner.set_parameter(index, value);
    }

    fn prepare(&mut self, sample_rate: usize, block_size: usize) -> HostResult<()> {
        self.inner.prepare_to_play(block_size, sample_rate);
        Ok(())
    }

    // MKAP's `Processor` trait has no explicit activate/deactivate hook.
    fn set_active(&mut self, _active: bool) -> HostResult<()> {
        Ok(())
    }

    fn process(&mut self, audio: &mut AudioIO) {
        self.inner.run(audio);
    }
}

pub(super) fn load(descriptor: &PluginDescriptor) -> HostResult<Box<dyn HostedPlugin>> {
    let path = descriptor
        .path
        .as_ref()
        .ok_or_else(|| HostError::LoadFailed("MKAP descriptor is missing a path".into()))?;
    let dir = path.parent().and_then(|p| p.to_str()).unwrap_or(".");
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| HostError::LoadFailed(format!("invalid MKAP path: {}", path.display())))?;

    let processor =
        crate::processor::load(dir, name).map_err(|e| HostError::LoadFailed(e.to_string()))?;
    Ok(Box::new(MkapHosted { inner: processor }))
}
