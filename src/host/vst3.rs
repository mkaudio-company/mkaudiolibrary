//! VST3 hosting backend.
//!
//! Talks directly to a VST3 module's `IPluginFactory`/`IComponent`/
//! `IAudioProcessor`/`IEditController` COM-style interfaces using hand
//! -written `repr(C)` vtables matching Steinberg's public, documented ABI
//! (verified against the real VST3 SDK headers rather than reconstructed
//! from memory), so no vendored SDK or C++ toolchain is needed to host a
//! real `.vst3` plugin - just `libloading` to dlopen the module binary and
//! raw pointer/vtable calls from there.
//!
//! # Scope
//! This is a functional, real host for the common case (one main audio
//! input bus, one main audio output bus, host-driven blocks, coarse
//! (non-sample-accurate) parameter get/set). It does **not** implement
//! sample-accurate parameter automation (`IParameterChanges`), MIDI/note
//! events, persisted plugin state, or multi-bus routing - those are real
//! gaps against the full spec, not silently-wrong shortcuts: unsupported
//! calls simply aren't made, and parameters just start at the plugin's
//! default values.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use crate::processor::AudioIO;

use super::{HostError, HostResult, HostedPlugin, PluginDescriptor, PluginFormat};

// ==========================================
// Low-level VST3 ABI types
// ==========================================

type TResult = i32;
type TUid = [u8; 16];
type TBool = u8;

const K_RESULT_OK: TResult = 0;

const K_AUDIO: i32 = 0; // MediaTypes::kAudio
const K_INPUT: i32 = 0; // BusDirections::kInput
const K_OUTPUT: i32 = 1; // BusDirections::kOutput

const K_SAMPLE_32: i32 = 0; // SymbolicSampleSizes::kSample32
const K_SAMPLE_64: i32 = 1; // SymbolicSampleSizes::kSample64

/// Steinberg's `INLINE_UID` byte-packing, exactly as `funknown.h` defines
/// it. The two platforms disagree on the encoding (`COM_COMPATIBLE` mixes
/// bytes the way a Win32 GUID does; everywhere else just packs big-endian),
/// so an interface ID computed the wrong way silently makes every
/// `queryInterface` call fail with "no such interface" - this has to match
/// exactly, verified against `pluginterfaces/base/funknown.h`.
#[cfg(target_os = "windows")]
const fn inline_uid(l1: u32, l2: u32, l3: u32, l4: u32) -> TUid {
    [
        (l1 & 0xFF) as u8,
        ((l1 >> 8) & 0xFF) as u8,
        ((l1 >> 16) & 0xFF) as u8,
        ((l1 >> 24) & 0xFF) as u8,
        ((l2 >> 16) & 0xFF) as u8,
        ((l2 >> 24) & 0xFF) as u8,
        (l2 & 0xFF) as u8,
        ((l2 >> 8) & 0xFF) as u8,
        ((l3 >> 24) & 0xFF) as u8,
        ((l3 >> 16) & 0xFF) as u8,
        ((l3 >> 8) & 0xFF) as u8,
        (l3 & 0xFF) as u8,
        ((l4 >> 24) & 0xFF) as u8,
        ((l4 >> 16) & 0xFF) as u8,
        ((l4 >> 8) & 0xFF) as u8,
        (l4 & 0xFF) as u8,
    ]
}

#[cfg(not(target_os = "windows"))]
const fn inline_uid(l1: u32, l2: u32, l3: u32, l4: u32) -> TUid {
    [
        ((l1 >> 24) & 0xFF) as u8,
        ((l1 >> 16) & 0xFF) as u8,
        ((l1 >> 8) & 0xFF) as u8,
        (l1 & 0xFF) as u8,
        ((l2 >> 24) & 0xFF) as u8,
        ((l2 >> 16) & 0xFF) as u8,
        ((l2 >> 8) & 0xFF) as u8,
        (l2 & 0xFF) as u8,
        ((l3 >> 24) & 0xFF) as u8,
        ((l3 >> 16) & 0xFF) as u8,
        ((l3 >> 8) & 0xFF) as u8,
        (l3 & 0xFF) as u8,
        ((l4 >> 24) & 0xFF) as u8,
        ((l4 >> 16) & 0xFF) as u8,
        ((l4 >> 8) & 0xFF) as u8,
        (l4 & 0xFF) as u8,
    ]
}

const IPLUGINFACTORY_IID: TUid = inline_uid(0x7A4D811C, 0x52114A1F, 0xAED9D2EE, 0x0B43BF9F);
const ICOMPONENT_IID: TUid = inline_uid(0xE831FF31, 0xF2D54301, 0x928EBBEE, 0x25697802);
const IAUDIOPROCESSOR_IID: TUid = inline_uid(0x42043F99, 0xB7DA453C, 0xA569E79D, 0x9AAEC33D);
const IEDITCONTROLLER_IID: TUid = inline_uid(0xDCD7BBE3, 0x7742448D, 0xA874AACC, 0x979C759E);

#[repr(C)]
struct FUnknownVtbl {
    query_interface:
        unsafe extern "system" fn(*mut c_void, *const TUid, *mut *mut c_void) -> TResult,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[repr(C)]
struct PFactoryInfo {
    vendor: [i8; 64],
    url: [i8; 256],
    email: [i8; 128],
    flags: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PClassInfo {
    cid: TUid,
    cardinality: i32,
    category: [i8; 32],
    name: [i8; 64],
}

#[repr(C)]
struct IPluginFactoryVtbl {
    query_interface:
        unsafe extern "system" fn(*mut c_void, *const TUid, *mut *mut c_void) -> TResult,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    get_factory_info: unsafe extern "system" fn(*mut c_void, *mut PFactoryInfo) -> TResult,
    count_classes: unsafe extern "system" fn(*mut c_void) -> i32,
    get_class_info: unsafe extern "system" fn(*mut c_void, i32, *mut PClassInfo) -> TResult,
    create_instance:
        unsafe extern "system" fn(*mut c_void, *const i8, *const i8, *mut *mut c_void) -> TResult,
}

#[repr(C)]
struct BusInfo {
    media_type: i32,
    direction: i32,
    channel_count: i32,
    name: [u16; 128],
    bus_type: i32,
    flags: u32,
}

#[repr(C)]
struct RoutingInfo {
    media_type: i32,
    bus_index: i32,
    channel: i32,
}

#[repr(C)]
struct IComponentVtbl {
    query_interface:
        unsafe extern "system" fn(*mut c_void, *const TUid, *mut *mut c_void) -> TResult,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    initialize: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    terminate: unsafe extern "system" fn(*mut c_void) -> TResult,
    get_controller_class_id: unsafe extern "system" fn(*mut c_void, *mut TUid) -> TResult,
    set_io_mode: unsafe extern "system" fn(*mut c_void, i32) -> TResult,
    get_bus_count: unsafe extern "system" fn(*mut c_void, i32, i32) -> i32,
    get_bus_info: unsafe extern "system" fn(*mut c_void, i32, i32, i32, *mut BusInfo) -> TResult,
    get_routing_info:
        unsafe extern "system" fn(*mut c_void, *mut RoutingInfo, *mut RoutingInfo) -> TResult,
    activate_bus: unsafe extern "system" fn(*mut c_void, i32, i32, i32, TBool) -> TResult,
    set_active: unsafe extern "system" fn(*mut c_void, TBool) -> TResult,
    set_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    get_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
}

#[repr(C)]
struct ProcessSetup {
    process_mode: i32,
    symbolic_sample_size: i32,
    max_samples_per_block: i32,
    sample_rate: f32,
}

#[repr(C)]
struct AudioBusBuffers {
    num_channels: i32,
    silence_flags: u64,
    channel_buffers: *mut *mut c_void,
}

#[repr(C)]
struct ProcessData {
    process_mode: i32,
    symbolic_sample_size: i32,
    num_samples: i32,
    num_inputs: i32,
    num_outputs: i32,
    inputs: *mut AudioBusBuffers,
    outputs: *mut AudioBusBuffers,
    input_parameter_changes: *mut c_void,
    output_parameter_changes: *mut c_void,
    input_events: *mut c_void,
    output_events: *mut c_void,
    process_context: *mut c_void,
}

#[repr(C)]
struct IAudioProcessorVtbl {
    query_interface:
        unsafe extern "system" fn(*mut c_void, *const TUid, *mut *mut c_void) -> TResult,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    set_bus_arrangements:
        unsafe extern "system" fn(*mut c_void, *mut u64, i32, *mut u64, i32) -> TResult,
    get_bus_arrangement: unsafe extern "system" fn(*mut c_void, i32, i32, *mut u64) -> TResult,
    can_process_sample_size: unsafe extern "system" fn(*mut c_void, i32) -> TResult,
    get_latency_samples: unsafe extern "system" fn(*mut c_void) -> u32,
    setup_processing: unsafe extern "system" fn(*mut c_void, *mut ProcessSetup) -> TResult,
    set_processing: unsafe extern "system" fn(*mut c_void, TBool) -> TResult,
    process: unsafe extern "system" fn(*mut c_void, *mut ProcessData) -> TResult,
    get_tail_samples: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[repr(C)]
struct ParameterInfo {
    id: u32,
    title: [u16; 128],
    short_title: [u16; 128],
    units: [u16; 128],
    step_count: i32,
    default_normalized_value: f32,
    unit_id: i32,
    flags: i32,
}

#[repr(C)]
struct IEditControllerVtbl {
    query_interface:
        unsafe extern "system" fn(*mut c_void, *const TUid, *mut *mut c_void) -> TResult,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    initialize: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    terminate: unsafe extern "system" fn(*mut c_void) -> TResult,
    set_component_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    set_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    get_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    get_parameter_count: unsafe extern "system" fn(*mut c_void) -> i32,
    get_parameter_info: unsafe extern "system" fn(*mut c_void, i32, *mut ParameterInfo) -> TResult,
    get_param_string_by_value:
        unsafe extern "system" fn(*mut c_void, u32, f32, *mut u16) -> TResult,
    get_param_value_by_string:
        unsafe extern "system" fn(*mut c_void, u32, *mut u16, *mut f32) -> TResult,
    normalized_param_to_plain: unsafe extern "system" fn(*mut c_void, u32, f32) -> f32,
    plain_param_to_normalized: unsafe extern "system" fn(*mut c_void, u32, f32) -> f32,
    get_param_normalized: unsafe extern "system" fn(*mut c_void, u32) -> f32,
    set_param_normalized: unsafe extern "system" fn(*mut c_void, u32, f32) -> TResult,
    set_component_handler: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    create_view: unsafe extern "system" fn(*mut c_void, *const i8) -> *mut c_void,
}

/// A reference-counted VST3 COM-style object pointer. Every interface's
/// vtable begins with the three `FUnknown` methods in the same order, so
/// `add_ref`/`release`/`query_interface` can always be called by
/// reinterpreting the vtable pointer as `*const FUnknownVtbl` regardless of
/// which specific interface this happens to be.
struct ComPtr(*mut c_void);

impl ComPtr {
    unsafe fn funknown_vtbl(&self) -> *const FUnknownVtbl {
        unsafe { *(self.0 as *const *const FUnknownVtbl) }
    }

    fn query_interface(&self, iid: &TUid) -> Option<ComPtr> {
        unsafe {
            let mut out: *mut c_void = std::ptr::null_mut();
            let vt = self.funknown_vtbl();
            let result = ((*vt).query_interface)(self.0, iid, &mut out);
            if result == K_RESULT_OK && !out.is_null() {
                Some(ComPtr(out))
            } else {
                None
            }
        }
    }
}

impl Clone for ComPtr {
    fn clone(&self) -> Self {
        unsafe {
            let vt = self.funknown_vtbl();
            ((*vt).add_ref)(self.0);
        }
        ComPtr(self.0)
    }
}

impl Drop for ComPtr {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let vt = self.funknown_vtbl();
                ((*vt).release)(self.0);
            }
        }
    }
}

fn utf16_z_to_string(buf: &[u16]) -> String {
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf16_lossy(&buf[..end])
}

fn latin1_z_to_string(buf: &[i8]) -> String {
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    buf[..end].iter().map(|&c| (c as u8) as char).collect()
}

// ==========================================
// Module loading (per platform)
// ==========================================

/// Keeps the dlopen'd library (and, on macOS, the `CFBundleRef`) alive for
/// as long as any COM object obtained from its factory is in use, and runs
/// the platform's documented module exit hook on drop.
struct LoadedModule {
    _library: libloading::Library,
    #[cfg(target_os = "macos")]
    bundle_ref: coreaudio_sys::CFBundleRef,
}

#[cfg(target_os = "macos")]
impl Drop for LoadedModule {
    fn drop(&mut self) {
        unsafe {
            if let Ok(bundle_exit) = self
                ._library
                .get::<unsafe extern "C" fn() -> bool>(b"bundleExit\0")
            {
                bundle_exit();
            }
            if !self.bundle_ref.is_null() {
                coreaudio_sys::CFRelease(self.bundle_ref as coreaudio_sys::CFTypeRef);
            }
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for LoadedModule {
    fn drop(&mut self) {
        unsafe {
            if let Ok(exit_dll) = self
                ._library
                .get::<unsafe extern "C" fn() -> bool>(b"ExitDll\0")
            {
                exit_dll();
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for LoadedModule {
    fn drop(&mut self) {
        unsafe {
            if let Ok(module_exit) = self
                ._library
                .get::<unsafe extern "C" fn() -> bool>(b"ModuleExit\0")
            {
                module_exit();
            }
        }
    }
}

// `coreaudio-sys` binds CoreFoundation's CFBundle/CFURL functions (pulled
// in transitively by the CoreAudio headers it wraps) but its build script
// only links AudioUnit/AudioToolbox/CoreAudio/OpenAL/CoreMIDI - not
// CoreFoundation itself, and this module is the only one in the `vst3`
// feature that needs it (see `macos_util.rs` for the same fix, needed
// there for the `realtime`/`au` features).
#[cfg(target_os = "macos")]
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {}

#[cfg(target_os = "macos")]
fn binary_path_for_bundle(bundle_path: &Path) -> HostResult<PathBuf> {
    let stem = bundle_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let candidate = bundle_path.join("Contents/MacOS").join(stem);
    if candidate.is_file() {
        return Ok(candidate);
    }

    let macos_dir = bundle_path.join("Contents/MacOS");
    let mut entries = std::fs::read_dir(&macos_dir)
        .map_err(|e| HostError::LoadFailed(format!("{}: {}", macos_dir.display(), e)))?;
    entries
        .find_map(|e| e.ok())
        .map(|e| e.path())
        .ok_or_else(|| HostError::LoadFailed(format!("no binary found in {}", macos_dir.display())))
}

#[cfg(target_os = "macos")]
fn open_module(bundle_path: &Path) -> HostResult<(LoadedModule, ComPtr)> {
    use coreaudio_sys::*;

    let binary_path = binary_path_for_bundle(bundle_path)?;
    let library = unsafe { libloading::Library::new(&binary_path) }
        .map_err(|e| HostError::LoadFailed(e.to_string()))?;

    let path_bytes = bundle_path.to_string_lossy().into_owned();
    let url = unsafe {
        CFURLCreateFromFileSystemRepresentation(
            kCFAllocatorDefault,
            path_bytes.as_ptr(),
            path_bytes.len() as CFIndex,
            1,
        )
    };
    if url.is_null() {
        return Err(HostError::LoadFailed(
            "failed to build CFURL for bundle path".into(),
        ));
    }
    let bundle_ref = unsafe { CFBundleCreate(kCFAllocatorDefault, url) };
    unsafe {
        CFRelease(url as CFTypeRef);
    }
    if bundle_ref.is_null() {
        return Err(HostError::LoadFailed("CFBundleCreate failed".into()));
    }

    unsafe {
        if let Ok(bundle_entry) =
            library.get::<unsafe extern "C" fn(CFBundleRef) -> bool>(b"bundleEntry\0")
            && !bundle_entry(bundle_ref)
        {
            CFRelease(bundle_ref as CFTypeRef);
            return Err(HostError::LoadFailed("bundleEntry returned false".into()));
        }
    }

    let raw_factory = unsafe {
        let factory_fn: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> =
            match library.get(b"GetPluginFactory\0") {
                Ok(f) => f,
                Err(e) => {
                    CFRelease(bundle_ref as CFTypeRef);
                    return Err(HostError::LoadFailed(e.to_string()));
                }
            };
        factory_fn()
    };
    if raw_factory.is_null() {
        return Err(HostError::LoadFailed(
            "GetPluginFactory returned null".into(),
        ));
    }

    Ok((
        LoadedModule {
            _library: library,
            bundle_ref,
        },
        ComPtr(raw_factory),
    ))
}

#[cfg(target_os = "windows")]
fn binary_path_for_module(path: &Path) -> HostResult<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let arch_dir = if cfg!(target_pointer_width = "64") {
        "x86_64-win"
    } else {
        "x86-win"
    };
    let candidate = path
        .join("Contents")
        .join(arch_dir)
        .join(format!("{}.vst3", stem));
    if candidate.is_file() {
        return Ok(candidate);
    }

    Err(HostError::LoadFailed(format!(
        "no VST3 binary found under {}",
        path.display()
    )))
}

#[cfg(target_os = "windows")]
fn open_module(path: &Path) -> HostResult<(LoadedModule, ComPtr)> {
    let binary_path = binary_path_for_module(path)?;
    let library = unsafe { libloading::Library::new(&binary_path) }
        .map_err(|e| HostError::LoadFailed(e.to_string()))?;

    unsafe {
        if let Ok(init_dll) = library.get::<unsafe extern "C" fn() -> bool>(b"InitDll\0")
            && !init_dll()
        {
            return Err(HostError::LoadFailed("InitDll returned false".into()));
        }
    }

    let raw_factory = unsafe {
        let factory_fn: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> = library
            .get(b"GetPluginFactory\0")
            .map_err(|e| HostError::LoadFailed(e.to_string()))?;
        factory_fn()
    };
    if raw_factory.is_null() {
        return Err(HostError::LoadFailed(
            "GetPluginFactory returned null".into(),
        ));
    }

    Ok((LoadedModule { _library: library }, ComPtr(raw_factory)))
}

#[cfg(target_os = "linux")]
fn binary_path_for_module(path: &Path) -> HostResult<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let candidate = path
        .join("Contents/x86_64-linux")
        .join(format!("{}.so", stem));
    if candidate.is_file() {
        return Ok(candidate);
    }
    Err(HostError::LoadFailed(format!(
        "no VST3 binary found under {}",
        path.display()
    )))
}

#[cfg(target_os = "linux")]
fn open_module(path: &Path) -> HostResult<(LoadedModule, ComPtr)> {
    let binary_path = binary_path_for_module(path)?;
    let library = unsafe { libloading::Library::new(&binary_path) }
        .map_err(|e| HostError::LoadFailed(e.to_string()))?;

    // Best-effort raw handle for ModuleEntry; some plugins never use it.
    // `into_raw`/`from_raw` live on the platform-specific `os::unix::Library`,
    // so round-trip through that to recover the handle without giving up
    // ownership (the reconstructed `Library` still closes it on drop).
    let unix_library: libloading::os::unix::Library = library.into();
    let raw_handle = unix_library.into_raw();
    let library: libloading::Library =
        unsafe { libloading::os::unix::Library::from_raw(raw_handle) }.into();

    unsafe {
        if let Ok(module_entry) =
            library.get::<unsafe extern "C" fn(*mut c_void) -> bool>(b"ModuleEntry\0")
            && !module_entry(raw_handle)
        {
            return Err(HostError::LoadFailed("ModuleEntry returned false".into()));
        }
    }

    let raw_factory = unsafe {
        let factory_fn: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> = library
            .get(b"GetPluginFactory\0")
            .map_err(|e| HostError::LoadFailed(e.to_string()))?;
        factory_fn()
    };
    if raw_factory.is_null() {
        return Err(HostError::LoadFailed(
            "GetPluginFactory returned null".into(),
        ));
    }

    Ok((LoadedModule { _library: library }, ComPtr(raw_factory)))
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn open_module(_path: &Path) -> HostResult<(LoadedModule, ComPtr)> {
    Err(HostError::UnsupportedFormat(
        "VST3 hosting isn't implemented for this platform".into(),
    ))
}

// ==========================================
// Scanning
// ==========================================

/// Scan a directory for `.vst3` bundles/files and list the classes each one's factory exports.
pub fn scan(directory: &Path) -> Vec<PluginDescriptor> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut results = Vec::new();

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "vst3") {
            continue;
        }

        let Ok((_module, factory)) = open_module(&path) else {
            continue;
        };
        let Some(factory_vtbl) = factory.query_interface(&IPLUGINFACTORY_IID) else {
            continue;
        };

        unsafe {
            let vt = *(factory_vtbl.0 as *const *const IPluginFactoryVtbl);

            let mut factory_info: PFactoryInfo = std::mem::zeroed();
            let vendor =
                if ((*vt).get_factory_info)(factory_vtbl.0, &mut factory_info) == K_RESULT_OK {
                    latin1_z_to_string(&factory_info.vendor)
                } else {
                    String::new()
                };

            let count = ((*vt).count_classes)(factory_vtbl.0);
            for index in 0..count {
                let mut info: PClassInfo = std::mem::zeroed();
                if ((*vt).get_class_info)(factory_vtbl.0, index, &mut info) != K_RESULT_OK {
                    continue;
                }

                let category = latin1_z_to_string(&info.category);
                if category != "Audio Module Class" {
                    continue;
                } // skip non-processor classes (e.g. controller-only)

                results.push(PluginDescriptor {
                    format: PluginFormat::Vst3,
                    name: latin1_z_to_string(&info.name),
                    vendor: vendor.clone(),
                    path: Some(path.clone()),
                    category,
                    au_component: None,
                    vst3_class_id: Some(info.cid),
                });
            }
        }
    }

    results
}

// ==========================================
// Hosted plugin
// ==========================================

pub(super) struct Vst3Hosted {
    _module: LoadedModule,
    component: ComPtr,
    processor: ComPtr,
    controller: Option<ComPtr>,
    controller_owned: bool,
    name: String,
    vendor: String,
    num_inputs: usize,
    num_outputs: usize,
    sample_size: i32,
    block_size: usize,
    active: bool,
    initialized_processing: bool,
    parameter_ids: Vec<u32>,
    // Only populated (and used) when `sample_size == K_SAMPLE_64`: per-channel
    // f32->f64 conversion scratch (this library's own buffers are f32; only
    // a plugin that rejects 32-bit float needs this).
    input_scratch_f64: Vec<Vec<f64>>,
    output_scratch_f64: Vec<Vec<f64>>,
}

// Every COM interaction happens through whichever thread owns this value;
// nothing here is shared concurrently, so the whole struct can move
// between threads freely even though it holds raw COM pointers.
unsafe impl Send for Vst3Hosted {}

impl Drop for Vst3Hosted {
    fn drop(&mut self) {
        unsafe {
            if self.active {
                let vt = *(self.processor.0 as *const *const IAudioProcessorVtbl);
                ((*vt).set_processing)(self.processor.0, 0);
                let cvt = *(self.component.0 as *const *const IComponentVtbl);
                ((*cvt).set_active)(self.component.0, 0);
            }
            if let (Some(controller), true) = (&self.controller, self.controller_owned) {
                let vt = *(controller.0 as *const *const IEditControllerVtbl);
                ((*vt).terminate)(controller.0);
            }
            let cvt = *(self.component.0 as *const *const IComponentVtbl);
            ((*cvt).terminate)(self.component.0);
        }
    }
}

pub(super) fn load(descriptor: &PluginDescriptor) -> HostResult<Box<dyn HostedPlugin>> {
    let path = descriptor
        .path
        .as_ref()
        .ok_or_else(|| HostError::LoadFailed("VST3 descriptor is missing a path".into()))?;
    let class_id = descriptor
        .vst3_class_id
        .ok_or_else(|| HostError::LoadFailed("VST3 descriptor is missing a class id".into()))?;

    let (module, factory) = open_module(path)?;
    let factory_iface = factory
        .query_interface(&IPLUGINFACTORY_IID)
        .ok_or_else(|| HostError::LoadFailed("module does not expose IPluginFactory".into()))?;

    let component = unsafe {
        let vt = *(factory_iface.0 as *const *const IPluginFactoryVtbl);
        let mut obj: *mut c_void = std::ptr::null_mut();
        let status = ((*vt).create_instance)(
            factory_iface.0,
            class_id.as_ptr() as *const i8,
            ICOMPONENT_IID.as_ptr() as *const i8,
            &mut obj,
        );
        if status != K_RESULT_OK || obj.is_null() {
            return Err(HostError::LoadFailed(
                "createInstance(IComponent) failed".into(),
            ));
        }
        ComPtr(obj)
    };

    unsafe {
        let vt = *(component.0 as *const *const IComponentVtbl);
        if ((*vt).initialize)(component.0, std::ptr::null_mut()) != K_RESULT_OK {
            return Err(HostError::InitializationFailed(
                "IComponent::initialize failed".into(),
            ));
        }
    }

    let processor = component
        .query_interface(&IAUDIOPROCESSOR_IID)
        .ok_or_else(|| {
            HostError::UnsupportedFormat("component does not implement IAudioProcessor".into())
        })?;

    // Try the "single component" pattern first (the IComponent object also
    // answers to IEditController); fall back to instantiating a separate
    // controller class via getControllerClassId.
    let (controller, controller_owned) = match component.query_interface(&IEDITCONTROLLER_IID) {
        Some(c) => (Some(c), false),
        None => unsafe {
            let vt = *(component.0 as *const *const IComponentVtbl);
            let mut controller_cid: TUid = [0; 16];
            if ((*vt).get_controller_class_id)(component.0, &mut controller_cid) == K_RESULT_OK
                && controller_cid != [0; 16]
            {
                let fvt = *(factory_iface.0 as *const *const IPluginFactoryVtbl);
                let mut obj: *mut c_void = std::ptr::null_mut();
                let status = ((*fvt).create_instance)(
                    factory_iface.0,
                    controller_cid.as_ptr() as *const i8,
                    IEDITCONTROLLER_IID.as_ptr() as *const i8,
                    &mut obj,
                );
                if status == K_RESULT_OK && !obj.is_null() {
                    let ctrl = ComPtr(obj);
                    let cvt = *(ctrl.0 as *const *const IEditControllerVtbl);
                    ((*cvt).initialize)(ctrl.0, std::ptr::null_mut());
                    (Some(ctrl), true)
                } else {
                    (None, false)
                }
            } else {
                (None, false)
            }
        },
    };

    let (num_inputs, num_outputs) = unsafe {
        let vt = *(component.0 as *const *const IComponentVtbl);
        let in_count = ((*vt).get_bus_count)(component.0, K_AUDIO, K_INPUT);
        let out_count = ((*vt).get_bus_count)(component.0, K_AUDIO, K_OUTPUT);

        let mut in_channels = 0usize;
        if in_count > 0 {
            let mut info: BusInfo = std::mem::zeroed();
            if ((*vt).get_bus_info)(component.0, K_AUDIO, K_INPUT, 0, &mut info) == K_RESULT_OK {
                ((*vt).activate_bus)(component.0, K_AUDIO, K_INPUT, 0, 1);
                in_channels = info.channel_count.max(0) as usize;
            }
        }

        let mut out_channels = 2usize;
        if out_count > 0 {
            let mut info: BusInfo = std::mem::zeroed();
            if ((*vt).get_bus_info)(component.0, K_AUDIO, K_OUTPUT, 0, &mut info) == K_RESULT_OK {
                ((*vt).activate_bus)(component.0, K_AUDIO, K_OUTPUT, 0, 1);
                out_channels = info.channel_count.max(0) as usize;
            }
        }

        (in_channels, out_channels)
    };

    let mut factory_info: PFactoryInfo = unsafe { std::mem::zeroed() };
    let vendor = unsafe {
        let vt = *(factory_iface.0 as *const *const IPluginFactoryVtbl);
        if ((*vt).get_factory_info)(factory_iface.0, &mut factory_info) == K_RESULT_OK {
            latin1_z_to_string(&factory_info.vendor)
        } else {
            descriptor.vendor.clone()
        }
    };

    Ok(Box::new(Vst3Hosted {
        _module: module,
        component,
        processor,
        controller,
        controller_owned,
        name: descriptor.name.clone(),
        vendor,
        num_inputs,
        num_outputs,
        sample_size: K_SAMPLE_32,
        block_size: 512,
        active: false,
        initialized_processing: false,
        parameter_ids: Vec::new(),
        input_scratch_f64: Vec::new(),
        output_scratch_f64: Vec::new(),
    }))
}

impl HostedPlugin for Vst3Hosted {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn vendor(&self) -> String {
        self.vendor.clone()
    }
    fn num_inputs(&self) -> usize {
        self.num_inputs
    }
    fn num_outputs(&self) -> usize {
        self.num_outputs
    }
    fn num_parameters(&self) -> usize {
        self.parameter_ids.len()
    }

    fn parameter_name(&self, index: usize) -> String {
        let (Some(controller), Some(&id)) = (&self.controller, self.parameter_ids.get(index))
        else {
            return String::new();
        };
        unsafe {
            let vt = *(controller.0 as *const *const IEditControllerVtbl);
            let param_index = index as i32;
            let mut info: ParameterInfo = std::mem::zeroed();
            if ((*vt).get_parameter_info)(controller.0, param_index, &mut info) == K_RESULT_OK
                && info.id == id
            {
                utf16_z_to_string(&info.title)
            } else {
                String::new()
            }
        }
    }

    fn get_parameter(&self, index: usize) -> f32 {
        let (Some(controller), Some(&id)) = (&self.controller, self.parameter_ids.get(index))
        else {
            return 0.0;
        };
        unsafe {
            let vt = *(controller.0 as *const *const IEditControllerVtbl);
            ((*vt).get_param_normalized)(controller.0, id)
        }
    }

    fn set_parameter(&mut self, index: usize, value: f32) {
        let (Some(controller), Some(&id)) = (&self.controller, self.parameter_ids.get(index))
        else {
            return;
        };
        unsafe {
            let vt = *(controller.0 as *const *const IEditControllerVtbl);
            ((*vt).set_param_normalized)(controller.0, id, value.clamp(0.0, 1.0));
        }
    }

    fn prepare(&mut self, sample_rate: usize, block_size: usize) -> HostResult<()> {
        if self.initialized_processing {
            unsafe {
                let vt = *(self.processor.0 as *const *const IAudioProcessorVtbl);
                ((*vt).set_processing)(self.processor.0, 0);
            }
        }

        // Prefer 32-bit float (matches this library's native sample type
        // with no conversion, and every VST3 plugin is required to support
        // it per the spec); fall back to 64-bit float.
        let sample_size = unsafe {
            let vt = *(self.processor.0 as *const *const IAudioProcessorVtbl);
            if ((*vt).can_process_sample_size)(self.processor.0, K_SAMPLE_32) == K_RESULT_OK {
                K_SAMPLE_32
            } else {
                K_SAMPLE_64
            }
        };

        let mut setup = ProcessSetup {
            process_mode: 0,
            symbolic_sample_size: sample_size,
            max_samples_per_block: block_size as i32,
            sample_rate: sample_rate as f32,
        };
        let status = unsafe {
            let vt = *(self.processor.0 as *const *const IAudioProcessorVtbl);
            ((*vt).setup_processing)(self.processor.0, &mut setup)
        };
        if status != K_RESULT_OK {
            return Err(HostError::InitializationFailed(
                "IAudioProcessor::setupProcessing failed".into(),
            ));
        }

        // Parameter list is queried from the controller (may be the same
        // object as `component`, or a separately-created one).
        let mut parameter_ids = Vec::new();
        if let Some(controller) = &self.controller {
            unsafe {
                let vt = *(controller.0 as *const *const IEditControllerVtbl);
                let count = ((*vt).get_parameter_count)(controller.0);
                for i in 0..count {
                    let mut info: ParameterInfo = std::mem::zeroed();
                    if ((*vt).get_parameter_info)(controller.0, i, &mut info) == K_RESULT_OK {
                        parameter_ids.push(info.id);
                    }
                }
            }
        }

        self.sample_size = sample_size;
        self.block_size = block_size;
        self.parameter_ids = parameter_ids;
        self.initialized_processing = true;

        if sample_size == K_SAMPLE_64 {
            self.input_scratch_f64 = (0..self.num_inputs)
                .map(|_| vec![0.0f64; block_size])
                .collect();
            self.output_scratch_f64 = (0..self.num_outputs)
                .map(|_| vec![0.0f64; block_size])
                .collect();
        } else {
            self.input_scratch_f64 = Vec::new();
            self.output_scratch_f64 = Vec::new();
        }

        Ok(())
    }

    fn set_active(&mut self, active: bool) -> HostResult<()> {
        if active == self.active {
            return Ok(());
        }
        if !self.initialized_processing {
            return Err(HostError::Unsupported(
                "prepare() must be called before set_active(true)".into(),
            ));
        }

        unsafe {
            let cvt = *(self.component.0 as *const *const IComponentVtbl);
            if ((*cvt).set_active)(self.component.0, if active { 1 } else { 0 }) != K_RESULT_OK {
                return Err(HostError::InitializationFailed(
                    "IComponent::setActive failed".into(),
                ));
            }

            let pvt = *(self.processor.0 as *const *const IAudioProcessorVtbl);
            ((*pvt).set_processing)(self.processor.0, if active { 1 } else { 0 });
        }

        self.active = active;
        Ok(())
    }

    fn process(&mut self, audio: &mut AudioIO<'_>) {
        if !self.active {
            return;
        }

        let frames = audio
            .output
            .first()
            .map(|b| b.len())
            .unwrap_or(0)
            .min(self.block_size);
        if frames == 0 {
            return;
        }

        let input_channels = audio.input.unwrap_or(&[]);

        let mut input_ptrs: Vec<*mut c_void>;
        let mut output_ptrs: Vec<*mut c_void>;

        if self.sample_size == K_SAMPLE_32 {
            input_ptrs = input_channels
                .iter()
                .take(self.num_inputs)
                .map(|b| b.as_ptr() as *mut c_void)
                .collect();
            output_ptrs = audio
                .output
                .iter_mut()
                .take(self.num_outputs)
                .map(|b| b.as_mut_ptr() as *mut c_void)
                .collect();
        } else {
            for (ch, buf) in input_channels.iter().take(self.num_inputs).enumerate() {
                for i in 0..frames {
                    self.input_scratch_f64[ch][i] = buf[i] as f64;
                }
            }
            input_ptrs = self
                .input_scratch_f64
                .iter_mut()
                .map(|v| v.as_mut_ptr() as *mut c_void)
                .collect();
            output_ptrs = self
                .output_scratch_f64
                .iter_mut()
                .map(|v| v.as_mut_ptr() as *mut c_void)
                .collect();
        }

        let mut input_bus = AudioBusBuffers {
            num_channels: input_ptrs.len() as i32,
            silence_flags: 0,
            channel_buffers: input_ptrs.as_mut_ptr(),
        };
        let mut output_bus = AudioBusBuffers {
            num_channels: output_ptrs.len() as i32,
            silence_flags: 0,
            channel_buffers: output_ptrs.as_mut_ptr(),
        };

        let mut data = ProcessData {
            process_mode: 0,
            symbolic_sample_size: self.sample_size,
            num_samples: frames as i32,
            num_inputs: if input_ptrs.is_empty() { 0 } else { 1 },
            num_outputs: if output_ptrs.is_empty() { 0 } else { 1 },
            inputs: if input_ptrs.is_empty() {
                std::ptr::null_mut()
            } else {
                &mut input_bus
            },
            outputs: if output_ptrs.is_empty() {
                std::ptr::null_mut()
            } else {
                &mut output_bus
            },
            input_parameter_changes: std::ptr::null_mut(),
            output_parameter_changes: std::ptr::null_mut(),
            input_events: std::ptr::null_mut(),
            output_events: std::ptr::null_mut(),
            process_context: std::ptr::null_mut(),
        };

        unsafe {
            let vt = *(self.processor.0 as *const *const IAudioProcessorVtbl);
            ((*vt).process)(self.processor.0, &mut data);
        }

        if self.sample_size == K_SAMPLE_64 {
            for (ch, buf) in audio.output.iter_mut().take(self.num_outputs).enumerate() {
                for i in 0..frames {
                    buf[i] = self.output_scratch_f64[ch][i] as f32;
                }
            }
        }
    }
}
