//! Small shared macOS FFI helpers used by both the CoreAudio realtime
//! backend (`realtime` feature) and the AUv2 plugin host (`au` feature).

use coreaudio_sys::{
    CFStringGetCString, CFStringGetLength, CFStringGetMaximumSizeForEncoding, CFStringRef,
    kCFStringEncodingUTF8,
};

// coreaudio-sys exposes CoreFoundation's CFString functions (they're pulled
// in transitively by the CoreAudio headers it binds), but its build script
// only links AudioUnit/AudioToolbox/CoreAudio/OpenAL/CoreMIDI - not
// CoreFoundation itself. Link it explicitly rather than pulling in the
// (unused, Rust-level) `core-foundation` crate just to get its build script.
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {}

/// Copy a `CFStringRef` into an owned Rust `String`. Returns an empty
/// string for a null reference or on conversion failure.
///
/// # Safety
/// `cf_str` must be a valid `CFStringRef` (or null).
pub(crate) unsafe fn cfstring_to_string(cf_str: CFStringRef) -> String {
    if cf_str.is_null() {
        return String::new();
    }

    unsafe {
        let length = CFStringGetLength(cf_str);
        let max_size = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
        let mut buf = vec![0u8; max_size.max(1) as usize];

        if CFStringGetCString(
            cf_str,
            buf.as_mut_ptr() as *mut i8,
            max_size,
            kCFStringEncodingUTF8,
        ) != 0
        {
            let cstr = std::ffi::CStr::from_ptr(buf.as_ptr() as *const i8);
            cstr.to_string_lossy().into_owned()
        } else {
            String::new()
        }
    }
}
