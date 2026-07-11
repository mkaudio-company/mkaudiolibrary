//! AUv2 (Audio Unit) hosting backend for macOS.
//!
//! Instantiates real installed Audio Units via the system's AudioComponent
//! registry (`AudioComponentFindNext`/`AudioComponentInstanceNew`) and
//! drives them in a "pull" fashion: `process()` sets up a small render
//! callback context with pointers to the current input buffers, then calls
//! `AudioUnitRender` on the output bus - the AU pulls input through that
//! callback as needed, exactly like a realtime AUHAL host would, just
//! without a hardware clock behind it.
//!
//! Audio is exchanged in **non-interleaved** 64-bit float form (one
//! `AudioBuffer` per channel), matching `AudioIO`'s planar `Vec<Buffer<f64>>`
//! layout directly with no interleave/deinterleave copy.

use std::ffi::c_void;

use coreaudio_sys::*;

use crate::macos_util::cfstring_to_string;
use crate::processor::AudioIO;

use super::{HostError, HostResult, HostedPlugin, PluginDescriptor, PluginFormat};

fn check(status: OSStatus, what: &str) -> HostResult<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(HostError::InitializationFailed(format!(
            "{} failed (OSStatus {})",
            what, status
        )))
    }
}

fn au_type_name(t: OSType) -> &'static str {
    if t == kAudioUnitType_Effect {
        "Effect"
    } else if t == kAudioUnitType_MusicEffect {
        "MusicEffect"
    } else if t == kAudioUnitType_MusicDevice {
        "Instrument"
    } else if t == kAudioUnitType_Generator {
        "Generator"
    } else if t == kAudioUnitType_Mixer {
        "Mixer"
    } else if t == kAudioUnitType_Panner {
        "Panner"
    } else if t == kAudioUnitType_OfflineEffect {
        "OfflineEffect"
    } else {
        "Unknown"
    }
}

/// Scan the system's AudioComponent registry for installed, hostable Audio Units.
pub fn scan() -> Vec<PluginDescriptor> {
    let mut results = Vec::new();
    let types = [
        kAudioUnitType_Effect,
        kAudioUnitType_MusicEffect,
        kAudioUnitType_MusicDevice,
        kAudioUnitType_Generator,
        kAudioUnitType_Mixer,
        kAudioUnitType_Panner,
        kAudioUnitType_OfflineEffect,
    ];

    for &component_type in &types {
        let wildcard = AudioComponentDescription {
            componentType: component_type,
            componentSubType: 0,
            componentManufacturer: 0,
            componentFlags: 0,
            componentFlagsMask: 0,
        };
        let mut component: AudioComponent = std::ptr::null_mut();

        loop {
            component = unsafe { AudioComponentFindNext(component, &wildcard) };
            if component.is_null() {
                break;
            }

            let mut desc = AudioComponentDescription {
                componentType: 0,
                componentSubType: 0,
                componentManufacturer: 0,
                componentFlags: 0,
                componentFlagsMask: 0,
            };
            if unsafe { AudioComponentGetDescription(component, &mut desc) } != 0 {
                continue;
            }

            let mut name_ref: CFStringRef = std::ptr::null();
            let full_name = if unsafe { AudioComponentCopyName(component, &mut name_ref) } == 0 {
                let s = unsafe { cfstring_to_string(name_ref) };
                if !name_ref.is_null() {
                    unsafe {
                        CFRelease(name_ref as CFTypeRef);
                    }
                }
                s
            } else {
                String::new()
            };

            let (vendor, name) = match full_name.split_once(": ") {
                Some((v, n)) => (v.to_string(), n.to_string()),
                None => (String::new(), full_name),
            };

            results.push(PluginDescriptor {
                format: PluginFormat::Au,
                name,
                vendor,
                path: None,
                category: au_type_name(desc.componentType).to_string(),
                au_component: Some((
                    desc.componentType,
                    desc.componentSubType,
                    desc.componentManufacturer,
                )),
                vst3_class_id: None,
            });
        }
    }

    results
}

fn query_channels(unit: AudioUnit, scope: AudioUnitScope, element: AudioUnitElement) -> usize {
    unsafe {
        let mut asbd: AudioStreamBasicDescription = std::mem::zeroed();
        let mut size = std::mem::size_of::<AudioStreamBasicDescription>() as UInt32;
        let status = AudioUnitGetProperty(
            unit,
            kAudioUnitProperty_StreamFormat as AudioUnitPropertyID,
            scope,
            element,
            &mut asbd as *mut _ as *mut c_void,
            &mut size,
        );
        if status == 0 {
            asbd.mChannelsPerFrame as usize
        } else {
            0
        }
    }
}

/// Most third-party AUs only implement the "canonical" 32-bit float format;
/// 64-bit float is tried first (to avoid a conversion copy) and this is the
/// fallback when a plugin rejects it, which is common in practice.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SampleWidth {
    F64,
    F32,
}

impl SampleWidth {
    fn bytes(self) -> usize {
        match self {
            SampleWidth::F64 => 8,
            SampleWidth::F32 => 4,
        }
    }
}

fn make_noninterleaved_asbd(
    sample_rate: usize,
    channels: usize,
    width: SampleWidth,
) -> AudioStreamBasicDescription {
    let bytes = width.bytes() as UInt32;
    AudioStreamBasicDescription {
        mSampleRate: sample_rate as Float64,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat
            | kAudioFormatFlagIsPacked
            | kAudioFormatFlagIsNonInterleaved,
        mBytesPerPacket: bytes,
        mFramesPerPacket: 1,
        mBytesPerFrame: bytes,
        mChannelsPerFrame: channels as UInt32,
        mBitsPerChannel: bytes * 8,
        mReserved: 0,
    }
}

/// Try to set the stream format for a scope/element; returns whether the AU accepted it.
fn try_set_format(
    unit: AudioUnit,
    scope: AudioUnitScope,
    element: AudioUnitElement,
    sample_rate: usize,
    channels: usize,
    width: SampleWidth,
) -> bool {
    if channels == 0 {
        return true;
    }
    let asbd = make_noninterleaved_asbd(sample_rate, channels, width);
    unsafe {
        AudioUnitSetProperty(
            unit,
            kAudioUnitProperty_StreamFormat as AudioUnitPropertyID,
            scope,
            element,
            &asbd as *const _ as *const c_void,
            std::mem::size_of::<AudioStreamBasicDescription>() as UInt32,
        ) == 0
    }
}

/// Backing storage sized to hold an `AudioBufferList` with `channels` buffers.
/// Allocated as `u64` words so the (C-flexible-array-member) layout is
/// correctly 8-byte aligned for the pointer fields inside each `AudioBuffer`.
fn alloc_buffer_list_storage(channels: usize) -> Vec<u64> {
    let total_bytes = std::mem::size_of::<AudioBufferList>()
        + channels.saturating_sub(1) * std::mem::size_of::<AudioBuffer>();
    vec![0u64; total_bytes.div_ceil(8)]
}

/// Shared with the input render callback via `inRefCon`: updated right
/// before each `AudioUnitRender` call in `process()` with pointers into
/// that call's (locked, and possibly f32-converted) input buffers, plus the
/// negotiated sample width so the copy is byte-count-correct either way.
/// `process()` isn't reentrant, so there's no concurrent access to guard against.
struct InputContext {
    channel_ptrs: Vec<*const u8>,
    frames: usize,
    bytes_per_sample: usize,
}

unsafe extern "C" fn au_input_proc(
    in_ref_con: *mut c_void,
    _io_action_flags: *mut AudioUnitRenderActionFlags,
    _in_time_stamp: *const AudioTimeStamp,
    _in_bus_number: UInt32,
    in_number_frames: UInt32,
    io_data: *mut AudioBufferList,
) -> OSStatus {
    unsafe {
        let ctx = &*(in_ref_con as *const InputContext);
        let frames = (in_number_frames as usize).min(ctx.frames);
        let byte_len = frames * ctx.bytes_per_sample;

        let num_buffers = (*io_data).mNumberBuffers as usize;
        let buffers = std::ptr::addr_of_mut!((*io_data).mBuffers) as *mut AudioBuffer;

        for i in 0..num_buffers.min(ctx.channel_ptrs.len()) {
            let dst = (*buffers.add(i)).mData as *mut u8;
            if dst.is_null() {
                continue;
            }

            let src = ctx.channel_ptrs[i];
            if src.is_null() {
                std::ptr::write_bytes(dst, 0, byte_len);
            } else {
                std::ptr::copy_nonoverlapping(src, dst, byte_len);
            }
        }

        0
    }
}

pub(super) struct AuHosted {
    unit: AudioUnit,
    input_ctx_ptr: *mut InputContext,
    num_inputs: usize,
    num_outputs: usize,
    name: String,
    vendor: String,
    parameter_ids: Vec<AudioUnitParameterID>,
    block_size: usize,
    active: bool,
    prepared: bool,
    width: SampleWidth,
    // Only populated (and used) when `width == SampleWidth::F32`: per-channel
    // conversion scratch, sized to `block_size` once `prepare()` runs.
    input_scratch_f32: Vec<Vec<f32>>,
    output_scratch_f32: Vec<Vec<f32>>,
    output_storage: Vec<u64>,
}

// `unit` and `input_ctx_ptr` are only ever touched from whichever thread
// owns this `AuHosted` value (there's no background thread here - unlike
// `realtime::coreaudio_impl`, this host is pulled synchronously by the
// caller of `process()`), so moving the whole struct across threads is safe.
unsafe impl Send for AuHosted {}

impl Drop for AuHosted {
    fn drop(&mut self) {
        unsafe {
            if self.prepared {
                AudioUnitUninitialize(self.unit);
            }
            AudioComponentInstanceDispose(self.unit);
            if !self.input_ctx_ptr.is_null() {
                drop(Box::from_raw(self.input_ctx_ptr));
            }
        }
    }
}

pub(super) fn load(descriptor: &PluginDescriptor) -> HostResult<Box<dyn HostedPlugin>> {
    let (component_type, subtype, manufacturer) = descriptor.au_component.ok_or_else(|| {
        HostError::LoadFailed("AU descriptor is missing its component identity".into())
    })?;

    let desc = AudioComponentDescription {
        componentType: component_type,
        componentSubType: subtype,
        componentManufacturer: manufacturer,
        componentFlags: 0,
        componentFlagsMask: 0,
    };
    let component = unsafe { AudioComponentFindNext(std::ptr::null_mut(), &desc) };
    if component.is_null() {
        return Err(HostError::NotFound(descriptor.name.clone()));
    }

    let mut unit: AudioUnit = std::ptr::null_mut();
    check(
        unsafe { AudioComponentInstanceNew(component, &mut unit) },
        "AudioComponentInstanceNew",
    )?;

    let num_inputs = query_channels(unit, kAudioUnitScope_Input as AudioUnitScope, 0);
    let num_outputs = query_channels(unit, kAudioUnitScope_Output as AudioUnitScope, 0).max(1);

    let input_ctx_ptr = Box::into_raw(Box::new(InputContext {
        channel_ptrs: vec![std::ptr::null(); num_inputs.max(1)],
        frames: 0,
        bytes_per_sample: 8,
    }));

    Ok(Box::new(AuHosted {
        unit,
        input_ctx_ptr,
        num_inputs,
        num_outputs,
        name: descriptor.name.clone(),
        vendor: descriptor.vendor.clone(),
        parameter_ids: Vec::new(),
        block_size: 512,
        active: false,
        prepared: false,
        width: SampleWidth::F64,
        input_scratch_f32: Vec::new(),
        output_scratch_f32: Vec::new(),
        output_storage: Vec::new(),
    }))
}

impl HostedPlugin for AuHosted {
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
        let Some(&param_id) = self.parameter_ids.get(index) else {
            return String::new();
        };

        unsafe {
            let mut info: AudioUnitParameterInfo = std::mem::zeroed();
            let mut size = std::mem::size_of::<AudioUnitParameterInfo>() as UInt32;
            if AudioUnitGetProperty(
                self.unit,
                kAudioUnitProperty_ParameterInfo as AudioUnitPropertyID,
                kAudioUnitScope_Global as AudioUnitScope,
                param_id,
                &mut info as *mut _ as *mut c_void,
                &mut size,
            ) != 0
            {
                return String::new();
            }

            if info.flags & kAudioUnitParameterFlag_HasCFNameString != 0
                && !info.cfNameString.is_null()
            {
                let name = cfstring_to_string(info.cfNameString);
                if info.flags & kAudioUnitParameterFlag_CFNameRelease != 0 {
                    CFRelease(info.cfNameString as CFTypeRef);
                }
                name
            } else {
                std::ffi::CStr::from_ptr(info.name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    fn get_parameter(&self, index: usize) -> f64 {
        let Some(&param_id) = self.parameter_ids.get(index) else {
            return 0.0;
        };
        unsafe {
            let mut value: AudioUnitParameterValue = 0.0;
            if AudioUnitGetParameter(
                self.unit,
                param_id,
                kAudioUnitScope_Global as AudioUnitScope,
                0,
                &mut value,
            ) == 0
            {
                value as f64
            } else {
                0.0
            }
        }
    }

    fn set_parameter(&mut self, index: usize, value: f64) {
        let Some(&param_id) = self.parameter_ids.get(index) else {
            return;
        };
        unsafe {
            AudioUnitSetParameter(
                self.unit,
                param_id,
                kAudioUnitScope_Global as AudioUnitScope,
                0,
                value as AudioUnitParameterValue,
                0,
            );
        }
    }

    fn prepare(&mut self, sample_rate: usize, block_size: usize) -> HostResult<()> {
        if self.prepared {
            unsafe {
                AudioUnitUninitialize(self.unit);
            }
            self.prepared = false;
        }

        // Try 64-bit float first (matches this library's native sample
        // type with no conversion); fall back to 32-bit float, which is
        // what most third-party AUs actually implement.
        let width = if try_set_format(
            self.unit,
            kAudioUnitScope_Input as AudioUnitScope,
            0,
            sample_rate,
            self.num_inputs,
            SampleWidth::F64,
        ) && try_set_format(
            self.unit,
            kAudioUnitScope_Output as AudioUnitScope,
            0,
            sample_rate,
            self.num_outputs,
            SampleWidth::F64,
        ) {
            SampleWidth::F64
        } else if try_set_format(
            self.unit,
            kAudioUnitScope_Input as AudioUnitScope,
            0,
            sample_rate,
            self.num_inputs,
            SampleWidth::F32,
        ) && try_set_format(
            self.unit,
            kAudioUnitScope_Output as AudioUnitScope,
            0,
            sample_rate,
            self.num_outputs,
            SampleWidth::F32,
        ) {
            SampleWidth::F32
        } else {
            return Err(HostError::InitializationFailed(
                "plugin accepts neither 64-bit nor 32-bit non-interleaved float".into(),
            ));
        };

        unsafe {
            let max_frames = block_size as UInt32;
            check(
                AudioUnitSetProperty(
                    self.unit,
                    kAudioUnitProperty_MaximumFramesPerSlice as AudioUnitPropertyID,
                    kAudioUnitScope_Global as AudioUnitScope,
                    0,
                    &max_frames as *const UInt32 as *const c_void,
                    std::mem::size_of::<UInt32>() as UInt32,
                ),
                "MaximumFramesPerSlice",
            )?;

            if self.num_inputs > 0 {
                let render_callback = AURenderCallbackStruct {
                    inputProc: Some(au_input_proc),
                    inputProcRefCon: self.input_ctx_ptr as *mut c_void,
                };
                check(
                    AudioUnitSetProperty(
                        self.unit,
                        kAudioUnitProperty_SetRenderCallback as AudioUnitPropertyID,
                        kAudioUnitScope_Input as AudioUnitScope,
                        0,
                        &render_callback as *const _ as *const c_void,
                        std::mem::size_of::<AURenderCallbackStruct>() as UInt32,
                    ),
                    "SetRenderCallback",
                )?;
            }

            check(AudioUnitInitialize(self.unit), "AudioUnitInitialize")?;

            // Parameter list is only guaranteed populated once initialized.
            let mut size: UInt32 = 0;
            if AudioUnitGetPropertyInfo(
                self.unit,
                kAudioUnitProperty_ParameterList as AudioUnitPropertyID,
                kAudioUnitScope_Global as AudioUnitScope,
                0,
                &mut size,
                std::ptr::null_mut(),
            ) == 0
                && size > 0
            {
                let count = size as usize / std::mem::size_of::<AudioUnitParameterID>();
                let mut ids = vec![0 as AudioUnitParameterID; count];
                if AudioUnitGetProperty(
                    self.unit,
                    kAudioUnitProperty_ParameterList as AudioUnitPropertyID,
                    kAudioUnitScope_Global as AudioUnitScope,
                    0,
                    ids.as_mut_ptr() as *mut c_void,
                    &mut size,
                ) == 0
                {
                    self.parameter_ids = ids;
                }
            }
        }

        self.width = width;
        self.block_size = block_size;
        self.output_storage = alloc_buffer_list_storage(self.num_outputs.max(1));
        if width == SampleWidth::F32 {
            self.input_scratch_f32 = (0..self.num_inputs)
                .map(|_| vec![0.0f32; block_size])
                .collect();
            self.output_scratch_f32 = (0..self.num_outputs)
                .map(|_| vec![0.0f32; block_size])
                .collect();
        } else {
            self.input_scratch_f32 = Vec::new();
            self.output_scratch_f32 = Vec::new();
        }
        self.prepared = true;
        Ok(())
    }

    fn set_active(&mut self, active: bool) -> HostResult<()> {
        self.active = active;
        // Best-effort: not every AU implements bypass, so a failure here
        // isn't fatal - `process()` just keeps running the AU either way.
        let bypass: UInt32 = if active { 0 } else { 1 };
        unsafe {
            AudioUnitSetProperty(
                self.unit,
                kAudioUnitProperty_BypassEffect as AudioUnitPropertyID,
                kAudioUnitScope_Global as AudioUnitScope,
                0,
                &bypass as *const UInt32 as *const c_void,
                std::mem::size_of::<UInt32>() as UInt32,
            );
        }
        Ok(())
    }

    fn process(&mut self, audio: &mut AudioIO) {
        if !self.prepared {
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

        let input_guards: Vec<_> = audio
            .input
            .iter()
            .take(self.num_inputs)
            .map(|b| b.read())
            .collect();
        let bytes_per_sample = self.width.bytes();

        unsafe {
            let ctx = &mut *self.input_ctx_ptr;
            ctx.channel_ptrs.clear();
            ctx.bytes_per_sample = bytes_per_sample;

            match self.width {
                SampleWidth::F64 => {
                    for guard in &input_guards {
                        ctx.channel_ptrs.push(guard.as_ptr() as *const u8);
                    }
                }
                SampleWidth::F32 => {
                    for (ch, guard) in input_guards.iter().enumerate() {
                        let scratch = &mut self.input_scratch_f32[ch];
                        for i in 0..frames {
                            scratch[i] = guard[i] as f32;
                        }
                        ctx.channel_ptrs.push(scratch.as_ptr() as *const u8);
                    }
                }
            }
            ctx.frames = frames;
        }

        let mut output_guards: Vec<_> = audio
            .output
            .iter()
            .take(self.num_outputs)
            .map(|b| b.write())
            .collect();

        unsafe {
            let list_ptr = self.output_storage.as_mut_ptr() as *mut AudioBufferList;
            (*list_ptr).mNumberBuffers = output_guards.len() as UInt32;
            let buffers = std::ptr::addr_of_mut!((*list_ptr).mBuffers) as *mut AudioBuffer;

            for (i, guard) in output_guards.iter_mut().enumerate() {
                let data_ptr = match self.width {
                    SampleWidth::F64 => guard.as_mut_ptr() as *mut c_void,
                    SampleWidth::F32 => self.output_scratch_f32[i].as_mut_ptr() as *mut c_void,
                };
                buffers.add(i).write(AudioBuffer {
                    mNumberChannels: 1,
                    mDataByteSize: (frames * bytes_per_sample) as UInt32,
                    mData: data_ptr,
                });
            }

            let mut flags: AudioUnitRenderActionFlags = 0;
            let mut timestamp: AudioTimeStamp = std::mem::zeroed();
            timestamp.mFlags = kAudioTimeStampSampleTimeValid;

            AudioUnitRender(
                self.unit,
                &mut flags,
                &timestamp,
                0,
                frames as UInt32,
                list_ptr,
            );

            if self.width == SampleWidth::F32 {
                for (i, guard) in output_guards.iter_mut().enumerate() {
                    let scratch = &self.output_scratch_f32[i];
                    for f in 0..frames {
                        guard[f] = scratch[f] as f64;
                    }
                }
            }
        }
    }
}
