//! CoreAudio (AUHAL) backend for macOS.
//!
//! Uses the AUHAL output unit (`kAudioUnitSubType_HALOutput`) for real,
//! hardware-clocked duplex audio I/O — the same mechanism upstream RtAudio's
//! `RtApiCore` uses. Device enumeration goes through `AudioObjectGetPropertyData`
//! against the system audio object; streaming goes through an `AudioUnit`
//! render callback that pulls captured input (via `AudioUnitRender` on the
//! input bus) before invoking the user callback and writing its output back
//! into the AUHAL-provided buffer.

use std::ffi::c_void;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use coreaudio_sys::*;

use super::{
    AudioCallback, Backend, DeviceInfo, MKAudioError, MKAudioResult, SampleFormat, StreamOptions,
    StreamParameters, StreamState, StreamStatus, invoke_callback,
};
use crate::macos_util::cfstring_to_string;

#[inline]
fn check(status: OSStatus, what: &str) -> MKAudioResult<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(MKAudioError::DriverError(format!(
            "{} failed (OSStatus {})",
            what, status
        )))
    }
}

fn property_address(
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn all_device_ids() -> Vec<AudioDeviceID> {
    unsafe {
        let address = property_address(
            kAudioHardwarePropertyDevices,
            kAudioObjectPropertyScopeGlobal,
        );
        let mut size: UInt32 = 0;
        if AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject,
            &address,
            0,
            std::ptr::null(),
            &mut size,
        ) != 0
        {
            return Vec::new();
        }

        let count = size as usize / std::mem::size_of::<AudioDeviceID>();
        let mut ids = vec![0 as AudioDeviceID; count];
        if AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &address,
            0,
            std::ptr::null(),
            &mut size,
            ids.as_mut_ptr() as *mut c_void,
        ) != 0
        {
            return Vec::new();
        }

        ids
    }
}

fn device_name(device_id: AudioDeviceID) -> String {
    unsafe {
        let address = property_address(
            kAudioDevicePropertyDeviceNameCFString,
            kAudioObjectPropertyScopeGlobal,
        );
        let mut name_ref: CFStringRef = std::ptr::null();
        let mut size = std::mem::size_of::<CFStringRef>() as UInt32;
        if AudioObjectGetPropertyData(
            device_id,
            &address,
            0,
            std::ptr::null(),
            &mut size,
            &mut name_ref as *mut _ as *mut c_void,
        ) != 0
        {
            return String::new();
        }

        let name = cfstring_to_string(name_ref);
        if !name_ref.is_null() {
            CFRelease(name_ref as CFTypeRef);
        }
        name
    }
}

/// Number of channels a device exposes in the given scope, by summing the
/// channel counts of every buffer in its stream configuration.
fn device_channel_count(device_id: AudioDeviceID, scope: AudioObjectPropertyScope) -> usize {
    unsafe {
        let address = property_address(kAudioDevicePropertyStreamConfiguration, scope);
        let mut size: UInt32 = 0;
        if AudioObjectGetPropertyDataSize(device_id, &address, 0, std::ptr::null(), &mut size) != 0
            || size == 0
        {
            return 0;
        }

        let mut raw = vec![0u8; size as usize];
        if AudioObjectGetPropertyData(
            device_id,
            &address,
            0,
            std::ptr::null(),
            &mut size,
            raw.as_mut_ptr() as *mut c_void,
        ) != 0
        {
            return 0;
        }

        let list = raw.as_ptr() as *const AudioBufferList;
        let num_buffers = (*list).mNumberBuffers as usize;
        let first_buffer = &(*list).mBuffers[0] as *const AudioBuffer;
        let mut total = 0usize;
        for i in 0..num_buffers {
            total += (*first_buffer.add(i)).mNumberChannels as usize;
        }
        total
    }
}

fn device_sample_rate(device_id: AudioDeviceID) -> usize {
    unsafe {
        let address = property_address(
            kAudioDevicePropertyNominalSampleRate,
            kAudioObjectPropertyScopeGlobal,
        );
        let mut rate: Float64 = 0.0;
        let mut size = std::mem::size_of::<Float64>() as UInt32;
        if AudioObjectGetPropertyData(
            device_id,
            &address,
            0,
            std::ptr::null(),
            &mut size,
            &mut rate as *mut _ as *mut c_void,
        ) != 0
        {
            return 44100;
        }
        rate.round() as usize
    }
}

fn default_device(selector: AudioObjectPropertySelector) -> AudioDeviceID {
    unsafe {
        let address = property_address(selector, kAudioObjectPropertyScopeGlobal);
        let mut device_id: AudioDeviceID = 0;
        let mut size = std::mem::size_of::<AudioDeviceID>() as UInt32;
        if AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &address,
            0,
            std::ptr::null(),
            &mut size,
            &mut device_id as *mut _ as *mut c_void,
        ) != 0
        {
            return 0;
        }
        device_id
    }
}

fn make_asbd(sample_rate: usize, channels: usize) -> AudioStreamBasicDescription {
    let bytes_per_frame = (channels * std::mem::size_of::<f64>()) as UInt32;
    AudioStreamBasicDescription {
        mSampleRate: sample_rate as Float64,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: bytes_per_frame,
        mFramesPerPacket: 1,
        mBytesPerFrame: bytes_per_frame,
        mChannelsPerFrame: channels as UInt32,
        mBitsPerChannel: 64,
        mReserved: 0,
    }
}

/// Context shared with the AUHAL render callback via `inRefCon`. Only the
/// audio thread touches `input_scratch`; CoreAudio never invokes the render
/// callback concurrently with itself, so no extra synchronization is needed
/// there.
struct RenderContext {
    unit: AudioUnit,
    callback: Arc<Mutex<Option<AudioCallback>>>,
    running: Arc<AtomicBool>,
    stream_time_bits: Arc<AtomicU64>,
    sample_rate: usize,
    input_channels: usize,
    input_enabled: bool,
    input_scratch: Vec<f64>,
}

unsafe extern "C" fn render_proc(
    in_ref_con: *mut c_void,
    io_action_flags: *mut AudioUnitRenderActionFlags,
    in_time_stamp: *const AudioTimeStamp,
    _in_bus_number: UInt32,
    in_number_frames: UInt32,
    io_data: *mut AudioBufferList,
) -> OSStatus {
    unsafe {
        let ctx = &mut *(in_ref_con as *mut RenderContext);
        let frames = in_number_frames as usize;

        if ctx.input_enabled && !ctx.input_scratch.is_empty() {
            let needed = frames * ctx.input_channels;
            if ctx.input_scratch.len() < needed {
                ctx.input_scratch.resize(needed, 0.0);
            }

            let mut input_list = AudioBufferList {
                mNumberBuffers: 1,
                mBuffers: [AudioBuffer {
                    mNumberChannels: ctx.input_channels as UInt32,
                    mDataByteSize: (needed * std::mem::size_of::<f64>()) as UInt32,
                    mData: ctx.input_scratch.as_mut_ptr() as *mut c_void,
                }],
            };
            let status = AudioUnitRender(
                ctx.unit,
                io_action_flags,
                in_time_stamp,
                1,
                in_number_frames,
                &mut input_list,
            );
            if status != 0 {
                ctx.input_scratch[..needed].fill(0.0);
            }
        }

        let out_buffer = &mut (*io_data).mBuffers[0];
        let out_len = (out_buffer.mDataByteSize as usize) / std::mem::size_of::<f64>();
        let output_slice = std::slice::from_raw_parts_mut(out_buffer.mData as *mut f64, out_len);
        let input_slice: &[f64] = if ctx.input_enabled {
            &ctx.input_scratch[..frames * ctx.input_channels]
        } else {
            &[]
        };

        let stream_time = f64::from_bits(ctx.stream_time_bits.load(Ordering::Relaxed));
        invoke_callback(
            &ctx.callback,
            &ctx.running,
            output_slice,
            input_slice,
            frames,
            stream_time,
            StreamStatus::default(),
        );
        ctx.stream_time_bits.store(
            (stream_time + frames as f64 / ctx.sample_rate as f64).to_bits(),
            Ordering::Relaxed,
        );

        0
    }
}

/// Input-only variant: registered via `kAudioOutputUnitProperty_SetInputCallback`
/// and fired whenever new captured input is ready. Pulls it via
/// `AudioUnitRender` on the input bus, then hands it to the user callback
/// with an empty output slice (there is no output bus enabled).
unsafe extern "C" fn input_only_proc(
    in_ref_con: *mut c_void,
    io_action_flags: *mut AudioUnitRenderActionFlags,
    in_time_stamp: *const AudioTimeStamp,
    _in_bus_number: UInt32,
    in_number_frames: UInt32,
    _io_data: *mut AudioBufferList,
) -> OSStatus {
    unsafe {
        let ctx = &mut *(in_ref_con as *mut RenderContext);
        let frames = in_number_frames as usize;
        let needed = frames * ctx.input_channels;
        if ctx.input_scratch.len() < needed {
            ctx.input_scratch.resize(needed, 0.0);
        }

        let mut input_list = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer {
                mNumberChannels: ctx.input_channels as UInt32,
                mDataByteSize: (needed * std::mem::size_of::<f64>()) as UInt32,
                mData: ctx.input_scratch.as_mut_ptr() as *mut c_void,
            }],
        };
        let status = AudioUnitRender(
            ctx.unit,
            io_action_flags,
            in_time_stamp,
            1,
            in_number_frames,
            &mut input_list,
        );
        if status != 0 {
            return status;
        }

        let stream_time = f64::from_bits(ctx.stream_time_bits.load(Ordering::Relaxed));
        invoke_callback(
            &ctx.callback,
            &ctx.running,
            &mut [],
            &ctx.input_scratch[..needed],
            frames,
            stream_time,
            StreamStatus::default(),
        );
        ctx.stream_time_bits.store(
            (stream_time + frames as f64 / ctx.sample_rate as f64).to_bits(),
            Ordering::Relaxed,
        );

        0
    }
}

pub(crate) struct CoreAudioBackend {
    state: StreamState,
    unit: AudioUnit,
    ctx_ptr: *mut RenderContext,
    callback: Arc<Mutex<Option<AudioCallback>>>,
    running: Arc<AtomicBool>,
    stream_time_bits: Arc<AtomicU64>,
    sample_rate: usize,
    buffer_frames: usize,
    number_of_buffers: usize,
}

// `unit`/`ctx_ptr` are only ever dereferenced from the owning thread (for
// control calls) or from CoreAudio's own real-time thread (via the raw
// `inRefCon` pointer handed to it) — never concurrently from safe Rust code.
unsafe impl Send for CoreAudioBackend {}

impl CoreAudioBackend {
    pub fn new() -> Self {
        Self {
            state: StreamState::Closed,
            unit: std::ptr::null_mut(),
            ctx_ptr: std::ptr::null_mut(),
            callback: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            stream_time_bits: Arc::new(AtomicU64::new(0)),
            sample_rate: 44100,
            buffer_frames: 256,
            number_of_buffers: 2,
        }
    }
}

impl Backend for CoreAudioBackend {
    fn device_ids(&self) -> Vec<usize> {
        all_device_ids().into_iter().map(|id| id as usize).collect()
    }

    fn device_info(&self, device_id: usize) -> MKAudioResult<DeviceInfo> {
        let id = device_id as AudioDeviceID;
        if !all_device_ids().contains(&id) {
            return Err(MKAudioError::InvalidDevice(format!(
                "Device {} not found",
                device_id
            )));
        }

        let output_channels = device_channel_count(id, kAudioObjectPropertyScopeOutput);
        let input_channels = device_channel_count(id, kAudioObjectPropertyScopeInput);
        let sample_rate = device_sample_rate(id);

        Ok(DeviceInfo {
            id: device_id,
            name: device_name(id),
            output_channels,
            input_channels,
            duplex_channels: output_channels.min(input_channels),
            is_default_output: id == default_device(kAudioHardwarePropertyDefaultOutputDevice),
            is_default_input: id == default_device(kAudioHardwarePropertyDefaultInputDevice),
            sample_rates: vec![sample_rate],
            preferred_sample_rate: sample_rate,
            native_formats: vec![SampleFormat::Float32, SampleFormat::Float64],
        })
    }

    fn default_output_device(&self) -> usize {
        default_device(kAudioHardwarePropertyDefaultOutputDevice) as usize
    }
    fn default_input_device(&self) -> usize {
        default_device(kAudioHardwarePropertyDefaultInputDevice) as usize
    }

    fn open_stream(
        &mut self,
        output_params: Option<&StreamParameters>,
        input_params: Option<&StreamParameters>,
        sample_rate: usize,
        buffer_frames: usize,
        callback: AudioCallback,
        options: &StreamOptions,
    ) -> MKAudioResult<usize> {
        if self.state != StreamState::Closed {
            return Err(MKAudioError::InvalidUse("Stream is already open".into()));
        }

        let device_id = match (output_params, input_params) {
            (Some(o), Some(i)) if o.device_id != i.device_id => {
                return Err(MKAudioError::InvalidParameter(
                    "the CoreAudio backend requires output and input to share the same device_id for a duplex stream".into()
                ));
            }
            (Some(o), _) => o.device_id,
            (None, Some(i)) => i.device_id,
            (None, None) => {
                return Err(MKAudioError::InvalidParameter(
                    "no stream parameters given".into(),
                ));
            }
        } as AudioDeviceID;

        let output_channels = output_params.map(|p| p.num_channels).unwrap_or(0);
        let input_channels = input_params.map(|p| p.num_channels).unwrap_or(0);

        unsafe {
            let description = AudioComponentDescription {
                componentType: kAudioUnitType_Output,
                componentSubType: kAudioUnitSubType_HALOutput,
                componentManufacturer: kAudioUnitManufacturer_Apple,
                componentFlags: 0,
                componentFlagsMask: 0,
            };

            let component = AudioComponentFindNext(std::ptr::null_mut(), &description);
            if component.is_null() {
                return Err(MKAudioError::DriverError("no AUHAL component found".into()));
            }

            let mut unit: AudioUnit = std::ptr::null_mut();
            check(
                AudioComponentInstanceNew(component, &mut unit),
                "AudioComponentInstanceNew",
            )?;

            let enable: UInt32 = 1;
            let disable: UInt32 = 0;

            check(
                AudioUnitSetProperty(
                    unit,
                    kAudioOutputUnitProperty_EnableIO as AudioUnitPropertyID,
                    kAudioUnitScope_Input as AudioUnitScope,
                    1,
                    (if input_channels > 0 {
                        &enable
                    } else {
                        &disable
                    }) as *const UInt32 as *const c_void,
                    std::mem::size_of::<UInt32>() as UInt32,
                ),
                "EnableIO(input)",
            )?;
            check(
                AudioUnitSetProperty(
                    unit,
                    kAudioOutputUnitProperty_EnableIO as AudioUnitPropertyID,
                    kAudioUnitScope_Output as AudioUnitScope,
                    0,
                    (if output_channels > 0 {
                        &enable
                    } else {
                        &disable
                    }) as *const UInt32 as *const c_void,
                    std::mem::size_of::<UInt32>() as UInt32,
                ),
                "EnableIO(output)",
            )?;

            check(
                AudioUnitSetProperty(
                    unit,
                    kAudioOutputUnitProperty_CurrentDevice as AudioUnitPropertyID,
                    kAudioUnitScope_Global as AudioUnitScope,
                    0,
                    &device_id as *const AudioDeviceID as *const c_void,
                    std::mem::size_of::<AudioDeviceID>() as UInt32,
                ),
                "CurrentDevice",
            )?;

            // Request the hardware buffer size; the device may adjust it.
            let mut requested_frames = buffer_frames as UInt32;
            let frame_address = property_address(
                kAudioDevicePropertyBufferFrameSize,
                kAudioObjectPropertyScopeGlobal,
            );
            let _ = AudioObjectSetPropertyData(
                device_id,
                &frame_address,
                0,
                std::ptr::null(),
                std::mem::size_of::<UInt32>() as UInt32,
                &requested_frames as *const UInt32 as *const c_void,
            );
            let mut size = std::mem::size_of::<UInt32>() as UInt32;
            let _ = AudioObjectGetPropertyData(
                device_id,
                &frame_address,
                0,
                std::ptr::null(),
                &mut size,
                &mut requested_frames as *mut UInt32 as *mut c_void,
            );
            let actual_frames = requested_frames as usize;

            check(
                AudioUnitSetProperty(
                    unit,
                    kAudioUnitProperty_MaximumFramesPerSlice as AudioUnitPropertyID,
                    kAudioUnitScope_Global as AudioUnitScope,
                    0,
                    &requested_frames as *const UInt32 as *const c_void,
                    std::mem::size_of::<UInt32>() as UInt32,
                ),
                "MaximumFramesPerSlice",
            )?;

            if output_channels > 0 {
                let asbd = make_asbd(sample_rate, output_channels);
                check(
                    AudioUnitSetProperty(
                        unit,
                        kAudioUnitProperty_StreamFormat as AudioUnitPropertyID,
                        kAudioUnitScope_Input as AudioUnitScope,
                        0,
                        &asbd as *const AudioStreamBasicDescription as *const c_void,
                        std::mem::size_of::<AudioStreamBasicDescription>() as UInt32,
                    ),
                    "StreamFormat(output)",
                )?;
            }
            if input_channels > 0 {
                let asbd = make_asbd(sample_rate, input_channels);
                check(
                    AudioUnitSetProperty(
                        unit,
                        kAudioUnitProperty_StreamFormat as AudioUnitPropertyID,
                        kAudioUnitScope_Output as AudioUnitScope,
                        1,
                        &asbd as *const AudioStreamBasicDescription as *const c_void,
                        std::mem::size_of::<AudioStreamBasicDescription>() as UInt32,
                    ),
                    "StreamFormat(input)",
                )?;
            }

            let ctx = Box::new(RenderContext {
                unit,
                callback: self.callback.clone(),
                running: self.running.clone(),
                stream_time_bits: self.stream_time_bits.clone(),
                sample_rate,
                input_channels,
                input_enabled: input_channels > 0,
                input_scratch: vec![0.0; actual_frames * input_channels.max(1)],
            });
            let ctx_ptr = Box::into_raw(ctx);

            if output_channels > 0 {
                let render_callback = AURenderCallbackStruct {
                    inputProc: Some(render_proc),
                    inputProcRefCon: ctx_ptr as *mut c_void,
                };
                check(
                    AudioUnitSetProperty(
                        unit,
                        kAudioUnitProperty_SetRenderCallback as AudioUnitPropertyID,
                        kAudioUnitScope_Input as AudioUnitScope,
                        0,
                        &render_callback as *const AURenderCallbackStruct as *const c_void,
                        std::mem::size_of::<AURenderCallbackStruct>() as UInt32,
                    ),
                    "SetRenderCallback",
                )?;
            } else if input_channels > 0 {
                let input_callback = AURenderCallbackStruct {
                    inputProc: Some(input_only_proc),
                    inputProcRefCon: ctx_ptr as *mut c_void,
                };
                check(
                    AudioUnitSetProperty(
                        unit,
                        kAudioOutputUnitProperty_SetInputCallback as AudioUnitPropertyID,
                        kAudioUnitScope_Global as AudioUnitScope,
                        0,
                        &input_callback as *const AURenderCallbackStruct as *const c_void,
                        std::mem::size_of::<AURenderCallbackStruct>() as UInt32,
                    ),
                    "SetInputCallback",
                )?;
            }

            if let Err(e) = check(AudioUnitInitialize(unit), "AudioUnitInitialize") {
                drop(Box::from_raw(ctx_ptr));
                AudioComponentInstanceDispose(unit);
                return Err(e);
            }

            self.unit = unit;
            self.ctx_ptr = ctx_ptr;
            self.sample_rate = sample_rate;
            self.buffer_frames = actual_frames;
            self.number_of_buffers = options.number_of_buffers.max(2);
            self.state = StreamState::Stopped;
            self.stream_time_bits
                .store(0.0f64.to_bits(), Ordering::SeqCst);
            *self.callback.lock().unwrap() = Some(callback);

            Ok(actual_frames)
        }
    }

    fn start(&mut self) -> MKAudioResult<()> {
        if self.state == StreamState::Closed {
            return Err(MKAudioError::InvalidUse("Stream is not open".into()));
        }
        if self.state == StreamState::Running {
            return Err(MKAudioError::InvalidUse("Stream is already running".into()));
        }

        unsafe {
            check(AudioOutputUnitStart(self.unit), "AudioOutputUnitStart")?;
        }
        self.running.store(true, Ordering::SeqCst);
        self.state = StreamState::Running;
        Ok(())
    }

    fn stop(&mut self) -> MKAudioResult<()> {
        if self.state != StreamState::Running {
            return Err(MKAudioError::InvalidUse("Stream is not running".into()));
        }

        unsafe {
            check(AudioOutputUnitStop(self.unit), "AudioOutputUnitStop")?;
        }
        self.running.store(false, Ordering::SeqCst);
        self.state = StreamState::Stopped;
        Ok(())
    }

    fn close(&mut self) {
        if self.state == StreamState::Closed {
            return;
        }
        if self.state == StreamState::Running {
            let _ = self.stop();
        }

        unsafe {
            AudioUnitUninitialize(self.unit);
            AudioComponentInstanceDispose(self.unit);
            if !self.ctx_ptr.is_null() {
                drop(Box::from_raw(self.ctx_ptr));
            }
        }

        self.unit = std::ptr::null_mut();
        self.ctx_ptr = std::ptr::null_mut();
        self.state = StreamState::Closed;
        *self.callback.lock().unwrap() = None;
    }

    fn is_running(&self) -> bool {
        self.state == StreamState::Running
    }

    fn stream_time(&self) -> f64 {
        f64::from_bits(self.stream_time_bits.load(Ordering::SeqCst))
    }

    fn latency_samples(&self) -> usize {
        self.buffer_frames * self.number_of_buffers
    }
}

impl Drop for CoreAudioBackend {
    fn drop(&mut self) {
        self.close();
    }
}
