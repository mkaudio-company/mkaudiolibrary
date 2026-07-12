//! WASAPI backend for Windows.
//!
//! Uses event-driven shared-mode `IAudioClient` streams — the same
//! mechanism upstream RtAudio's `RtApiWasapi` uses. All COM interaction
//! (device activation, stream initialization, and the render/capture loop)
//! happens on one dedicated MTA thread per open stream, so no COM interface
//! pointer ever needs to cross a thread boundary; control calls from
//! `Realtime` communicate with that thread over a small command channel.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU32, Ordering},
};

use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Media::Audio::*;
use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
use windows::Win32::System::Com::StructuredStorage::PropVariantClear;
use windows::Win32::System::Com::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::Win32::System::Variant::VT_LPWSTR;
use windows::core::PCWSTR;

use super::{
    AudioCallback, Backend, DeviceInfo, MKAudioError, MKAudioResult, SampleFormat, StreamOptions,
    StreamParameters, StreamState, StreamStatus, invoke_callback,
};

fn check<T>(result: windows::core::Result<T>, what: &str) -> MKAudioResult<T> {
    result.map_err(|e| MKAudioError::DriverError(format!("{} failed ({})", what, e)))
}

thread_local! { static COM_INITIALIZED : std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

/// Ensure COM is initialized (MTA) on the calling thread. Deliberately never
/// paired with `CoUninitialize` - the apartment is released when the thread
/// exits, matching common practice for library code that doesn't own the
/// calling thread's lifetime.
fn ensure_com_initialized() {
    COM_INITIALIZED.with(|flag| {
        if !flag.get() {
            unsafe {
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            }
            flag.set(true);
        }
    });
}

fn device_key(id_str: &str) -> usize {
    let mut hasher = DefaultHasher::new();
    id_str.hash(&mut hasher);
    (hasher.finish() & 0x7fff_ffff_ffff_ffff) as usize
}

fn device_friendly_name(device: &IMMDevice) -> String {
    unsafe {
        let Ok(store) = device.OpenPropertyStore(STGM_READ) else {
            return String::new();
        };
        let Ok(mut variant) = store.GetValue(&PKEY_Device_FriendlyName) else {
            return String::new();
        };

        let name = if variant.Anonymous.Anonymous.vt == VT_LPWSTR {
            variant
                .Anonymous
                .Anonymous
                .Anonymous
                .pwszVal
                .to_string()
                .unwrap_or_default()
        } else {
            String::new()
        };

        let _ = PropVariantClear(&mut variant);
        name
    }
}

/// Native channel count and sample rate for a device, taken from its mix format.
fn device_mix_format(device: &IMMDevice) -> (usize, usize) {
    unsafe {
        let Ok(client): windows::core::Result<IAudioClient> = device.Activate(CLSCTX_ALL, None)
        else {
            return (0, 44100);
        };
        let Ok(format_ptr) = client.GetMixFormat() else {
            return (0, 44100);
        };
        let channels = (*format_ptr).nChannels as usize;
        let sample_rate = (*format_ptr).nSamplesPerSec as usize;
        windows::Win32::System::Com::CoTaskMemFree(Some(format_ptr as *const core::ffi::c_void));
        (channels, sample_rate)
    }
}

struct EnumeratedDevice {
    key: usize,
    id_str: String,
    flow: EDataFlow,
}

fn enumerate_all(enumerator: &IMMDeviceEnumerator) -> Vec<EnumeratedDevice> {
    let mut devices = Vec::new();

    for &flow in &[eRender, eCapture] {
        let Ok(collection) = (unsafe { enumerator.EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE) })
        else {
            continue;
        };
        let Ok(count) = (unsafe { collection.GetCount() }) else {
            continue;
        };

        for i in 0..count {
            let Ok(device) = (unsafe { collection.Item(i) }) else {
                continue;
            };
            let Ok(id_pwstr) = (unsafe { device.GetId() }) else {
                continue;
            };
            let Ok(id_str) = (unsafe { id_pwstr.to_string() }) else {
                continue;
            };
            devices.push(EnumeratedDevice {
                key: device_key(&id_str),
                id_str,
                flow,
            });
        }
    }

    devices
}

fn find_device(
    enumerator: &IMMDeviceEnumerator,
    key: usize,
) -> MKAudioResult<(IMMDevice, EDataFlow)> {
    for entry in enumerate_all(enumerator) {
        if entry.key == key {
            let id_wide: Vec<u16> = entry
                .id_str
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect();
            let device = check(
                unsafe { enumerator.GetDevice(PCWSTR::from_raw(id_wide.as_ptr())) },
                "GetDevice",
            )?;
            return Ok((device, entry.flow));
        }
    }
    Err(MKAudioError::InvalidDevice(format!(
        "Device {} not found",
        key
    )))
}

fn make_waveformat(sample_rate: usize, channels: usize) -> WAVEFORMATEXTENSIBLE {
    let bytes_per_frame = (channels * std::mem::size_of::<f32>()) as u16;
    WAVEFORMATEXTENSIBLE {
        Format: WAVEFORMATEX {
            wFormatTag: 0xFFFE, // WAVE_FORMAT_EXTENSIBLE
            nChannels: channels as u16,
            nSamplesPerSec: sample_rate as u32,
            nAvgBytesPerSec: (sample_rate * channels * std::mem::size_of::<f32>()) as u32,
            nBlockAlign: bytes_per_frame,
            wBitsPerSample: 64,
            cbSize: 22,
        },
        Samples: WAVEFORMATEXTENSIBLE_0 {
            wValidBitsPerSample: 64,
        },
        dwChannelMask: 0,
        SubFormat: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
    }
}

struct RenderResources {
    client: IAudioClient,
    render: IAudioRenderClient,
    event: HANDLE,
    buffer_frames: u32,
    channels: usize,
}
struct CaptureResources {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    event: HANDLE,
    channels: usize,
}

impl Drop for RenderResources {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}
impl Drop for CaptureResources {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

fn setup_render(
    device: &IMMDevice,
    sample_rate: usize,
    buffer_frames: usize,
    channels: usize,
) -> windows::core::Result<RenderResources> {
    unsafe {
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let format = make_waveformat(sample_rate, channels);
        let hns_duration = (buffer_frames as i64 * 10_000_000) / sample_rate as i64;
        client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            hns_duration,
            0,
            &format.Format,
            None,
        )?;

        let event = CreateEventW(None, false, false, PCWSTR::null())?;
        client.SetEventHandle(event)?;

        let actual_buffer_frames = client.GetBufferSize()?;
        let render: IAudioRenderClient = client.GetService()?;

        Ok(RenderResources {
            client,
            render,
            event,
            buffer_frames: actual_buffer_frames,
            channels,
        })
    }
}

fn setup_capture(
    device: &IMMDevice,
    sample_rate: usize,
    buffer_frames: usize,
    channels: usize,
) -> windows::core::Result<CaptureResources> {
    unsafe {
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let format = make_waveformat(sample_rate, channels);
        let hns_duration = (buffer_frames as i64 * 10_000_000) / sample_rate as i64;
        client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            hns_duration,
            0,
            &format.Format,
            None,
        )?;

        let event = CreateEventW(None, false, false, PCWSTR::null())?;
        client.SetEventHandle(event)?;

        let capture: IAudioCaptureClient = client.GetService()?;

        Ok(CaptureResources {
            client,
            capture,
            event,
            channels,
        })
    }
}

enum ThreadCommand {
    Start,
    Stop,
    Shutdown,
}

struct ThreadSetup {
    output_device_key: Option<usize>,
    output_channels: usize,
    input_device_key: Option<usize>,
    input_channels: usize,
    sample_rate: usize,
    buffer_frames: usize,
}

fn audio_thread(
    setup: ThreadSetup,
    callback: Arc<Mutex<Option<AudioCallback>>>,
    running: Arc<AtomicBool>,
    stream_time_bits: Arc<AtomicU32>,
    cmd_rx: Receiver<ThreadCommand>,
    setup_tx: Sender<MKAudioResult<usize>>,
) {
    ensure_com_initialized();

    let build = || -> windows::core::Result<(Option<RenderResources>, Option<CaptureResources>)> {
        let enumerator: IMMDeviceEnumerator =
            unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? };

        let render = if let Some(key) = setup.output_device_key {
            let (device, _) = find_device(&enumerator, key)
                .map_err(|_| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
            Some(setup_render(
                &device,
                setup.sample_rate,
                setup.buffer_frames,
                setup.output_channels,
            )?)
        } else {
            None
        };

        let capture = if let Some(key) = setup.input_device_key {
            let (device, _) = find_device(&enumerator, key)
                .map_err(|_| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
            Some(setup_capture(
                &device,
                setup.sample_rate,
                setup.buffer_frames,
                setup.input_channels,
            )?)
        } else {
            None
        };

        Ok((render, capture))
    };

    let (render, capture) = match build() {
        Ok(v) => v,
        Err(e) => {
            let _ = setup_tx.send(Err(MKAudioError::DriverError(format!(
                "WASAPI stream setup failed: {}",
                e
            ))));
            unsafe {
                CoUninitialize();
            }
            return;
        }
    };

    let actual_frames = render
        .as_ref()
        .map(|r| r.buffer_frames as usize)
        .unwrap_or(setup.buffer_frames);
    if setup_tx.send(Ok(actual_frames)).is_err() {
        unsafe {
            CoUninitialize();
        }
        return;
    }

    // Block until the first Start command (or an early shutdown/close).
    loop {
        match cmd_rx.recv() {
            Ok(ThreadCommand::Start) => break,
            Ok(ThreadCommand::Shutdown) | Err(_) => {
                unsafe {
                    CoUninitialize();
                }
                return;
            }
            Ok(ThreadCommand::Stop) => continue,
        }
    }

    unsafe {
        if let Some(r) = &render {
            let _ = r.client.Start();
        }
        if let Some(c) = &capture {
            let _ = c.client.Start();
        }
    }
    running.store(true, Ordering::SeqCst);

    let input_channels = capture.as_ref().map(|c| c.channels).unwrap_or(0);
    let mut input_ring: Vec<f32> = Vec::new();

    'main: loop {
        loop {
            match cmd_rx.try_recv() {
                Ok(ThreadCommand::Stop) => {
                    unsafe {
                        if let Some(r) = &render {
                            let _ = r.client.Stop();
                        }
                        if let Some(c) = &capture {
                            let _ = c.client.Stop();
                        }
                    }
                    running.store(false, Ordering::SeqCst);

                    loop {
                        match cmd_rx.recv() {
                            Ok(ThreadCommand::Start) => {
                                unsafe {
                                    if let Some(r) = &render {
                                        let _ = r.client.Start();
                                    }
                                    if let Some(c) = &capture {
                                        let _ = c.client.Start();
                                    }
                                }
                                running.store(true, Ordering::SeqCst);
                                break;
                            }
                            Ok(ThreadCommand::Shutdown) | Err(_) => break 'main,
                            Ok(ThreadCommand::Stop) => continue,
                        }
                    }
                }
                Ok(ThreadCommand::Shutdown) => break 'main,
                Ok(ThreadCommand::Start) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break 'main,
            }
        }

        // Drain any available capture packets into the input ring buffer.
        if let Some(c) = &capture {
            unsafe {
                loop {
                    let next = c.capture.GetNextPacketSize().unwrap_or(0);
                    if next == 0 {
                        break;
                    }

                    let mut data_ptr: *mut u8 = std::ptr::null_mut();
                    let mut frames_available = 0u32;
                    let mut flags = 0u32;
                    if c.capture
                        .GetBuffer(&mut data_ptr, &mut frames_available, &mut flags, None, None)
                        .is_err()
                    {
                        break;
                    }

                    let count = frames_available as usize * c.channels;
                    if (flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0 {
                        input_ring.extend(std::iter::repeat_n(0.0, count));
                    } else {
                        let src = std::slice::from_raw_parts(data_ptr as *const f32, count);
                        input_ring.extend_from_slice(src);
                    }

                    let _ = c.capture.ReleaseBuffer(frames_available);
                }
            }
        }

        if let Some(r) = &render {
            unsafe {
                let padding = r.client.GetCurrentPadding().unwrap_or(0);
                let available = r.buffer_frames.saturating_sub(padding);
                if available == 0 {
                    let _ = WaitForSingleObject(r.event, 20);
                    continue;
                }

                let Ok(out_ptr) = r.render.GetBuffer(available) else {
                    continue;
                };
                let frames = available as usize;
                let out_slice =
                    std::slice::from_raw_parts_mut(out_ptr as *mut f32, frames * r.channels);

                let needed_in = frames * input_channels;
                let input_slice: Vec<f32> = if input_channels > 0 {
                    if input_ring.len() >= needed_in {
                        input_ring.drain(0..needed_in).collect()
                    } else {
                        vec![0.0; needed_in]
                    }
                } else {
                    Vec::new()
                };

                let stream_time = f32::from_bits(stream_time_bits.load(Ordering::Relaxed));
                invoke_callback(
                    &callback,
                    &running,
                    out_slice,
                    &input_slice,
                    frames,
                    stream_time,
                    StreamStatus::default(),
                );
                stream_time_bits.store(
                    (stream_time + frames as f32 / setup.sample_rate as f32).to_bits(),
                    Ordering::Relaxed,
                );

                let _ = r.render.ReleaseBuffer(available, 0);
                let _ = WaitForSingleObject(r.event, 2000);
            }
        } else if let Some(c) = &capture {
            let chunk_frames = setup.buffer_frames;
            let needed = chunk_frames * input_channels;
            if input_ring.len() >= needed {
                let chunk: Vec<f32> = input_ring.drain(0..needed).collect();
                let stream_time = f32::from_bits(stream_time_bits.load(Ordering::Relaxed));
                invoke_callback(
                    &callback,
                    &running,
                    &mut [],
                    &chunk,
                    chunk_frames,
                    stream_time,
                    StreamStatus::default(),
                );
                stream_time_bits.store(
                    (stream_time + chunk_frames as f32 / setup.sample_rate as f32).to_bits(),
                    Ordering::Relaxed,
                );
            } else {
                unsafe {
                    let _ = WaitForSingleObject(c.event, 20);
                }
            }
        }
    }

    unsafe {
        if let Some(r) = &render {
            let _ = r.client.Stop();
        }
        if let Some(c) = &capture {
            let _ = c.client.Stop();
        }
    }
    running.store(false, Ordering::SeqCst);

    drop(render);
    drop(capture);
    unsafe {
        CoUninitialize();
    }
}

struct StreamHandle {
    cmd_tx: Sender<ThreadCommand>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

pub(crate) struct WasapiBackend {
    state: StreamState,
    handle: Option<StreamHandle>,
    callback: Arc<Mutex<Option<AudioCallback>>>,
    running: Arc<AtomicBool>,
    stream_time_bits: Arc<AtomicU32>,
    buffer_frames: usize,
    number_of_buffers: usize,
}

impl WasapiBackend {
    pub fn new() -> Self {
        Self {
            state: StreamState::Closed,
            handle: None,
            callback: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            stream_time_bits: Arc::new(AtomicU32::new(0)),
            buffer_frames: 256,
            number_of_buffers: 2,
        }
    }
}

impl Backend for WasapiBackend {
    fn device_ids(&self) -> Vec<usize> {
        ensure_com_initialized();
        let Ok(enumerator): windows::core::Result<IMMDeviceEnumerator> =
            (unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) })
        else {
            return Vec::new();
        };
        enumerate_all(&enumerator)
            .into_iter()
            .map(|d| d.key)
            .collect()
    }

    fn device_info(&self, device_id: usize) -> MKAudioResult<DeviceInfo> {
        ensure_com_initialized();
        let enumerator: IMMDeviceEnumerator = check(
            unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) },
            "CoCreateInstance(MMDeviceEnumerator)",
        )?;
        let (device, flow) = find_device(&enumerator, device_id)?;

        let name = device_friendly_name(&device);
        let (channels, sample_rate) = device_mix_format(&device);
        let is_output = flow == eRender;

        let default_output = unsafe { enumerator.GetDefaultAudioEndpoint(eRender, eConsole) }
            .ok()
            .and_then(|d| unsafe { d.GetId() }.ok())
            .and_then(|id| unsafe { id.to_string() }.ok())
            .map(|s| device_key(&s));
        let default_input = unsafe { enumerator.GetDefaultAudioEndpoint(eCapture, eConsole) }
            .ok()
            .and_then(|d| unsafe { d.GetId() }.ok())
            .and_then(|id| unsafe { id.to_string() }.ok())
            .map(|s| device_key(&s));

        Ok(DeviceInfo {
            id: device_id,
            name,
            output_channels: if is_output { channels } else { 0 },
            input_channels: if !is_output { channels } else { 0 },
            duplex_channels: 0,
            is_default_output: default_output == Some(device_id),
            is_default_input: default_input == Some(device_id),
            sample_rates: vec![sample_rate],
            preferred_sample_rate: sample_rate,
            native_formats: vec![SampleFormat::Float32, SampleFormat::Float64],
        })
    }

    fn default_output_device(&self) -> usize {
        ensure_com_initialized();
        let Ok(enumerator): windows::core::Result<IMMDeviceEnumerator> =
            (unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) })
        else {
            return 0;
        };
        unsafe { enumerator.GetDefaultAudioEndpoint(eRender, eConsole) }
            .ok()
            .and_then(|d| unsafe { d.GetId() }.ok())
            .and_then(|id| unsafe { id.to_string() }.ok())
            .map(|s| device_key(&s))
            .unwrap_or(0)
    }

    fn default_input_device(&self) -> usize {
        ensure_com_initialized();
        let Ok(enumerator): windows::core::Result<IMMDeviceEnumerator> =
            (unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) })
        else {
            return 0;
        };
        unsafe { enumerator.GetDefaultAudioEndpoint(eCapture, eConsole) }
            .ok()
            .and_then(|d| unsafe { d.GetId() }.ok())
            .and_then(|id| unsafe { id.to_string() }.ok())
            .map(|s| device_key(&s))
            .unwrap_or(0)
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

        let setup = ThreadSetup {
            output_device_key: output_params.map(|p| p.device_id),
            output_channels: output_params.map(|p| p.num_channels).unwrap_or(0),
            input_device_key: input_params.map(|p| p.device_id),
            input_channels: input_params.map(|p| p.num_channels).unwrap_or(0),
            sample_rate,
            buffer_frames,
        };

        let (cmd_tx, cmd_rx) = channel::<ThreadCommand>();
        let (setup_tx, setup_rx) = channel::<MKAudioResult<usize>>();

        let callback_arc = self.callback.clone();
        let running = self.running.clone();
        let stream_time_bits = self.stream_time_bits.clone();

        *self.callback.lock().unwrap() = Some(callback);

        let join_handle = std::thread::spawn(move || {
            audio_thread(
                setup,
                callback_arc,
                running,
                stream_time_bits,
                cmd_rx,
                setup_tx,
            )
        });

        let actual_frames = match setup_rx.recv() {
            Ok(Ok(frames)) => frames,
            Ok(Err(e)) => {
                let _ = join_handle.join();
                *self.callback.lock().unwrap() = None;
                return Err(e);
            }
            Err(_) => {
                let _ = join_handle.join();
                *self.callback.lock().unwrap() = None;
                return Err(MKAudioError::DriverError(
                    "WASAPI setup thread exited unexpectedly".into(),
                ));
            }
        };

        self.handle = Some(StreamHandle {
            cmd_tx,
            join_handle: Some(join_handle),
        });
        self.buffer_frames = actual_frames;
        self.number_of_buffers = options.number_of_buffers.max(2);
        self.state = StreamState::Stopped;
        self.stream_time_bits
            .store(0.0f32.to_bits(), Ordering::SeqCst);

        Ok(actual_frames)
    }

    fn start(&mut self) -> MKAudioResult<()> {
        if self.state == StreamState::Closed {
            return Err(MKAudioError::InvalidUse("Stream is not open".into()));
        }
        if self.state == StreamState::Running {
            return Err(MKAudioError::InvalidUse("Stream is already running".into()));
        }

        if let Some(handle) = &self.handle {
            let _ = handle.cmd_tx.send(ThreadCommand::Start);
        }
        self.state = StreamState::Running;
        Ok(())
    }

    fn stop(&mut self) -> MKAudioResult<()> {
        if self.state != StreamState::Running {
            return Err(MKAudioError::InvalidUse("Stream is not running".into()));
        }

        if let Some(handle) = &self.handle {
            let _ = handle.cmd_tx.send(ThreadCommand::Stop);
        }
        self.state = StreamState::Stopped;
        Ok(())
    }

    fn close(&mut self) {
        if self.state == StreamState::Closed {
            return;
        }

        if let Some(mut handle) = self.handle.take() {
            let _ = handle.cmd_tx.send(ThreadCommand::Shutdown);
            if let Some(join_handle) = handle.join_handle.take() {
                let _ = join_handle.join();
            }
        }

        self.state = StreamState::Closed;
        *self.callback.lock().unwrap() = None;
    }

    fn is_running(&self) -> bool {
        self.state == StreamState::Running
    }

    fn stream_time(&self) -> f32 {
        f32::from_bits(self.stream_time_bits.load(Ordering::SeqCst))
    }

    fn latency_samples(&self) -> usize {
        self.buffer_frames * self.number_of_buffers
    }
}

impl Drop for WasapiBackend {
    fn drop(&mut self) {
        self.close();
    }
}
