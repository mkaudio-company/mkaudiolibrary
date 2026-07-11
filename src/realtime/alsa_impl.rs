//! ALSA backend for Linux.
//!
//! Uses blocking-mode `snd_pcm` read/write in a dedicated per-stream thread,
//! the same fundamental mechanism upstream RtAudio's `RtApiAlsa` uses for
//! its default (non-mmap) path. Devices are enumerated via ALSA's
//! `snd_device_name_hint` (through `alsa::device_name::HintIter`), and
//! since string-keyed ALSA device names have no native integer identity,
//! each is exposed here as a stable hash of its ALSA device name.

use std::collections::hash_map::DefaultHasher;
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use alsa::device_name::HintIter;
use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, ValueOr};

use super::{
    AudioCallback, Backend, DeviceInfo, MKAudioError, MKAudioResult, SampleFormat, StreamOptions,
    StreamParameters, StreamState, StreamStatus, invoke_callback,
};

fn device_key(name: &str) -> usize {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    (hasher.finish() & 0x7fff_ffff_ffff_ffff) as usize
}

struct EnumeratedDevice {
    key: usize,
    name: String,
    desc: String,
    direction: Option<Direction>,
}

fn enumerate_all() -> Vec<EnumeratedDevice> {
    let Ok(iface) = CString::new("pcm") else {
        return Vec::new();
    };
    let Ok(hints) = HintIter::new(None, &iface) else {
        return Vec::new();
    };

    hints
        .filter_map(|hint| {
            let name = hint.name?;
            if name == "null" {
                return None;
            }
            let desc = hint.desc.unwrap_or_else(|| name.clone());
            Some(EnumeratedDevice {
                key: device_key(&name),
                name,
                desc,
                direction: hint.direction,
            })
        })
        .collect()
}

fn find_device(key: usize) -> MKAudioResult<EnumeratedDevice> {
    enumerate_all()
        .into_iter()
        .find(|d| d.key == key)
        .ok_or_else(|| MKAudioError::InvalidDevice(format!("Device {} not found", key)))
}

/// Best-effort channel-count probe: opens the device just long enough to
/// read its hardware parameter ranges, then closes it again. Devices that
/// are busy (already open elsewhere) report 0 rather than failing the whole
/// call, matching how a device mid-use would show up in most audio hosts.
fn probe_channels(name: &str, dir: Direction) -> usize {
    let Ok(pcm) = PCM::new(name, dir, true) else {
        return 0;
    };
    let Ok(hwp) = HwParams::any(&pcm) else {
        return 0;
    };
    hwp.get_channels_max().unwrap_or(0) as usize
}

fn open_configured(
    name: &str,
    dir: Direction,
    sample_rate: usize,
    channels: usize,
    buffer_frames: usize,
) -> MKAudioResult<(PCM, usize)> {
    let pcm = PCM::new(name, dir, false).map_err(|e| {
        MKAudioError::InvalidDevice(format!("failed to open ALSA device '{}': {}", name, e))
    })?;

    {
        let hwp = HwParams::any(&pcm)
            .map_err(|e| MKAudioError::DriverError(format!("HwParams::any failed: {}", e)))?;
        hwp.set_access(Access::RWInterleaved)
            .map_err(|e| MKAudioError::DriverError(format!("set_access failed: {}", e)))?;
        hwp.set_format(Format::float64())
            .map_err(|e| MKAudioError::DriverError(format!("set_format(float64) failed: {}", e)))?;
        hwp.set_channels(channels as u32).map_err(|e| {
            MKAudioError::DriverError(format!("set_channels({}) failed: {}", channels, e))
        })?;
        hwp.set_rate(sample_rate as u32, ValueOr::Nearest)
            .map_err(|e| {
                MKAudioError::DriverError(format!("set_rate({}) failed: {}", sample_rate, e))
            })?;
        hwp.set_period_size_near(buffer_frames as alsa::pcm::Frames, ValueOr::Nearest)
            .map_err(|e| {
                MKAudioError::DriverError(format!("set_period_size_near failed: {}", e))
            })?;
        hwp.set_buffer_size_near((buffer_frames * 4) as alsa::pcm::Frames)
            .map_err(|e| {
                MKAudioError::DriverError(format!("set_buffer_size_near failed: {}", e))
            })?;
        pcm.hw_params(&hwp)
            .map_err(|e| MKAudioError::DriverError(format!("hw_params failed: {}", e)))?;
    }

    let actual_frames = pcm
        .hw_params_current()
        .and_then(|hwp| hwp.get_period_size())
        .map(|f| f as usize)
        .unwrap_or(buffer_frames);

    pcm.prepare()
        .map_err(|e| MKAudioError::DriverError(format!("prepare failed: {}", e)))?;

    Ok((pcm, actual_frames))
}

/// Runs the blocking read/write loop, then hands the PCM handles back to
/// the owning `AlsaBackend` over `return_tx` once it exits.
///
/// `PCM` is `Send` but not `Sync` (per the `alsa` crate), so handles are
/// moved into this thread by value rather than shared behind an `Arc` -
/// they only ever exist on one thread at a time, which `Send` alone permits.
fn audio_thread(
    playback: Option<PCM>,
    capture: Option<PCM>,
    playback_channels: usize,
    capture_channels: usize,
    period_frames: usize,
    sample_rate: usize,
    callback: Arc<Mutex<Option<AudioCallback>>>,
    running: Arc<AtomicBool>,
    stream_time_bits: Arc<AtomicU64>,
    return_tx: std::sync::mpsc::Sender<(Option<PCM>, Option<PCM>)>,
) {
    if let Some(c) = &capture {
        let _ = c.start();
    }

    let capture_io = capture.as_ref().and_then(|p| p.io_f64().ok());
    let playback_io = playback.as_ref().and_then(|p| p.io_f64().ok());

    let mut input_buf = vec![0.0f64; period_frames * capture_channels];
    let mut output_buf = vec![0.0f64; period_frames * playback_channels];

    while running.load(Ordering::SeqCst) {
        if let (Some(io), Some(pcm)) = (&capture_io, &capture) {
            if let Err(e) = io.readi(&mut input_buf) {
                if pcm.try_recover(e, true).is_err() {
                    break;
                }
                continue;
            }
        }

        let stream_time = f64::from_bits(stream_time_bits.load(Ordering::Relaxed));
        invoke_callback(
            &callback,
            &running,
            &mut output_buf,
            &input_buf,
            period_frames,
            stream_time,
            StreamStatus::default(),
        );
        stream_time_bits.store(
            (stream_time + period_frames as f64 / sample_rate as f64).to_bits(),
            Ordering::Relaxed,
        );

        if let (Some(io), Some(pcm)) = (&playback_io, &playback)
            && let Err(e) = io.writei(&output_buf)
            && pcm.try_recover(e, true).is_err()
        {
            break;
        }
    }

    drop(capture_io);
    drop(playback_io);

    if let Some(p) = &playback {
        let _ = p.drain();
    }
    if let Some(c) = &capture {
        let _ = c.drop();
    }
    running.store(false, Ordering::SeqCst);

    let _ = return_tx.send((playback, capture));
}

pub(crate) struct AlsaBackend {
    state: StreamState,
    playback: Option<PCM>,
    capture: Option<PCM>,
    playback_channels: usize,
    capture_channels: usize,
    sample_rate: usize,
    buffer_frames: usize,
    number_of_buffers: usize,
    callback: Arc<Mutex<Option<AudioCallback>>>,
    running: Arc<AtomicBool>,
    stream_time_bits: Arc<AtomicU64>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    return_rx: Option<std::sync::mpsc::Receiver<(Option<PCM>, Option<PCM>)>>,
}

impl AlsaBackend {
    pub fn new() -> Self {
        Self {
            state: StreamState::Closed,
            playback: None,
            capture: None,
            playback_channels: 0,
            capture_channels: 0,
            sample_rate: 44100,
            buffer_frames: 256,
            number_of_buffers: 4,
            callback: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            stream_time_bits: Arc::new(AtomicU64::new(0)),
            thread_handle: None,
            return_rx: None,
        }
    }
}

impl Backend for AlsaBackend {
    fn device_ids(&self) -> Vec<usize> {
        enumerate_all().into_iter().map(|d| d.key).collect()
    }

    fn device_info(&self, device_id: usize) -> MKAudioResult<DeviceInfo> {
        let device = find_device(device_id)?;

        let output_channels = if device.direction != Some(Direction::Capture) {
            probe_channels(&device.name, Direction::Playback)
        } else {
            0
        };
        let input_channels = if device.direction != Some(Direction::Playback) {
            probe_channels(&device.name, Direction::Capture)
        } else {
            0
        };

        Ok(DeviceInfo {
            id: device_id,
            name: device.desc,
            output_channels,
            input_channels,
            duplex_channels: output_channels.min(input_channels),
            is_default_output: device.name == "default",
            is_default_input: device.name == "default",
            sample_rates: vec![44100, 48000, 96000],
            preferred_sample_rate: 48000,
            native_formats: vec![SampleFormat::Float32, SampleFormat::Float64],
        })
    }

    fn default_output_device(&self) -> usize {
        enumerate_all()
            .into_iter()
            .find(|d| d.name == "default")
            .map(|d| d.key)
            .or_else(|| {
                enumerate_all()
                    .into_iter()
                    .find(|d| d.direction != Some(Direction::Capture))
                    .map(|d| d.key)
            })
            .unwrap_or(0)
    }

    fn default_input_device(&self) -> usize {
        enumerate_all()
            .into_iter()
            .find(|d| d.name == "default")
            .map(|d| d.key)
            .or_else(|| {
                enumerate_all()
                    .into_iter()
                    .find(|d| d.direction != Some(Direction::Playback))
                    .map(|d| d.key)
            })
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

        let mut actual_frames = buffer_frames;

        let playback = if let Some(p) = output_params {
            let device = find_device(p.device_id)?;
            let (pcm, frames) = open_configured(
                &device.name,
                Direction::Playback,
                sample_rate,
                p.num_channels,
                buffer_frames,
            )?;
            actual_frames = frames;
            self.playback_channels = p.num_channels;
            Some(pcm)
        } else {
            None
        };

        let capture = if let Some(p) = input_params {
            let device = find_device(p.device_id)?;
            let (pcm, frames) = open_configured(
                &device.name,
                Direction::Capture,
                sample_rate,
                p.num_channels,
                buffer_frames,
            )?;
            if playback.is_none() {
                actual_frames = frames;
            }
            self.capture_channels = p.num_channels;
            Some(pcm)
        } else {
            None
        };

        self.playback = playback;
        self.capture = capture;
        self.sample_rate = sample_rate;
        self.buffer_frames = actual_frames;
        self.number_of_buffers = options.number_of_buffers.max(4);
        self.state = StreamState::Stopped;
        self.stream_time_bits
            .store(0.0f64.to_bits(), Ordering::SeqCst);
        *self.callback.lock().unwrap() = Some(callback);

        Ok(actual_frames)
    }

    fn start(&mut self) -> MKAudioResult<()> {
        if self.state == StreamState::Closed {
            return Err(MKAudioError::InvalidUse("Stream is not open".into()));
        }
        if self.state == StreamState::Running {
            return Err(MKAudioError::InvalidUse("Stream is already running".into()));
        }

        if self.playback.is_none() && self.capture.is_none() {
            return Err(MKAudioError::InvalidUse(
                "Stream has no open PCM devices".into(),
            ));
        }
        // Move the PCM handles into the audio thread; `stop()` gets them
        // back over `return_rx` once that thread exits (see `audio_thread`'s
        // doc comment for why this hand-off, rather than shared `Arc`, is
        // needed).
        let playback = self.playback.take();
        let capture = self.capture.take();

        self.running.store(true, Ordering::SeqCst);
        self.state = StreamState::Running;

        let callback = self.callback.clone();
        let running = self.running.clone();
        let stream_time_bits = self.stream_time_bits.clone();
        let playback_channels = self.playback_channels;
        let capture_channels = self.capture_channels;
        let period_frames = self.buffer_frames;
        let sample_rate = self.sample_rate;

        let (return_tx, return_rx) = std::sync::mpsc::channel();
        self.return_rx = Some(return_rx);

        self.thread_handle = Some(std::thread::spawn(move || {
            audio_thread(
                playback,
                capture,
                playback_channels,
                capture_channels,
                period_frames,
                sample_rate,
                callback,
                running,
                stream_time_bits,
                return_tx,
            );
        }));

        Ok(())
    }

    fn stop(&mut self) -> MKAudioResult<()> {
        if self.state != StreamState::Running {
            return Err(MKAudioError::InvalidUse("Stream is not running".into()));
        }

        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        self.state = StreamState::Stopped;

        if let Some(rx) = self.return_rx.take()
            && let Ok((playback, capture)) = rx.recv()
        {
            self.playback = playback;
            self.capture = capture;
        }

        // Re-prepare so a subsequent `start()` can spawn a fresh thread
        // against the same still-open devices.
        if let Some(p) = &self.playback {
            let _ = p.prepare();
        }
        if let Some(c) = &self.capture {
            let _ = c.prepare();
        }

        Ok(())
    }

    fn close(&mut self) {
        if self.state == StreamState::Closed {
            return;
        }
        if self.state == StreamState::Running {
            let _ = self.stop();
        }

        self.playback = None;
        self.capture = None;
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

impl Drop for AlsaBackend {
    fn drop(&mut self) {
        self.close();
    }
}
