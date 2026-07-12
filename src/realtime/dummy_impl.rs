//! Dummy backend: generates silence on a wall-clock-paced thread.
//!
//! Used for `Api::Dummy` (explicit, e.g. for tests/CI with no audio hardware)
//! and as the fallback when an `Api` variant has no real backend compiled in
//! for the current platform (e.g. requesting `Api::Jack` on macOS).

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU32, Ordering},
};

use super::{
    AudioCallback, Backend, DeviceInfo, MKAudioError, MKAudioResult, StreamOptions,
    StreamParameters, StreamState, StreamStatus, invoke_callback,
};

pub(crate) struct DummyBackend {
    state: StreamState,
    sample_rate: usize,
    buffer_frames: usize,
    output_channels: usize,
    input_channels: usize,
    number_of_buffers: usize,
    callback: Arc<Mutex<Option<AudioCallback>>>,
    running: Arc<AtomicBool>,
    stream_time_bits: Arc<AtomicU32>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
}

impl DummyBackend {
    pub fn new() -> Self {
        Self {
            state: StreamState::Closed,
            sample_rate: 44100,
            buffer_frames: 256,
            output_channels: 0,
            input_channels: 0,
            number_of_buffers: 2,
            callback: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            stream_time_bits: Arc::new(AtomicU32::new(0)),
            thread_handle: None,
        }
    }
}

impl Backend for DummyBackend {
    fn device_ids(&self) -> Vec<usize> {
        vec![0]
    }

    fn device_info(&self, device_id: usize) -> MKAudioResult<DeviceInfo> {
        if device_id != 0 {
            return Err(MKAudioError::InvalidDevice(format!(
                "Device {} not found",
                device_id
            )));
        }
        Ok(DeviceInfo {
            id: 0,
            name: String::from("Dummy Audio Device"),
            output_channels: 2,
            input_channels: 2,
            duplex_channels: 2,
            is_default_output: true,
            is_default_input: true,
            sample_rates: vec![44100, 48000, 96000],
            preferred_sample_rate: 44100,
            native_formats: vec![super::SampleFormat::Float32, super::SampleFormat::Float64],
        })
    }

    fn default_output_device(&self) -> usize {
        0
    }
    fn default_input_device(&self) -> usize {
        0
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

        self.sample_rate = sample_rate;
        self.buffer_frames = buffer_frames;
        self.output_channels = output_params.map(|p| p.num_channels).unwrap_or(0);
        self.input_channels = input_params.map(|p| p.num_channels).unwrap_or(0);
        self.number_of_buffers = options.number_of_buffers.max(2);
        *self.callback.lock().unwrap() = Some(callback);
        self.state = StreamState::Stopped;

        Ok(buffer_frames)
    }

    fn start(&mut self) -> MKAudioResult<()> {
        if self.state == StreamState::Closed {
            return Err(MKAudioError::InvalidUse("Stream is not open".into()));
        }
        if self.state == StreamState::Running {
            return Err(MKAudioError::InvalidUse("Stream is already running".into()));
        }

        self.state = StreamState::Running;
        self.running.store(true, Ordering::SeqCst);
        self.stream_time_bits
            .store(0.0f32.to_bits(), Ordering::SeqCst);

        let callback = self.callback.clone();
        let running = self.running.clone();
        let stream_time_bits = self.stream_time_bits.clone();
        let sample_rate = self.sample_rate;
        let buffer_frames = self.buffer_frames;
        let output_channels = self.output_channels;
        let input_channels = self.input_channels;

        self.thread_handle = Some(std::thread::spawn(move || {
            let frame_duration =
                std::time::Duration::from_secs_f32(buffer_frames as f32 / sample_rate as f32);
            let input = vec![0.0f32; buffer_frames * input_channels];
            let mut output = vec![0.0f32; buffer_frames * output_channels];

            while running.load(Ordering::SeqCst) {
                let stream_time = f32::from_bits(stream_time_bits.load(Ordering::SeqCst));
                invoke_callback(
                    &callback,
                    &running,
                    &mut output,
                    &input,
                    buffer_frames,
                    stream_time,
                    StreamStatus::default(),
                );
                stream_time_bits.store(
                    (stream_time + buffer_frames as f32 / sample_rate as f32).to_bits(),
                    Ordering::SeqCst,
                );
                std::thread::sleep(frame_duration);
            }
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

        Ok(())
    }

    fn close(&mut self) {
        if self.state == StreamState::Running {
            let _ = self.stop();
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
