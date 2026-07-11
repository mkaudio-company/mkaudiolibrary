//! Manual verification example for the CoreAudio realtime backend: plays a
//! brief, quiet 440Hz sine tone through the default output device.
//!
//! Run with: cargo run --example realtime_tone --features realtime

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mkaudiolibrary::realtime::{Realtime, StreamParameters, StreamStatus};

fn main() {
    let mut audio = Realtime::new(None).expect("failed to create Realtime");
    println!("Using API: {}", audio.get_current_api());

    for id in audio.get_device_ids() {
        if let Ok(info) = audio.get_device_info(id) {
            println!(
                "  device {}: {} (in:{} out:{}{}{})",
                info.id,
                info.name,
                info.input_channels,
                info.output_channels,
                if info.is_default_output {
                    " [default out]"
                } else {
                    ""
                },
                if info.is_default_input {
                    " [default in]"
                } else {
                    ""
                }
            );
        }
    }

    let output_device = audio.get_default_output_device();
    let output_params = StreamParameters {
        device_id: output_device,
        num_channels: 2,
        first_channel: 0,
    };

    let sample_rate = 48000usize;
    let frame_counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = frame_counter.clone();

    let mut phase = 0.0f64;
    let freq = 440.0;
    let amplitude = 0.05; // quiet on purpose

    let callback = Box::new(
        move |output: &mut [f64],
              _input: &[f64],
              frames: usize,
              _time: f64,
              _status: StreamStatus|
              -> i32 {
            let channels = output.len() / frames.max(1);
            for frame in 0..frames {
                let sample = (phase * 2.0 * std::f64::consts::PI).sin() * amplitude;
                phase += freq / sample_rate as f64;
                if phase >= 1.0 {
                    phase -= 1.0;
                }
                for ch in 0..channels {
                    output[frame * channels + ch] = sample;
                }
            }
            counter_clone.fetch_add(frames, Ordering::Relaxed);
            0
        },
    );

    let actual_buffer_size = audio
        .open_stream(Some(&output_params), None, sample_rate, 256, callback, None)
        .expect("failed to open stream");
    println!(
        "Opened stream with buffer size {} (requested 256)",
        actual_buffer_size
    );

    audio.start_stream().expect("failed to start stream");
    println!("Streaming for 0.4s...");
    std::thread::sleep(std::time::Duration::from_millis(400));
    audio.stop_stream().expect("failed to stop stream");
    audio.close_stream();

    let rendered = frame_counter.load(Ordering::Relaxed);
    println!(
        "Rendered {} frames (~{:.3}s of audio)",
        rendered,
        rendered as f64 / sample_rate as f64
    );
    assert!(
        rendered > 0,
        "no frames were rendered - the callback never fired"
    );
    println!("OK: real CoreAudio callback fired and rendered audio.");
}
