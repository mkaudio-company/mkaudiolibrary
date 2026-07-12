//! Manual verification example for the VST3 host: scans a real VST3
//! directory, loads one plugin, and pushes a block of audio through it.
//!
//! Run with: cargo run --example vst3_host_test --features vst3

use std::path::Path;

use mkaudiolibrary::host::{load, scan_vst3};
use mkaudiolibrary::processor::AudioIO;

fn main() {
    let dir = Path::new("/Library/Audio/Plug-Ins/VST3");
    let found = scan_vst3(dir);
    println!("Found {} VST3 classes in {}", found.len(), dir.display());
    for d in found.iter().take(8) {
        println!("  [{}] {} - {}", d.format, d.vendor, d.name);
    }

    let descriptor = found
        .first()
        .expect("no VST3 plugins found to test against");
    println!("\nLoading: {} - {}", descriptor.vendor, descriptor.name);

    let mut plugin = load(descriptor).expect("failed to load VST3 plugin");
    println!(
        "Loaded '{}' by '{}': {} in / {} out, {} parameters",
        plugin.name(),
        plugin.vendor(),
        plugin.num_inputs(),
        plugin.num_outputs(),
        plugin.num_parameters()
    );

    plugin.prepare(48000, 512).expect("prepare failed");
    plugin.set_active(true).expect("set_active failed");

    for i in 0..plugin.num_parameters().min(5) {
        println!(
            "  param[{}] '{}' = {:.4}",
            i,
            plugin.parameter_name(i),
            plugin.get_parameter(i)
        );
    }

    let in_channels = plugin.num_inputs().max(1);
    let out_channels = plugin.num_outputs().max(1);

    // AudioIO borrows into caller-owned storage rather than allocating its
    // own - own the actual sample memory here, for the life of this block.
    let mut input_storage = vec![vec![0.0f32; 512]; in_channels];
    let mut output_storage = vec![vec![0.0f32; 512]; out_channels];

    for channel in &mut input_storage {
        for (i, sample) in channel.iter_mut().enumerate() {
            *sample = (i as f32 / 48000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.25;
        }
    }

    let input: Vec<&[f32]> = input_storage.iter().map(Vec::as_slice).collect();
    let mut output: Vec<&mut [f32]> = output_storage.iter_mut().map(Vec::as_mut_slice).collect();
    let mut audio = AudioIO::new(Some(&input), &mut output, None, None);

    plugin.process(&mut audio);

    let mut peak = 0.0f32;
    let mut nonzero = false;
    for channel in &output_storage {
        for &s in channel.iter() {
            if s != 0.0 {
                nonzero = true;
            }
            peak = peak.max(s.abs());
        }
    }

    println!("\nOutput peak: {:.6}, nonzero: {}", peak, nonzero);
    println!(
        "OK: VST3 host rendered a real block of audio through '{}'.",
        plugin.name()
    );
}
