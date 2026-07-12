//! Manual verification example for the AUv2 host: scans installed Audio
//! Units, loads one real effect, and pushes a block of audio through it.
//!
//! Run with: cargo run --example au_host_test --features au

use mkaudiolibrary::host::{PluginFormat, load, scan_au};
use mkaudiolibrary::processor::AudioIO;

fn main() {
    let found = scan_au();
    println!("Found {} Audio Units", found.len());
    for d in found.iter().take(5) {
        println!(
            "  [{}] {} - {} ({})",
            d.format, d.vendor, d.name, d.category
        );
    }

    let effect = found
        .iter()
        .find(|d| d.category == "Effect")
        .expect("no Effect-type Audio Unit found to test against");
    println!("\nLoading: {} - {}", effect.vendor, effect.name);
    assert_eq!(effect.format, PluginFormat::Au);

    let mut plugin = load(effect).expect("failed to load AU");
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

    let in_channels = plugin.num_inputs();
    let out_channels = plugin.num_outputs().max(1);

    // AudioIO borrows into caller-owned storage rather than allocating its
    // own - own the actual sample memory here, for the life of this block.
    let mut input_storage = vec![vec![0.0f32; 512]; in_channels];
    let mut output_storage = vec![vec![0.0f32; 512]; out_channels];

    // Feed a simple 440Hz sine into every input channel.
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

    println!("Output peak: {:.6}, nonzero: {}", peak, nonzero);
    println!(
        "OK: AU host rendered a real block of audio through '{}'.",
        plugin.name()
    );

    // Also verify parameter enumeration/get/set against whichever scanned
    // Effect actually exposes AU parameters (Doomsday above may not).
    println!("\nSearching for an Effect with exposed parameters...");
    for candidate in found.iter().filter(|d| d.category == "Effect").take(40) {
        let Ok(mut p) = load(candidate) else {
            continue;
        };
        if p.prepare(48000, 512).is_err() {
            continue;
        }
        if p.num_parameters() == 0 {
            continue;
        }

        println!(
            "  {} - {}: {} parameters",
            candidate.vendor,
            candidate.name,
            p.num_parameters()
        );
        for i in 0..p.num_parameters().min(3) {
            let before = p.get_parameter(i);
            p.set_parameter(i, 0.75);
            let after = p.get_parameter(i);
            println!(
                "    [{}] '{}': {:.4} -> set 0.75 -> {:.4}",
                i,
                p.parameter_name(i),
                before,
                after
            );
        }
        break;
    }
}
