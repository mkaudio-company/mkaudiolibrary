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

    let channels = plugin.num_outputs().max(1);
    let mut audio = AudioIO::new(plugin.num_inputs(), channels, 0, 0, 512);

    // Feed a simple 440Hz sine into every input channel.
    for ch in 0..audio.input.len() {
        let mut guard = audio.input[ch].write();
        for (i, sample) in guard.iter_mut().enumerate() {
            *sample = (i as f64 / 48000.0 * 440.0 * std::f64::consts::TAU).sin() * 0.25;
        }
    }

    plugin.process(&mut audio);

    let mut peak = 0.0f64;
    let mut nonzero = false;
    for ch in 0..audio.output.len() {
        let guard = audio.output[ch].read();
        for &s in guard.iter() {
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
