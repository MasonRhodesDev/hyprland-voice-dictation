//! Smoke test for the native PipeWire backend against a live daemon.
//!
//! Ignored by default because it needs a running PipeWire instance. Run with:
//!   cargo test -p dictation-engine --test pipewire_smoke -- --ignored
#![cfg(feature = "pipewire")]

use dictation_engine::audio_backend::pipewire_backend::PipewireBackend;
use dictation_engine::audio_backend::AudioBackendFactory;

#[test]
#[ignore = "requires a running PipeWire daemon"]
fn connects_and_enumerates_sources() {
    assert!(PipewireBackend::is_available(), "PipeWire daemon not reachable");

    let devices = PipewireBackend::list_devices().expect("enumeration failed");
    // Even with no microphones, the backend reports a synthetic default entry.
    assert!(!devices.is_empty(), "expected at least one device entry");
    for d in &devices {
        println!("device: {} ({})", d.name, d.description);
    }
}
