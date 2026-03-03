use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;
use serde_json::Value;
use schema_tui::SchemaTUIBuilder;
use zbus::Connection;

mod utils;

const STATE_FILE: &str = "/tmp/voice-dictation-state";
const MEDIA_STATE_FILE: &str = "/tmp/voice-dictation-media-state";
const DBUS_SERVICE_NAME: &str = "com.voicedictation.Daemon";
const DBUS_OBJECT_PATH: &str = "/com/voicedictation/Control";
const DBUS_INTERFACE_NAME: &str = "com.voicedictation.Control";

#[derive(Parser)]
#[command(name = "voice-dictation")]
#[command(about = "Voice dictation system with Parakeet speech recognition", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Start the dictation engine daemon")]
    Daemon,
    #[command(about = "Start recording session")]
    Start,
    #[command(about = "Stop recording session")]
    Stop,
    #[command(about = "Confirm and finalize transcription")]
    Confirm,
    #[command(about = "Toggle recording (start if stopped, confirm if recording)")]
    Toggle,
    #[command(about = "Show current status")]
    Status,
    #[command(about = "Open configuration TUI")]
    Config,
    #[command(about = "List available models")]
    ListModels,
    #[command(about = "List available preview (fast) models")]
    ListPreviewModels {
        #[arg(default_value = "en")]
        language: String,
    },
    #[command(about = "List available final (accurate) models")]
    ListFinalModels {
        #[arg(default_value = "en")]
        language: String,
    },
    #[command(about = "List available audio input devices")]
    ListAudioDevices,
    #[command(about = "Debug recording tools (requires VOICE_DICTATION_DEBUG_AUDIO=1)")]
    Debug {
        #[command(subcommand)]
        command: DebugCommands,
    },
    #[command(about = "Show audio backend diagnostics and configuration")]
    Diagnose,
}

#[derive(Subcommand)]
enum DebugCommands {
    #[command(about = "List debug recordings in /tmp/voice-dictation-debug")]
    List,
    #[command(about = "Play a debug recording WAV file")]
    Play {
        #[arg(help = "WAV filename to play (from 'debug list' output)")]
        filename: String,
    },
}

fn get_state() -> String {
    fs::read_to_string(STATE_FILE).unwrap_or_else(|_| "stopped".to_string()).trim().to_string()
}

fn set_state(state: &str) -> std::io::Result<()> {
    fs::write(STATE_FILE, state)
}

async fn call_dbus_method(method: &str) -> Result<(), Box<dyn std::error::Error>> {
    let connection = Connection::session().await?;
    let proxy = zbus::Proxy::new(
        &connection,
        DBUS_SERVICE_NAME,
        DBUS_OBJECT_PATH,
        DBUS_INTERFACE_NAME,
    ).await?;

    proxy.call::<_, _, ()>(method, &()).await?;
    Ok(())
}

fn send_start_recording() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?.block_on(call_dbus_method("StartRecording"))
        .map_err(dbus_error_with_hint)
}

fn send_stop_recording() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?.block_on(call_dbus_method("StopRecording"))
        .map_err(dbus_error_with_hint)
}

fn send_confirm() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?.block_on(call_dbus_method("Confirm"))
        .map_err(dbus_error_with_hint)
}

fn dbus_error_with_hint(e: Box<dyn std::error::Error>) -> Box<dyn std::error::Error> {
    format!(
        "Failed to communicate with daemon: {}\nTry: systemctl --user status voice-dictation",
        e
    ).into()
}

async fn call_health_check() -> Result<(String, String, String), Box<dyn std::error::Error>> {
    let connection = Connection::session().await?;
    let proxy = zbus::Proxy::new(
        &connection,
        DBUS_SERVICE_NAME,
        DBUS_OBJECT_PATH,
        DBUS_INTERFACE_NAME,
    ).await?;

    let result: (String, String, String) = proxy.call("HealthCheck", &()).await?;
    Ok(result)
}

fn get_health_check() -> Result<(String, String, String), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?.block_on(call_health_check())
}

fn is_daemon_running() -> bool {
    if let Ok(rt) = tokio::runtime::Runtime::new() {
        rt.block_on(async {
            if let Ok(conn) = Connection::session().await {
                if let Ok(proxy) = zbus::Proxy::new(
                    &conn,
                    DBUS_SERVICE_NAME,
                    DBUS_OBJECT_PATH,
                    DBUS_INTERFACE_NAME,
                ).await {
                    proxy.introspect().await.is_ok()
                } else {
                    false
                }
            } else {
                false
            }
        })
    } else {
        false
    }
}

fn check_command_available(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn check_runtime_dependencies(require_wtype: bool, require_wayland: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut missing = Vec::new();
    let mut warnings = Vec::new();

    if require_wtype && !check_command_available("wtype") {
        missing.push("wtype - required for keyboard input injection");
    }

    if require_wayland {
        if std::env::var("WAYLAND_DISPLAY").is_err() {
            if std::env::var("DISPLAY").is_ok() {
                missing.push("Wayland compositor - X11 detected but Wayland is required");
            } else {
                missing.push("Wayland compositor - no display server detected");
            }
        }
    }

    if !check_command_available("pactl") && !check_command_available("pw-cli") {
        warnings.push("pactl or pw-cli - audio device enumeration may not work");
    }

    if !warnings.is_empty() {
        eprintln!("Warnings:");
        for warning in warnings {
            eprintln!("  - {}", warning);
        }
        eprintln!();
    }

    if !missing.is_empty() {
        eprintln!("Missing required runtime dependencies:");
        for dep in missing {
            eprintln!("  - {}", dep);
        }
        eprintln!();
        eprintln!("Install missing dependencies:");
        eprintln!("  Arch: sudo pacman -S wtype pipewire");
        eprintln!("  Fedora: sudo dnf install wtype pipewire");
        return Err("Missing runtime dependencies".into());
    }

    Ok(())
}

fn start_recording() -> Result<(), Box<dyn std::error::Error>> {
    if !is_daemon_running() {
        eprintln!("Error: Daemon not running");
        eprintln!("Start the daemon with: systemctl --user start voice-dictation");
        eprintln!("Or run manually: voice-dictation daemon");
        return Err("Daemon not running".into());
    }

    let state = get_state();
    if state == "recording" {
        println!("Already recording");
        return Ok(());
    }

    // Pause media
    let media_playing = Command::new("playerctl")
        .arg("status")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|status| status.contains("Playing"))
        .unwrap_or(false);

    if media_playing {
        fs::write(MEDIA_STATE_FILE, "playing")?;
        let _ = Command::new("playerctl").arg("pause").output();
    } else {
        fs::write(MEDIA_STATE_FILE, "stopped")?;
    }

    send_start_recording()?;

    set_state("recording")?;
    println!("Voice dictation started - recording");

    Ok(())
}

fn stop_recording() -> Result<(), Box<dyn std::error::Error>> {
    let state = get_state();
    if state == "stopped" {
        println!("Not recording");
        return Ok(());
    }

    if !is_daemon_running() {
        eprintln!("Daemon not running");
        set_state("stopped")?;
        return Ok(());
    }

    send_stop_recording()?;

    if let Ok(media_state) = fs::read_to_string(MEDIA_STATE_FILE) {
        if media_state.trim() == "playing" {
            let _ = Command::new("playerctl").arg("play").output();
        }
    }
    let _ = fs::remove_file(MEDIA_STATE_FILE);

    set_state("stopped")?;
    println!("Recording canceled");

    Ok(())
}

fn confirm_recording() -> Result<(), Box<dyn std::error::Error>> {
    let state = get_state();
    if state != "recording" {
        eprintln!("Not in recording state (current: {})", state);
        return Err("Invalid state".into());
    }

    if !is_daemon_running() {
        eprintln!("Error: Daemon not running");
        eprintln!("Start the daemon with: systemctl --user start voice-dictation");
        eprintln!("Or run manually: voice-dictation daemon");
        return Err("Daemon not running".into());
    }

    println!("Confirming transcription...");
    send_confirm()?;

    thread::sleep(Duration::from_millis(500));

    if let Ok(media_state) = fs::read_to_string(MEDIA_STATE_FILE) {
        if media_state.trim() == "playing" {
            let _ = Command::new("playerctl").arg("play").output();
        }
    }
    let _ = fs::remove_file(MEDIA_STATE_FILE);

    set_state("stopped")?;
    println!("Transcription confirmed");

    Ok(())
}

fn toggle_recording() -> Result<(), Box<dyn std::error::Error>> {
    let state = get_state();

    match state.as_str() {
        "stopped" => start_recording(),
        "recording" => confirm_recording(),
        _ => {
            eprintln!("Unknown state: {}", state);
            Err("Unknown state".into())
        }
    }
}

fn show_status() {
    let daemon_running = is_daemon_running();
    println!("Daemon: {}", if daemon_running { "running" } else { "NOT running" });

    if daemon_running {
        let state = get_state();
        println!("State: {}", state);

        match get_health_check() {
            Ok((gui, engine, audio)) => {
                println!("\nSubsystem Health:");
                println!("  GUI:    {}", gui);
                println!("  Engine: {}", engine);
                println!("  Audio:  {}", audio);
            }
            Err(e) => {
                println!("Health check unavailable: {}", e);
            }
        }
    }
}

fn validate_and_prompt_models(_config_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let models_dir = PathBuf::from(&home).join(".config/voice-dictation/models");

    if !models_dir.exists() {
        fs::create_dir_all(&models_dir)?;
    }

    // Check Parakeet model
    let parakeet_dir = models_dir.join("parakeet");
    if !parakeet_dir.join("encoder-model.onnx").exists() || !parakeet_dir.join("decoder_joint-model.onnx").exists() {
        eprintln!("Parakeet model not found at {:?}", parakeet_dir);
        eprintln!("The Parakeet model is required for speech recognition.");
        eprintln!("Please install the model files to: {}", parakeet_dir.display());
    }

    Ok(())
}

// Embed schema in binary for installation
const CONFIG_SCHEMA: &str = include_str!("../config-schema.json");

// Embed UI examples for installation
const UI_STYLE1_EXAMPLE: &str = include_str!("../slint-gui/ui/examples/style1-default.slint");
const UI_STYLE2_EXAMPLE: &str = include_str!("../slint-gui/ui/examples/style2-minimal.slint");
const UI_EXAMPLES_README: &str = include_str!("../slint-gui/ui/examples/README.md");

/// Migrate old config format to Parakeet-only format
fn migrate_config(config_path: &PathBuf) -> Result<bool, Box<dyn std::error::Error>> {
    if !config_path.exists() {
        return Ok(false);
    }

    let content = fs::read_to_string(config_path)?;

    // Check if migration is needed (old format has vosk/whisper references or muxer fields)
    let has_old_format = content.lines().any(|line| {
        let line = line.trim();
        line.starts_with("transcription_engine")
            || line.starts_with("preview_model_custom_path")
            || line.starts_with("final_model_custom_path")
            || line.starts_with("whisper_final_model")
            || line.starts_with("whisper_model_path")
            || line.starts_with("use_gpu")
            || line.starts_with("muxer_")
    });

    // Check if models reference vosk or whisper
    let has_vosk_whisper_model = content.lines().any(|line| {
        let line = line.trim();
        (line.starts_with("preview_model") || line.starts_with("final_model"))
            && (line.contains("vosk:") || line.contains("whisper:"))
    });

    if !has_old_format && !has_vosk_whisper_model {
        return Ok(false);
    }

    println!("Migrating config to Parakeet-only format...");

    // Remove old fields and update model references
    let mut new_lines: Vec<String> = Vec::new();
    let mut updated_preview = false;
    let mut updated_final = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip deprecated fields
        if trimmed.starts_with("transcription_engine")
            || trimmed.starts_with("preview_model_custom_path")
            || trimmed.starts_with("final_model_custom_path")
            || trimmed.starts_with("whisper_preview_model")
            || trimmed.starts_with("whisper_final_model")
            || trimmed.starts_with("whisper_model_path")
            || trimmed.starts_with("use_gpu")
            || trimmed.starts_with("muxer_")
        {
            continue;
        }

        // Update preview_model to parakeet
        if trimmed.starts_with("preview_model") && !trimmed.contains("custom_path") {
            if trimmed.contains("vosk:") || trimmed.contains("whisper:") {
                new_lines.push("preview_model = \"parakeet:default\"".to_string());
                updated_preview = true;
                continue;
            }
        }

        // Update final_model to parakeet
        if trimmed.starts_with("final_model") && !trimmed.contains("custom_path") {
            if trimmed.contains("vosk:") || trimmed.contains("whisper:") {
                new_lines.push("final_model = \"parakeet:default\"".to_string());
                updated_final = true;
                continue;
            }
        }

        new_lines.push(line.to_string());
    }

    // Write migrated config
    let new_content = new_lines.join("\n");
    fs::write(config_path, &new_content)?;

    println!("Config migrated to Parakeet-only format");
    if updated_preview || updated_final {
        println!("  Models updated to parakeet:default");
    }

    Ok(true)
}

fn open_config() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let config_dir = PathBuf::from(&home).join(".config/voice-dictation");
    let config_path = config_dir.join("config.toml");
    let schema_path = config_dir.join("config-schema.json");

    if !config_dir.exists() {
        fs::create_dir_all(&config_dir)?;
    }

    // Initialize UI examples directory
    let ui_examples_dir = config_dir.join("ui/examples");
    if !ui_examples_dir.exists() {
        fs::create_dir_all(&ui_examples_dir)?;
        fs::write(ui_examples_dir.join("style1-default.slint"), UI_STYLE1_EXAMPLE)?;
        fs::write(ui_examples_dir.join("style2-minimal.slint"), UI_STYLE2_EXAMPLE)?;
        fs::write(ui_examples_dir.join("README.md"), UI_EXAMPLES_README)?;
    }

    if !config_path.exists() {
        fs::write(&config_path, "")?;
    }

    // Migrate old config format if needed
    migrate_config(&config_path)?;

    // Install/update schema from embedded version
    fs::write(&schema_path, CONFIG_SCHEMA)?;

    let mut tui = SchemaTUIBuilder::new()
        .schema_file(&schema_path)?
        .config_file(&config_path)?
        .build()?;

    tui.run()?;

    validate_and_prompt_models(&config_path)?;

    Ok(())
}

const DEBUG_DIR: &str = "/tmp/voice-dictation-debug";

fn debug_list() -> Result<(), Box<dyn std::error::Error>> {
    let debug_dir = std::path::Path::new(DEBUG_DIR);
    if !debug_dir.exists() {
        println!("No debug recordings found (directory does not exist)");
        println!("Enable debug audio with: VOICE_DICTATION_DEBUG_AUDIO=1 voice-dictation daemon");
        return Ok(());
    }

    let mut entries: Vec<_> = fs::read_dir(debug_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();

    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    if entries.is_empty() {
        println!("No debug recordings found");
        println!("Enable debug audio with: VOICE_DICTATION_DEBUG_AUDIO=1 voice-dictation daemon");
        return Ok(());
    }

    println!("{:<35} {:>8} {:>6}  {}", "File", "Duration", "Device", "Text preview");
    println!("{}", "-".repeat(80));

    for entry in entries {
        let json_path = entry.path();
        let wav_name = json_path.with_extension("wav")
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        if let Ok(content) = fs::read_to_string(&json_path) {
            if let Ok(meta) = serde_json::from_str::<Value>(&content) {
                let duration_ms = meta["duration_ms"].as_u64().unwrap_or(0);
                let device = meta["active_device"].as_str().unwrap_or("?");
                let device_short = if device.len() > 6 { &device[..6] } else { device };
                let text = meta["final_text"].as_str()
                    .or_else(|| meta["preview_text"].as_str())
                    .unwrap_or("(no text)");
                let text_preview = if text.len() > 35 {
                    format!("{}...", &text[..32])
                } else {
                    text.to_string()
                };
                println!("{:<35} {:>6}ms {:>6}  {}", wav_name, duration_ms, device_short, text_preview);
            } else {
                println!("{:<35} (unreadable metadata)", wav_name);
            }
        }
    }

    Ok(())
}

fn debug_play(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let wav_path = if filename.contains('/') {
        PathBuf::from(filename)
    } else {
        PathBuf::from(DEBUG_DIR).join(filename)
    };

    if !wav_path.exists() {
        return Err(format!("File not found: {}", wav_path.display()).into());
    }

    let player = if Command::new("which").arg("paplay").output().map(|o| o.status.success()).unwrap_or(false) {
        "paplay"
    } else if Command::new("which").arg("aplay").output().map(|o| o.status.success()).unwrap_or(false) {
        "aplay"
    } else {
        return Err("No audio player found. Install pipewire-utils (paplay) or alsa-utils (aplay).".into());
    };

    println!("Playing: {}", wav_path.display());
    let status = Command::new(player).arg(&wav_path).status()?;
    if !status.success() {
        return Err(format!("{} failed with status: {}", player, status).into());
    }

    Ok(())
}

fn diagnose() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let config_path = PathBuf::from(&home).join(".config/voice-dictation/config.toml");

    println!("=== Voice Dictation Diagnostics ===\n");

    // Audio devices
    println!("Audio Input Devices:");
    for device in utils::list_audio_devices() {
        println!("  {}", device);
    }

    // Backend and config
    println!("\nConfiguration ({}):", config_path.display());
    if config_path.exists() {
        let config = fs::read_to_string(&config_path)?;
        let mut shown_any = false;
        for line in config.lines() {
            let t = line.trim();
            if t.starts_with("audio_backend")
                || t.starts_with("preview_model")
                || t.starts_with("final_model")
                || t.starts_with("audio_device")
            {
                println!("  {}", t);
                shown_any = true;
            }
        }
        if !shown_any {
            println!("  (using defaults - no relevant settings found)");
        }
    } else {
        println!("  (config file not found - using defaults)");
    }

    // Engine availability
    println!("\nAvailable engines: {}", utils::get_engine_summary());

    // Check Parakeet model
    let models_dir = PathBuf::from(&home).join(".config/voice-dictation/models/parakeet");
    let encoder_exists = models_dir.join("encoder-model.onnx").exists();
    let decoder_exists = models_dir.join("decoder_joint-model.onnx").exists();
    println!("\nParakeet model:");
    println!("  Directory: {}", models_dir.display());
    println!("  Encoder:   {}", if encoder_exists { "found" } else { "MISSING" });
    println!("  Decoder:   {}", if decoder_exists { "found" } else { "MISSING" });

    // Debug audio status
    let debug_enabled = std::env::var("VOICE_DICTATION_DEBUG_AUDIO")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let rust_log_debug = std::env::var("RUST_LOG")
        .map(|v| v.contains("debug") || v.contains("trace"))
        .unwrap_or(false);
    println!("\nDebug audio recording: {}", if debug_enabled || rust_log_debug { "enabled" } else { "disabled" });
    if !debug_enabled && !rust_log_debug {
        println!("  Enable with: VOICE_DICTATION_DEBUG_AUDIO=1 voice-dictation daemon");
    } else {
        println!("  Recordings saved to: {}", DEBUG_DIR);
        let count = fs::read_dir(DEBUG_DIR).ok()
            .map(|d| d.filter_map(|e| e.ok()).filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false)).count())
            .unwrap_or(0);
        println!("  Current recordings: {} (use 'voice-dictation debug list' to view)", count);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon => {
            check_runtime_dependencies(true, true)?;
            dictation_engine::run()?;
        }
        Commands::Start => {
            check_runtime_dependencies(true, false)?;
            start_recording()?;
        }
        Commands::Stop => {
            stop_recording()?;
        }
        Commands::Confirm => {
            check_runtime_dependencies(true, false)?;
            confirm_recording()?;
        }
        Commands::Toggle => {
            check_runtime_dependencies(true, false)?;
            toggle_recording()?;
        }
        Commands::Status => {
            show_status();
        }
        Commands::Config => {
            open_config()?;
        }
        Commands::ListModels => {
            for model in utils::list_models() {
                println!("{}", model);
            }
        }
        Commands::ListPreviewModels { language } => {
            for model in utils::list_preview_models(&language) {
                println!("{}", model);
            }
        }
        Commands::ListFinalModels { language } => {
            for model in utils::list_final_models(&language) {
                println!("{}", model);
            }
        }
        Commands::ListAudioDevices => {
            for device in utils::list_audio_devices() {
                println!("{}", device);
            }
        }
        Commands::Debug { command } => match command {
            DebugCommands::List => debug_list()?,
            DebugCommands::Play { filename } => debug_play(&filename)?,
        },
        Commands::Diagnose => diagnose()?,
    }

    Ok(())
}
