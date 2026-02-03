use clap::{Parser, Subcommand};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;
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
#[command(about = "Voice dictation system with speech recognition", long_about = None)]
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
}

fn get_state() -> String {
    fs::read_to_string(STATE_FILE).unwrap_or_else(|_| "stopped".to_string()).trim().to_string()
}

fn set_state(state: &str) -> std::io::Result<()> {
    fs::write(STATE_FILE, state)
}

fn is_process_running(pattern: &str) -> bool {
    Command::new("pgrep")
        .arg("-f")
        .arg(pattern)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
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
}

fn send_stop_recording() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?.block_on(call_dbus_method("StopRecording"))
}

fn send_confirm() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?.block_on(call_dbus_method("Confirm"))
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
    // Check if D-Bus service name is registered
    if let Ok(rt) = tokio::runtime::Runtime::new() {
        rt.block_on(async {
            if let Ok(conn) = Connection::session().await {
                if let Ok(proxy) = zbus::Proxy::new(
                    &conn,
                    DBUS_SERVICE_NAME,
                    DBUS_OBJECT_PATH,
                    DBUS_INTERFACE_NAME,
                ).await {
                    // Try to introspect to verify service is alive
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

    // Check for wtype (critical for keyboard typing)
    if require_wtype && !check_command_available("wtype") {
        missing.push("wtype - required for keyboard input injection");
    }

    // Check for Wayland display (critical for GUI)
    if require_wayland {
        if std::env::var("WAYLAND_DISPLAY").is_err() {
            if std::env::var("DISPLAY").is_ok() {
                missing.push("Wayland compositor - X11 detected but Wayland is required");
            } else {
                missing.push("Wayland compositor - no display server detected");
            }
        }
    }

    // Check for audio tools (warn if missing)
    if !check_command_available("pactl") && !check_command_available("pw-cli") {
        warnings.push("pactl or pw-cli - audio device enumeration may not work");
    }

    // Print warnings (non-fatal)
    if !warnings.is_empty() {
        eprintln!("⚠️  Warnings:");
        for warning in warnings {
            eprintln!("  - {}", warning);
        }
        eprintln!();
    }

    // Print errors and fail if any critical dependencies missing
    if !missing.is_empty() {
        eprintln!("❌ Missing required runtime dependencies:");
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
    // Check if daemon is running
    if !is_daemon_running() {
        eprintln!("Error: Daemon not running");
        eprintln!("Start the daemon with: systemctl --user start voice-dictation");
        eprintln!("Or run manually: voice-dictation daemon");
        return Err("Daemon not running".into());
    }

    // Check current state
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

    // Send StartRecording command to daemon via D-Bus
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

    // Send StopRecording command to daemon via D-Bus
    send_stop_recording()?;

    // Resume media if it was playing
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
        eprintln!("Daemon not running");
        return Err("Daemon not running".into());
    }

    println!("Confirming transcription...");
    send_confirm()?;

    // Wait a moment for processing to complete
    thread::sleep(Duration::from_millis(500));

    // Resume media if it was playing
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
    // Always show daemon status first
    let daemon_running = is_daemon_running();
    println!("Daemon: {}", if daemon_running { "running" } else { "NOT running" });

    if daemon_running {
        let state = get_state();
        println!("State: {}", state);

        // Try to get health check info
        match get_health_check() {
            Ok((gui, monitor, audio)) => {
                println!("\nSubsystem Health:");
                println!("  GUI:     {}", gui);
                println!("  Monitor: {}", monitor);
                println!("  Audio:   {}", audio);
            }
            Err(e) => {
                println!("Health check unavailable: {}", e);
            }
        }
    }
}

/// Parse model spec format: "engine:model_name"
fn parse_model_spec(spec: &str) -> Option<(&str, &str)> {
    let parts: Vec<&str> = spec.splitn(2, ':').collect();
    if parts.len() == 2 {
        Some((parts[0], parts[1]))
    } else {
        None
    }
}

fn check_model_exists(model_spec: &str, models_dir: &PathBuf) -> bool {
    if let Some((engine, model_name)) = parse_model_spec(model_spec) {
        match engine {
            "whisper" => {
                // Whisper models are auto-downloaded, so always "available"
                true
            }
            "parakeet" => {
                // Parakeet uses a fixed model location
                models_dir.join("parakeet").exists()
            }
            "vosk" => {
                // Vosk models must be manually downloaded
                models_dir.join(model_name).exists()
            }
            _ => false,
        }
    } else {
        // Legacy format (just model name without engine prefix)
        models_dir.join(model_spec).exists()
    }
}

fn get_vosk_model_url(model_name: &str) -> String {
    format!("https://alphacephei.com/vosk/models/{}.zip", model_name)
}

fn download_vosk_model(model_name: &str, models_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let url = get_vosk_model_url(model_name);
    let zip_path = models_dir.join(format!("{}.zip", model_name));

    println!("Downloading {} ({})...", model_name, url);
    println!("This may take several minutes depending on model size...");

    let status = Command::new("curl")
        .arg("-L")
        .arg("-o")
        .arg(&zip_path)
        .arg(&url)
        .status()?;

    if !status.success() {
        return Err("Download failed".into());
    }

    println!("Extracting model...");
    let status = Command::new("unzip")
        .arg("-q")
        .arg(&zip_path)
        .arg("-d")
        .arg(models_dir)
        .status()?;

    if !status.success() {
        return Err("Extraction failed".into());
    }

    fs::remove_file(&zip_path)?;
    println!("✓ Model installed successfully");

    Ok(())
}

fn validate_and_prompt_models(config_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let config_content = fs::read_to_string(config_path)?;

    let home = std::env::var("HOME")?;
    let models_dir = PathBuf::from(&home).join(".config/voice-dictation/models");

    if !models_dir.exists() {
        fs::create_dir_all(&models_dir)?;
    }

    let preview_model = config_content
        .lines()
        .find(|line| line.starts_with("preview_model"))
        .and_then(|line| line.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"').to_string());

    let final_model = config_content
        .lines()
        .find(|line| line.starts_with("final_model"))
        .and_then(|line| line.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"').to_string());

    // Collect missing Vosk models (whisper auto-downloads, parakeet bundled)
    let mut missing_vosk_models = Vec::new();

    if let Some(model_spec) = &preview_model {
        if let Some(("vosk", model_name)) = parse_model_spec(model_spec) {
            if !models_dir.join(model_name).exists() {
                missing_vosk_models.push(("Preview", model_name.to_string()));
            }
        }
    }

    if let Some(model_spec) = &final_model {
        if let Some(("vosk", model_name)) = parse_model_spec(model_spec) {
            if !models_dir.join(model_name).exists() {
                missing_vosk_models.push(("Final", model_name.to_string()));
            }
        }
    }

    if missing_vosk_models.is_empty() {
        return Ok(());
    }

    println!("\n⚠️  Missing Vosk models detected:");
    for (model_type, model_name) in &missing_vosk_models {
        println!("  - {} model: {}", model_type, model_name);
        println!("    URL: {}", get_vosk_model_url(model_name));
    }

    print!("\nWould you like to download missing models now? [y/N]: ");
    io::stdout().flush()?;

    let mut response = String::new();
    io::stdin().read_line(&mut response)?;

    if response.trim().to_lowercase() == "y" {
        for (model_type, model_name) in &missing_vosk_models {
            println!("\nDownloading {} model: {}", model_type, model_name);
            if let Err(e) = download_vosk_model(model_name, &models_dir) {
                eprintln!("✗ Failed to download {}: {}", model_name, e);
                eprintln!("  Please download manually from: {}", get_vosk_model_url(model_name));
            }
        }
    } else {
        println!("\nSkipping download. You can download models manually with:");
        println!("  cd ~/.config/voice-dictation/models");
        for (_, model_name) in &missing_vosk_models {
            println!("  curl -L -O {}", get_vosk_model_url(model_name));
            println!("  unzip {}.zip", model_name);
        }
    }

    Ok(())
}

// Embed schema in binary for installation
const CONFIG_SCHEMA: &str = include_str!("../config-schema.json");

// Embed UI examples for installation
const UI_STYLE1_EXAMPLE: &str = include_str!("../slint-gui/ui/examples/style1-default.slint");
const UI_STYLE2_EXAMPLE: &str = include_str!("../slint-gui/ui/examples/style2-minimal.slint");
const UI_EXAMPLES_README: &str = include_str!("../slint-gui/ui/examples/README.md");

/// Migrate old config format to new unified model selection format
fn migrate_config(config_path: &PathBuf) -> Result<bool, Box<dyn std::error::Error>> {
    if !config_path.exists() {
        return Ok(false);
    }

    let content = fs::read_to_string(config_path)?;

    // Check if migration is needed (old format has transcription_engine or model without engine prefix)
    let has_old_format = content.lines().any(|line| {
        let line = line.trim();
        line.starts_with("transcription_engine")
            || line.starts_with("preview_model_custom_path")
            || line.starts_with("final_model_custom_path")
            || line.starts_with("whisper_final_model")
            || line.starts_with("whisper_model_path")
            || line.starts_with("use_gpu")
    });

    // Also check if preview_model/final_model are in old format (no colon)
    let preview_model_old = content.lines()
        .find(|line| line.trim().starts_with("preview_model") && !line.contains("custom_path"))
        .and_then(|line| line.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"'))
        .map(|s| !s.contains(':'))
        .unwrap_or(false);

    let final_model_old = content.lines()
        .find(|line| line.trim().starts_with("final_model") && !line.contains("custom_path"))
        .and_then(|line| line.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"'))
        .map(|s| !s.contains(':'))
        .unwrap_or(false);

    if !has_old_format && !preview_model_old && !final_model_old {
        return Ok(false);
    }

    println!("Migrating config to new unified model selection format...");

    // Parse old values
    let engine = content.lines()
        .find(|line| line.trim().starts_with("transcription_engine"))
        .and_then(|line| line.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"').to_string())
        .unwrap_or_else(|| "vosk".to_string());

    let old_preview = content.lines()
        .find(|line| line.trim().starts_with("preview_model") && !line.contains("custom_path"))
        .and_then(|line| line.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"').to_string());

    let old_final = content.lines()
        .find(|line| line.trim().starts_with("final_model") && !line.contains("custom_path"))
        .and_then(|line| line.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"').to_string());

    let whisper_final = content.lines()
        .find(|line| line.trim().starts_with("whisper_final_model"))
        .and_then(|line| line.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"').to_string());

    // Build new model specs
    let new_preview = if let Some(model) = old_preview {
        if model.contains(':') {
            model // Already in new format
        } else if model == "custom" {
            "vosk:vosk-model-en-us-daanzu-20200905-lgraph".to_string() // Default
        } else {
            format!("{}:{}", engine, model)
        }
    } else {
        "vosk:vosk-model-en-us-daanzu-20200905-lgraph".to_string()
    };

    let new_final = if let Some(model) = old_final {
        if model.contains(':') {
            model // Already in new format
        } else if model == "custom" {
            whisper_final.map(|m| format!("whisper:{}", m))
                .unwrap_or_else(|| "whisper:ggml-small.en.bin".to_string())
        } else {
            format!("{}:{}", engine, model)
        }
    } else if engine == "whisper" {
        whisper_final.map(|m| format!("whisper:{}", m))
            .unwrap_or_else(|| "whisper:ggml-small.en.bin".to_string())
    } else {
        "whisper:ggml-small.en.bin".to_string()
    };

    // Remove old fields and update with new format
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
        {
            continue;
        }

        // Update preview_model
        if trimmed.starts_with("preview_model") && !trimmed.contains("custom_path") {
            new_lines.push(format!("preview_model = \"{}\"", new_preview));
            updated_preview = true;
            continue;
        }

        // Update final_model
        if trimmed.starts_with("final_model") && !trimmed.contains("custom_path") {
            new_lines.push(format!("final_model = \"{}\"", new_final));
            updated_final = true;
            continue;
        }

        new_lines.push(line.to_string());
    }

    // Add fields if not already present
    if !updated_preview {
        // Find [daemon] section and add after it
        if let Some(pos) = new_lines.iter().position(|l| l.trim() == "[daemon]") {
            new_lines.insert(pos + 1, format!("preview_model = \"{}\"", new_preview));
        }
    }
    if !updated_final {
        if let Some(pos) = new_lines.iter().position(|l| l.trim() == "[daemon]") {
            new_lines.insert(pos + 2, format!("final_model = \"{}\"", new_final));
        }
    }

    // Write migrated config
    let new_content = new_lines.join("\n");
    fs::write(config_path, &new_content)?;

    println!("✓ Config migrated successfully");
    println!("  preview_model = \"{}\"", new_preview);
    println!("  final_model = \"{}\"", new_final);

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

    // Initialize UI examples directory (user can copy to ui/dictation.slint to activate)
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon => {
            // Daemon requires both wtype (for typing) and Wayland (for GUI)
            check_runtime_dependencies(true, true)?;
            dictation_engine::run()?;
        }
        Commands::Start => {
            // Start requires wtype for eventual typing (daemon handles Wayland)
            check_runtime_dependencies(true, false)?;
            start_recording()?;
        }
        Commands::Stop => {
            stop_recording()?;
        }
        Commands::Confirm => {
            // Confirm requires wtype for typing (daemon handles Wayland)
            check_runtime_dependencies(true, false)?;
            confirm_recording()?;
        }
        Commands::Toggle => {
            // Toggle may start or confirm, so require wtype
            check_runtime_dependencies(true, false)?;
            toggle_recording()?;
        }
        Commands::Status => {
            show_status();
        }
        Commands::Config => {
            open_config()?;
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
    }

    Ok(())
}
