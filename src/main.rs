use clap::{Parser, Subcommand};
use schema_tui::SchemaTUIBuilder;
use serde_json::Value;
use std::fs;
use std::io::{self, Write as IoWrite};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;
use zbus::Connection;

mod utils;

// Well-known bus name (live). Interface is com.voicedictation.Control.
// Do not rename: CLI, keybinds, and other clients depend on this name.
const DBUS_SERVICE_NAME: &str = "com.voicedictation.Daemon";
const DBUS_OBJECT_PATH: &str = "/com/voicedictation/Control";
const DBUS_INTERFACE_NAME: &str = "com.voicedictation.Control";

#[derive(Parser)]
#[command(name = "voice-dictation")]
#[command(version)]
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
    #[command(about = "Download Parakeet speech recognition model from HuggingFace")]
    DownloadModel,
    #[command(about = "Open the test loop UI for iterative transcription improvement")]
    TestLoop {
        /// Path to a specific recording WAV file (defaults to most recent)
        #[arg(long)]
        recording: Option<String>,
    },
    #[command(
        about = "Re-arm correction monitoring for the last dictation (captures late edits as corrections)"
    )]
    SnapshotCorrection,
    #[command(about = "Show correction learning statistics")]
    CorrectionStats,
    #[command(about = "Manage learned corrections")]
    Corrections {
        #[command(subcommand)]
        command: CorrectionCommands,
    },
    #[command(
        about = "Manage the user dictionary (words exempt from spell checking)",
        long_about = "Manage the user dictionary (words exempt from spell checking).\n\n\
                      Edits ~/.local/share/voice-dictation/user_words.txt directly. The running\n\
                      daemon hot-reloads the file via its watcher — no restart needed."
    )]
    Dict {
        #[command(subcommand)]
        command: DictCommands,
    },
    #[command(
        about = "Manage word substitutions (spoken phrase -> replacement)",
        long_about = "Manage word substitutions (spoken phrase -> replacement).\n\n\
                      Edits ~/.local/share/voice-dictation/substitutions.txt directly. The running\n\
                      daemon hot-reloads the file via its watcher — no restart needed."
    )]
    Subst {
        #[command(subcommand)]
        command: SubstCommands,
    },
}

#[derive(Subcommand)]
enum DictCommands {
    #[command(about = "Add a word to the user dictionary")]
    Add {
        #[arg(help = "Word to add")]
        word: String,
    },
    #[command(about = "Remove a word from the user dictionary")]
    Remove {
        #[arg(help = "Word to remove")]
        word: String,
    },
    #[command(about = "List all user dictionary words")]
    List,
}

#[derive(Subcommand)]
enum SubstCommands {
    #[command(about = "Add a substitution (spoken phrase -> replacement)")]
    Add {
        #[arg(help = "Spoken phrase as transcribed (e.g. \"shay moy\")")]
        spoken: String,
        #[arg(help = "Replacement text (e.g. \"chezmoi\")")]
        replacement: String,
    },
    #[command(about = "Remove a substitution by its spoken phrase")]
    Remove {
        #[arg(help = "Spoken phrase of the substitution to remove")]
        spoken: String,
    },
    #[command(about = "List all substitutions")]
    List,
}

#[derive(Subcommand)]
enum CorrectionCommands {
    #[command(about = "List all learned corrections (and blocklisted pairs)")]
    List,
    #[command(
        about = "Remove learned corrections by their original phrase and blocklist them",
        long_about = "Remove learned corrections by their original phrase (from 'corrections list').\n\n\
                      Removed pairs go to a persisted blocklist and are never re-recorded or\n\
                      auto-promoted again."
    )]
    Remove {
        #[arg(help = "Original (transcribed) phrase of the correction(s) to remove")]
        original: String,
    },
    #[command(about = "Remove all learned corrections and start fresh (blocklist is kept)")]
    Clear,
    #[command(about = "Open corrections file in $EDITOR")]
    Edit,
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
    #[command(
        about = "Probe the wezterm mux plumbing used by the wezterm correction backend",
        long_about = "Probe the wezterm mux plumbing used by the wezterm correction backend.\n\n\
                      Read-only: discovers the live wezterm socket, lists panes, and prints the\n\
                      first lines of the active pane's text — validating everything the daemon\n\
                      needs for wezterm-native correction detection in one command."
    )]
    WeztermProbe,
    #[command(
        about = "Time-boxed zwp_input_method_v2 diagnostic probe (holds the compositor IME slot)",
        long_about = "Bind zwp_input_method_v2 for a fixed number of seconds and log every IME\n\
                      event (activate/deactivate, surrounding_text, text_change_cause,\n\
                      content_type), tagged with the focused window class from hyprctl.\n\n\
                      Never grabs the keyboard. Exits immediately if another IME holds the slot.\n\
                      --commit sends commit_string into the focused editor for write-path\n\
                      validation, but ONLY while a window whose class exactly matches\n\
                      --commit-class is focused (use a throwaway target like `zenity --entry`)."
    )]
    ImeProbe {
        #[arg(long, default_value_t = 30, help = "Seconds to hold the IME slot")]
        secs: u64,
        #[arg(long, help = "Emit a machine-readable JSON summary on stdout")]
        json: bool,
        #[arg(
            long,
            requires = "commit_class",
            help = "Text to commit_string into the focused editor (write-path test)"
        )]
        commit: Option<String>,
        #[arg(
            long,
            help = "Exact window class the commit is restricted to (safety gate, e.g. 'zenity')"
        )]
        commit_class: Option<String>,
    },
}

async fn call_status() -> Result<(String, bool), Box<dyn std::error::Error>> {
    let connection = Connection::session().await?;
    let proxy =
        zbus::Proxy::new(&connection, DBUS_SERVICE_NAME, DBUS_OBJECT_PATH, DBUS_INTERFACE_NAME)
            .await?;

    let result: (String, bool) = proxy.call("Status", &()).await?;
    Ok(result)
}

fn get_state() -> Result<String, Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?
        .block_on(call_status())
        .map(|(state, _session_active)| state)
        .map_err(dbus_error_with_hint)
}

async fn call_dbus_method(method: &str) -> Result<(), Box<dyn std::error::Error>> {
    let connection = Connection::session().await?;
    let proxy =
        zbus::Proxy::new(&connection, DBUS_SERVICE_NAME, DBUS_OBJECT_PATH, DBUS_INTERFACE_NAME)
            .await?;

    proxy.call::<_, _, ()>(method, &()).await?;
    Ok(())
}

fn send_start_recording() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?
        .block_on(call_dbus_method("StartRecording"))
        .map_err(dbus_error_with_hint)
}

fn send_stop_recording() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?
        .block_on(call_dbus_method("StopRecording"))
        .map_err(dbus_error_with_hint)
}

fn send_confirm() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?
        .block_on(call_dbus_method("Confirm"))
        .map_err(dbus_error_with_hint)
}

fn dbus_error_with_hint(e: Box<dyn std::error::Error>) -> Box<dyn std::error::Error> {
    format!(
        "Failed to communicate with daemon: {}\nTry: systemctl --user status voice-dictation",
        e
    )
    .into()
}

async fn call_health_check() -> Result<(String, String, String), Box<dyn std::error::Error>> {
    let connection = Connection::session().await?;
    let proxy =
        zbus::Proxy::new(&connection, DBUS_SERVICE_NAME, DBUS_OBJECT_PATH, DBUS_INTERFACE_NAME)
            .await?;

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
                )
                .await
                {
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
    Command::new("which").arg(cmd).output().map(|output| output.status.success()).unwrap_or(false)
}

fn check_runtime_dependencies(
    require_wtype: bool,
    require_wayland: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut missing = Vec::new();
    let mut warnings = Vec::new();

    if require_wtype && !check_command_available("wtype") {
        missing.push("wtype - required for keyboard input injection");
    }

    if require_wayland && std::env::var("WAYLAND_DISPLAY").is_err() {
        if std::env::var("DISPLAY").is_ok() {
            missing.push("Wayland compositor - X11 detected but Wayland is required");
        } else {
            missing.push("Wayland compositor - no display server detected");
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

    let state = get_state()?;
    if state == "recording" {
        println!("Already recording");
        return Ok(());
    }

    send_start_recording()?;
    println!("Voice dictation started - recording");

    Ok(())
}

fn stop_recording() -> Result<(), Box<dyn std::error::Error>> {
    if !is_daemon_running() {
        eprintln!("Daemon not running");
        return Ok(());
    }

    let state = get_state()?;
    if state == "idle" || state == "stopped" {
        println!("Not recording");
        return Ok(());
    }

    send_stop_recording()?;
    println!("Recording canceled");

    Ok(())
}

fn confirm_recording() -> Result<(), Box<dyn std::error::Error>> {
    if !is_daemon_running() {
        eprintln!("Error: Daemon not running");
        eprintln!("Start the daemon with: systemctl --user start voice-dictation");
        eprintln!("Or run manually: voice-dictation daemon");
        return Err("Daemon not running".into());
    }

    let state = get_state()?;
    if state != "recording" {
        eprintln!("Not in recording state (current: {})", state);
        return Err("Invalid state".into());
    }

    println!("Confirming transcription...");
    send_confirm()?;

    thread::sleep(Duration::from_millis(500));

    println!("Transcription confirmed");

    Ok(())
}

fn toggle_recording() -> Result<(), Box<dyn std::error::Error>> {
    if !is_daemon_running() {
        eprintln!("Error: Daemon not running");
        eprintln!("Start the daemon with: systemctl --user start voice-dictation");
        eprintln!("Or run manually: voice-dictation daemon");
        return Err("Daemon not running".into());
    }

    let state = get_state()?;

    match state.as_str() {
        "idle" | "stopped" => start_recording(),
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
        match get_state() {
            Ok(state) => println!("State: {}", state),
            Err(e) => println!("State: unavailable ({})", e),
        }

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
    if !parakeet_dir.join("encoder-model.onnx").exists()
        || !parakeet_dir.join("decoder_joint-model.onnx").exists()
    {
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
        if trimmed.starts_with("preview_model")
            && !trimmed.contains("custom_path")
            && (trimmed.contains("vosk:") || trimmed.contains("whisper:"))
        {
            new_lines.push("preview_model = \"parakeet:default\"".to_string());
            updated_preview = true;
            continue;
        }

        // Update final_model to parakeet
        if trimmed.starts_with("final_model")
            && !trimmed.contains("custom_path")
            && (trimmed.contains("vosk:") || trimmed.contains("whisper:"))
        {
            new_lines.push("final_model = \"parakeet:default\"".to_string());
            updated_final = true;
            continue;
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

    let mut tui =
        SchemaTUIBuilder::new().schema_file(&schema_path)?.config_file(&config_path)?.build()?;

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

    entries.sort_by_key(|a| a.file_name());

    if entries.is_empty() {
        println!("No debug recordings found");
        println!("Enable debug audio with: VOICE_DICTATION_DEBUG_AUDIO=1 voice-dictation daemon");
        return Ok(());
    }

    println!("{:<35} {:>8} {:>6}  Text preview", "File", "Duration", "Device");
    println!("{}", "-".repeat(80));

    for entry in entries {
        let json_path = entry.path();
        let wav_name = json_path
            .with_extension("wav")
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        if let Ok(content) = fs::read_to_string(&json_path) {
            if let Ok(meta) = serde_json::from_str::<Value>(&content) {
                let duration_ms = meta["duration_ms"].as_u64().unwrap_or(0);
                let device = meta["active_device"].as_str().unwrap_or("?");
                let device_short = if device.len() > 6 { &device[..6] } else { device };
                let text = meta["final_text"]
                    .as_str()
                    .or_else(|| meta["preview_text"].as_str())
                    .unwrap_or("(no text)");
                let text_preview =
                    if text.len() > 35 { format!("{}...", &text[..32]) } else { text.to_string() };
                println!(
                    "{:<35} {:>6}ms {:>6}  {}",
                    wav_name, duration_ms, device_short, text_preview
                );
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

    let player = if Command::new("which")
        .arg("paplay")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        "paplay"
    } else if Command::new("which")
        .arg("aplay")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        "aplay"
    } else {
        return Err(
            "No audio player found. Install pipewire-utils (paplay) or alsa-utils (aplay).".into(),
        );
    };

    println!("Playing: {}", wav_path.display());
    let status = Command::new(player).arg(&wav_path).status()?;
    if !status.success() {
        return Err(format!("{} failed with status: {}", player, status).into());
    }

    Ok(())
}

/// One-command validation of the wezterm correction backend plumbing:
/// socket discovery, pane listing, active-pane selection, get-text.
/// Strictly read-only against the running wezterm.
fn debug_wezterm_probe() -> Result<(), Box<dyn std::error::Error>> {
    use correction_engine::wezterm::{discover_socket, WeztermCli, WeztermClient};

    let Some(socket) = discover_socket() else {
        println!("Socket: NOT FOUND");
        println!(
            "No live wezterm socket under $XDG_RUNTIME_DIR/wezterm/ (and $WEZTERM_UNIX_SOCKET \
             is unset or stale). Is a wezterm GUI instance running?"
        );
        return Ok(());
    };
    println!("Socket: {}", socket.display());

    let cli = WeztermCli::with_socket(socket);
    tokio::runtime::Runtime::new()?.block_on(async move {
        let panes = cli.list_panes().await?;
        println!("Panes:  {}", panes.len());
        for p in &panes {
            println!(
                "  pane {:<4} {}  workspace={:<10} title={}",
                p.pane_id,
                if p.is_active { "[active]" } else { "        " },
                p.workspace,
                p.title
            );
        }

        let Some(target) = panes.iter().find(|p| p.is_active).or_else(|| panes.first()) else {
            println!("No panes found — nothing to get-text from.");
            return Ok(());
        };
        println!("Active pane: {}", target.pane_id);

        let text = cli.get_text(target.pane_id).await?;
        println!("First 5 lines of get-text:");
        for line in text.lines().take(5) {
            println!("  | {}", line.trim_end());
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;
    Ok(())
}

fn corrections_command(command: CorrectionCommands) -> Result<(), Box<dyn std::error::Error>> {
    use correction_engine::{CorrectionStore, MonitorConfig};

    // Only the store/substitution paths matter for CLI use; the monitor
    // settings are irrelevant here.
    let config = MonitorConfig::default();

    match command {
        CorrectionCommands::List => {
            let store = CorrectionStore::load(&config)?;
            let records = store.records();
            if records.is_empty() {
                println!("No corrections recorded yet.");
            } else {
                println!("{:>5}  {:<20} {:<20} Status", "Count", "Original", "Corrected");
                println!("{}", "-".repeat(70));
                for r in records {
                    let status = if r.promoted { "promoted" } else { "pending" };
                    println!("{:>5}  {:<20} {:<20} {}", r.count, r.original, r.corrected, status);
                }
            }
            let blocklist = store.blocklist();
            if !blocklist.is_empty() {
                println!("\nBlocklisted (never re-learned):");
                for b in blocklist {
                    println!("  '{}' → '{}'", b.original, b.corrected);
                }
            }
            println!("\nFile: {}", config.store_path.display());
        }
        CorrectionCommands::Remove { original } => {
            let mut store = CorrectionStore::load(&config)?;
            let removed = store.remove(&original)?;
            if removed.is_empty() {
                println!(
                    "No correction found for '{}'. Use 'voice-dictation corrections list' to see originals.",
                    original
                );
                return Err("No matching correction".into());
            }
            for r in &removed {
                println!("Removed and blocklisted: '{}' → '{}'", r.original, r.corrected);
            }
        }
        CorrectionCommands::Clear => {
            print!("Remove all learned corrections? [y/N] ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim().to_lowercase() == "y" {
                let mut store = CorrectionStore::load(&config)?;
                let count = store.clear()?;
                println!("Cleared {} correction(s). Blocklist kept.", count);
            } else {
                println!("Cancelled.");
            }
        }
        CorrectionCommands::Edit => {
            let corrections_path = &config.store_path;
            if !corrections_path.exists() {
                // Create empty file
                if let Some(dir) = corrections_path.parent() {
                    fs::create_dir_all(dir)?;
                }
                let data = serde_json::json!({"version": 1, "corrections": [], "blocklist": []});
                fs::write(corrections_path, serde_json::to_string_pretty(&data)?)?;
            }
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
            let status = Command::new(&editor).arg(corrections_path).status()?;
            if !status.success() {
                eprintln!("{} exited with: {}", editor, status);
            }
        }
    }

    Ok(())
}

fn dict_command(command: DictCommands) -> Result<(), Box<dyn std::error::Error>> {
    use dictation_engine::user_dictionary::UserDictionary;

    let dict = UserDictionary::new()?;

    match command {
        DictCommands::Add { word } => {
            let word = word.trim().to_lowercase();
            if word.is_empty() {
                return Err("Word must not be empty".into());
            }
            if dict.app_words().contains(&word) {
                println!("'{}' is already in the user dictionary", word);
                return Ok(());
            }
            dict.add(&word)?;
            println!("Added '{}' to the user dictionary (daemon hot-reloads automatically)", word);
        }
        DictCommands::Remove { word } => {
            let word = word.trim().to_lowercase();
            if !dict.app_words().contains(&word) {
                println!("'{}' is not in the user dictionary", word);
                return Ok(());
            }
            dict.remove(&word)?;
            println!(
                "Removed '{}' from the user dictionary (daemon hot-reloads automatically)",
                word
            );
        }
        DictCommands::List => {
            let words = dict.app_words();
            if words.is_empty() {
                println!("User dictionary is empty.");
            } else {
                for word in &words {
                    println!("{}", word);
                }
                println!("\n{} word(s)", words.len());
            }
        }
    }

    Ok(())
}

/// Extract the spoken part of a `spoken -> replacement` line, normalized to
/// lowercase words. Returns None for comments, blanks, and malformed lines.
fn parse_substitution_spoken(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') || !trimmed.contains("->") {
        return None;
    }
    let spoken = trimmed.split("->").next()?.trim();
    if spoken.is_empty() {
        return None;
    }
    Some(spoken.split_whitespace().map(|w| w.to_lowercase()).collect())
}

fn subst_command(command: SubstCommands) -> Result<(), Box<dyn std::error::Error>> {
    use dictation_engine::post_processing::WordSubstitutionProcessor;

    let path = WordSubstitutionProcessor::get_substitutions_path()?;

    match command {
        SubstCommands::Add { spoken, replacement } => {
            let spoken = spoken.trim();
            let replacement = replacement.trim();
            if spoken.is_empty() || replacement.is_empty() {
                return Err("Spoken phrase and replacement must not be empty".into());
            }

            let spoken_words: Vec<String> =
                spoken.split_whitespace().map(|w| w.to_lowercase()).collect();
            let entries = WordSubstitutionProcessor::load_substitutions(&path)?;
            if let Some((_, existing)) = entries.iter().find(|(words, _)| *words == spoken_words) {
                println!(
                    "Substitution already exists: {} -> {}\nRemove it first: voice-dictation subst remove \"{}\"",
                    spoken_words.join(" "),
                    existing,
                    spoken_words.join(" ")
                );
                return Ok(());
            }

            let mut content =
                if path.exists() { fs::read_to_string(&path)? } else { String::new() };
            if !content.is_empty() && !content.ends_with('\n') {
                content.push('\n');
            }
            content.push_str(&format!("{} -> {}\n", spoken, replacement));
            fs::write(&path, content)?;
            println!(
                "Added substitution: {} -> {} (daemon hot-reloads automatically)",
                spoken, replacement
            );
        }
        SubstCommands::Remove { spoken } => {
            if !path.exists() {
                println!("No substitutions file found at {}", path.display());
                return Ok(());
            }
            let spoken_words: Vec<String> =
                spoken.split_whitespace().map(|w| w.to_lowercase()).collect();

            let content = fs::read_to_string(&path)?;
            let kept: Vec<&str> = content
                .lines()
                .filter(|line| parse_substitution_spoken(line).as_ref() != Some(&spoken_words))
                .collect();

            let removed = content.lines().count() - kept.len();
            if removed == 0 {
                println!("No substitution found for '{}'", spoken_words.join(" "));
                return Ok(());
            }

            let mut new_content = kept.join("\n");
            if !new_content.is_empty() {
                new_content.push('\n');
            }
            fs::write(&path, new_content)?;
            println!(
                "Removed {} substitution(s) for '{}' (daemon hot-reloads automatically)",
                removed,
                spoken_words.join(" ")
            );
        }
        SubstCommands::List => {
            let entries = WordSubstitutionProcessor::load_substitutions(&path)?;
            if entries.is_empty() {
                println!("No substitutions defined.");
                println!("File: {}", path.display());
            } else {
                for (spoken_words, replacement) in &entries {
                    println!("{} -> {}", spoken_words.join(" "), replacement);
                }
                println!("\n{} substitution(s)  File: {}", entries.len(), path.display());
            }
        }
    }

    Ok(())
}

/// Find all `voice-dictation` executables on `$PATH`, deduplicated by their
/// canonical target. Returns the PATH-order entries so the first one (the one
/// that actually wins) is listed first.
fn binaries_on_path(name: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut seen_canonical: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            if dir.is_empty() {
                continue;
            }
            let candidate = PathBuf::from(dir).join(name);
            if !candidate.is_file() {
                continue;
            }
            let canonical = fs::canonicalize(&candidate).unwrap_or_else(|_| candidate.clone());
            if !seen_canonical.contains(&canonical) {
                seen_canonical.push(canonical);
                found.push(candidate);
            }
        }
    }
    found
}

/// Ask the session bus which PID owns the daemon's well-known name, then
/// resolve that PID's executable via /proc. Returns None if the daemon isn't
/// running or the PID can't be resolved.
async fn get_daemon_pid() -> Option<u32> {
    let connection = Connection::session().await.ok()?;
    let proxy = zbus::Proxy::new(
        &connection,
        "org.freedesktop.DBus",
        "/org/freedesktop/DBus",
        "org.freedesktop.DBus",
    )
    .await
    .ok()?;
    proxy.call("GetConnectionUnixProcessID", &(DBUS_SERVICE_NAME)).await.ok()
}

fn daemon_exe() -> Option<PathBuf> {
    let pid = tokio::runtime::Runtime::new().ok()?.block_on(get_daemon_pid())?;
    fs::read_link(format!("/proc/{}/exe", pid)).ok()
}

/// Print a "Binary consistency" section: warn when more than one
/// `voice-dictation` is on PATH (PATH order silently picks the winner) and when
/// the running daemon is a different binary than this client (stale daemon after
/// an install). This is the most common cause of "the keybind does nothing".
fn check_binary_consistency() {
    println!("\nBinary consistency:");
    println!("  Version: {}", env!("CARGO_PKG_VERSION"));

    let self_exe = std::env::current_exe().ok();
    let self_canonical =
        self_exe.as_ref().map(|p| fs::canonicalize(p).unwrap_or_else(|_| p.clone()));

    let bins = binaries_on_path("voice-dictation");
    if bins.len() > 1 {
        println!("  WARNING: {} 'voice-dictation' binaries on PATH (first wins):", bins.len());
        for b in &bins {
            println!("    {}", b.display());
        }
        println!("  Keep only one — prefer ~/.local/bin (see README). Remove the others.");
    } else if let Some(b) = bins.first() {
        println!("  Client binary: {}", b.display());
    }

    match daemon_exe() {
        Some(daemon) => {
            println!("  Daemon binary: {}", daemon.display());
            let daemon_canonical = fs::canonicalize(&daemon).unwrap_or_else(|_| daemon.clone());
            if let Some(self_canonical) = &self_canonical {
                if *self_canonical != daemon_canonical {
                    println!("  WARNING: client and daemon are different binaries.");
                    println!("  Restart the daemon so it matches the installed binary:");
                    println!("    systemctl --user restart voice-dictation");
                }
            }
        }
        None => println!("  Daemon binary: (daemon not running)"),
    }
}

fn diagnose() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let config_path = PathBuf::from(&home).join(".config/voice-dictation/config.toml");

    println!("=== Voice Dictation Diagnostics ===\n");

    // Audio devices
    println!("Audio Input Devices:");
    for dev in utils::list_audio_devices() {
        let marker = if dev.is_default { " (default)" } else { "" };
        println!("  {}{}", dev.description, marker);
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
    println!(
        "\nDebug audio recording: {}",
        if debug_enabled || rust_log_debug { "enabled" } else { "disabled" }
    );
    if !debug_enabled && !rust_log_debug {
        println!("  Enable with: VOICE_DICTATION_DEBUG_AUDIO=1 voice-dictation daemon");
    } else {
        println!("  Recordings saved to: {}", DEBUG_DIR);
        let count = fs::read_dir(DEBUG_DIR)
            .ok()
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
                    .count()
            })
            .unwrap_or(0);
        println!("  Current recordings: {} (use 'voice-dictation debug list' to view)", count);
    }

    check_binary_consistency();

    Ok(())
}

fn download_model() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let model_dir = PathBuf::from(&home).join(".config/voice-dictation/models/parakeet");

    fs::create_dir_all(&model_dir)?;

    const BASE_URL: &str =
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main";
    const FILES: &[&str] =
        &["encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"];

    println!("Model directory: {}", model_dir.display());
    println!("Source: {}", BASE_URL);
    println!();

    let client = reqwest::blocking::Client::builder().timeout(None).build()?;

    for filename in FILES {
        let dest = model_dir.join(filename);

        if dest.exists() {
            let size = fs::metadata(&dest)?.len();
            if size > 0 {
                println!(
                    "  {} — already exists ({:.1} MB), skipping",
                    filename,
                    size as f64 / 1_048_576.0
                );
                continue;
            }
        }

        let url = format!("{}/{}", BASE_URL, filename);
        print!("  Downloading {}... ", filename);
        io::stdout().flush()?;

        let response = client.get(&url).send()?;
        if !response.status().is_success() {
            eprintln!("HTTP {}", response.status());
            return Err(
                format!("Failed to download {}: HTTP {}", filename, response.status()).into()
            );
        }

        let total = response.content_length();
        let bytes = response.bytes()?;
        let size = bytes.len();

        fs::write(&dest, &bytes)?;

        if let Some(t) = total {
            println!("{:.1} MB", t as f64 / 1_048_576.0);
        } else {
            println!("{:.1} MB", size as f64 / 1_048_576.0);
        }
    }

    println!();
    println!("Model download complete.");
    println!("You can now start the daemon: systemctl --user start voice-dictation");

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
            for dev in utils::list_audio_devices() {
                let marker = if dev.is_default { " (default)" } else { "" };
                println!("  {}{}", dev.description, marker);
            }
        }
        Commands::Debug { command } => match command {
            DebugCommands::List => debug_list()?,
            DebugCommands::Play { filename } => debug_play(&filename)?,
            DebugCommands::WeztermProbe => debug_wezterm_probe()?,
            DebugCommands::ImeProbe { secs, json, commit, commit_class } => {
                dictation_engine::ime_probe::run(dictation_engine::ime_probe::ProbeOptions {
                    secs,
                    json,
                    commit_text: commit,
                    commit_class,
                })?;
            }
        },
        Commands::Diagnose => diagnose()?,
        Commands::Dict { command } => dict_command(command)?,
        Commands::Subst { command } => subst_command(command)?,
        Commands::DownloadModel => download_model()?,
        Commands::TestLoop { recording } => {
            let rt = tokio::runtime::Runtime::new()?;
            let _guard = rt.enter();
            test_loop_ui::run(recording.as_deref())?;
        }
        Commands::SnapshotCorrection => {
            tokio::runtime::Runtime::new()?
                .block_on(call_dbus_method("SnapshotCorrection"))
                .map_err(dbus_error_with_hint)?;
            println!("Correction monitoring re-armed for the last dictation");
        }
        Commands::CorrectionStats => {
            let home = std::env::var("HOME")?;
            let corrections_path = std::path::PathBuf::from(&home)
                .join(".local/share/voice-dictation/corrections.json");
            if corrections_path.exists() {
                let content = fs::read_to_string(&corrections_path)?;
                let data: Value = serde_json::from_str(&content)?;
                if let Some(corrections) = data["corrections"].as_array() {
                    let total = corrections.len();
                    let promoted = corrections
                        .iter()
                        .filter(|c| c["promoted"].as_bool().unwrap_or(false))
                        .count();
                    let total_obs: u64 =
                        corrections.iter().filter_map(|c| c["count"].as_u64()).sum();
                    println!("Correction Learning Statistics:");
                    println!("  Unique corrections: {}", total);
                    println!("  Total observations: {}", total_obs);
                    println!("  Auto-promoted:      {}", promoted);
                    println!("  Pending:            {}", total - promoted);
                    println!();
                    if total > 0 {
                        println!("Top corrections:");
                        let mut sorted: Vec<&Value> = corrections.iter().collect();
                        sorted.sort_by(|a, b| {
                            b["count"].as_u64().unwrap_or(0).cmp(&a["count"].as_u64().unwrap_or(0))
                        });
                        for c in sorted.iter().take(10) {
                            let orig = c["original"].as_str().unwrap_or("?");
                            let corr = c["corrected"].as_str().unwrap_or("?");
                            let count = c["count"].as_u64().unwrap_or(0);
                            let prom = if c["promoted"].as_bool().unwrap_or(false) {
                                " [promoted]"
                            } else {
                                ""
                            };
                            println!("  {:>3}x  '{}' → '{}'{}", count, orig, corr, prom);
                        }
                    }
                } else {
                    println!("No corrections recorded yet.");
                }
            } else {
                println!(
                    "No corrections file found. Correction learning has not recorded any data yet."
                );
                println!("File location: {}", corrections_path.display());
            }
        }
        Commands::Corrections { command } => corrections_command(command)?,
    }

    Ok(())
}
