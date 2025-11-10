use clap::{Parser, Subcommand};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;
use schema_tui::SchemaTUIBuilder;
use zbus::Connection;

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
    let state = get_state();
    println!("State: {}", state);

    if state != "stopped" {
        if is_process_running("voice-dictation daemon") {
            println!("  Daemon: running");
        } else {
            println!("  Daemon: NOT running");
        }
    }
}

fn check_model_exists(model_name: &str, models_dir: &PathBuf) -> bool {
    if model_name == "custom" {
        return true;
    }
    models_dir.join(model_name).exists()
}

fn get_model_url(model_name: &str) -> String {
    format!("https://alphacephei.com/vosk/models/{}.zip", model_name)
}

fn download_model(model_name: &str, models_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let url = get_model_url(model_name);
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
    
    let mut missing_models = Vec::new();
    
    if let Some(model) = &preview_model {
        if !check_model_exists(model, &models_dir) {
            missing_models.push(("Preview", model.clone()));
        }
    }
    
    if let Some(model) = &final_model {
        if !check_model_exists(model, &models_dir) {
            missing_models.push(("Final", model.clone()));
        }
    }
    
    if missing_models.is_empty() {
        return Ok(());
    }
    
    println!("\n⚠️  Missing models detected:");
    for (model_type, model_name) in &missing_models {
        println!("  - {} model: {}", model_type, model_name);
        println!("    URL: {}", get_model_url(model_name));
    }
    
    print!("\nWould you like to download missing models now? [y/N]: ");
    io::stdout().flush()?;
    
    let mut response = String::new();
    io::stdin().read_line(&mut response)?;
    
    if response.trim().to_lowercase() == "y" {
        for (model_type, model_name) in &missing_models {
            println!("\nDownloading {} model: {}", model_type, model_name);
            if let Err(e) = download_model(model_name, &models_dir) {
                eprintln!("✗ Failed to download {}: {}", model_name, e);
                eprintln!("  Please download manually from: {}", get_model_url(model_name));
            }
        }
    } else {
        println!("\nSkipping download. You can download models manually with:");
        println!("  cd ~/.config/voice-dictation/models");
        for (_, model_name) in &missing_models {
            println!("  curl -L -O {}", get_model_url(model_name));
            println!("  unzip {}.zip", model_name);
        }
    }
    
    Ok(())
}

fn open_config() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let config_dir = PathBuf::from(&home).join(".config/voice-dictation");
    let config_path = config_dir.join("config.toml");
    let schema_path = PathBuf::from(&home)
        .join("repos/voice-dictation-rust/config-schema.json");

    if !config_dir.exists() {
        fs::create_dir_all(&config_dir)?;
    }

    if !config_path.exists() {
        fs::write(&config_path, "")?;
    }

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
            dictation_engine::run()?;
        }
        Commands::Start => {
            start_recording()?;
        }
        Commands::Stop => {
            stop_recording()?;
        }
        Commands::Confirm => {
            confirm_recording()?;
        }
        Commands::Toggle => {
            toggle_recording()?;
        }
        Commands::Status => {
            show_status();
        }
        Commands::Config => {
            open_config()?;
        }
    }

    Ok(())
}
