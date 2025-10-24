use clap::{Parser, Subcommand};
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use schema_tui::SchemaTUIBuilder;

const STATE_FILE: &str = "/tmp/voice-dictation-state";
const MEDIA_STATE_FILE: &str = "/tmp/voice-dictation-media-state";
const CONTROL_SOCKET: &str = "/tmp/voice-dictation-control.sock";

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
    #[command(about = "Start the GUI overlay")]
    Gui,
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

fn send_confirm() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(CONTROL_SOCKET)?;
    let msg = r#"{"Confirm":null}"#;
    let length = (msg.len() as u32).to_be_bytes();

    stream.write_all(&length)?;
    stream.write_all(msg.as_bytes())?;
    stream.flush()?;

    Ok(())
}

fn start_recording() -> Result<(), Box<dyn std::error::Error>> {
    let state = get_state();
    if state != "stopped" {
        println!("Voice dictation already running (state: {})", state);
        return Ok(());
    }

    let media_playing = Command::new("playerctl")
        .arg("status")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|status| status.contains("Playing"))
        .unwrap_or(false);

    if media_playing {
        fs::write(MEDIA_STATE_FILE, "playing")?;
    } else {
        fs::write(MEDIA_STATE_FILE, "stopped")?;
    }

    let current_exe = std::env::current_exe()?;
    let work_dir = PathBuf::from(std::env::var("HOME")?).join("repos/voice-dictation-rust");
    
    let daemon_log = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("/tmp/dictation-engine.log")?;
    let daemon_log_stderr = daemon_log.try_clone()?;
    
    Command::new(&current_exe)
        .arg("daemon")
        .current_dir(&work_dir)
        .env("RUST_LOG", "info")
        .stdout(Stdio::from(daemon_log))
        .stderr(Stdio::from(daemon_log_stderr))
        .spawn()?;
    
    thread::sleep(Duration::from_millis(300));
    
    let gui_log = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("/tmp/dictation-gui.log")?;
    let gui_log_stderr = gui_log.try_clone()?;
    
    Command::new(&current_exe)
        .arg("gui")
        .current_dir(&work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(gui_log))
        .stderr(Stdio::from(gui_log_stderr))
        .spawn()?;

    let _ = Command::new("playerctl").arg("pause").output();

    set_state("recording")?;
    println!("Voice dictation started - recording");

    Ok(())
}

fn stop_recording() -> Result<(), Box<dyn std::error::Error>> {
    let state = get_state();
    if state == "stopped" {
        println!("Voice dictation not running");
        return Ok(());
    }

    let _ = Command::new("pkill").arg("-TERM").arg("-f").arg("voice-dictation daemon").output();

    let _ = Command::new("pkill").arg("-TERM").arg("-f").arg("voice-dictation gui").output();

    if let Ok(media_state) = fs::read_to_string(MEDIA_STATE_FILE) {
        if media_state.trim() == "playing" {
            let _ = Command::new("playerctl").arg("play").output();
        }
    }
    let _ = fs::remove_file(MEDIA_STATE_FILE);

    set_state("stopped")?;
    let _ = fs::remove_file(CONTROL_SOCKET);
    println!("Voice dictation stopped");

    Ok(())
}

fn confirm_recording() -> Result<(), Box<dyn std::error::Error>> {
    let state = get_state();
    if state != "recording" {
        eprintln!("Not in recording state (current: {})", state);
        return Err("Invalid state".into());
    }

    println!("Sending confirm command...");
    send_confirm()?;

    for _ in 0..60 {
        if !is_process_running("voice-dictation daemon") {
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }

    let _ = Command::new("pkill").arg("-TERM").arg("-f").arg("voice-dictation gui").output();

    if let Ok(media_state) = fs::read_to_string(MEDIA_STATE_FILE) {
        if media_state.trim() == "playing" {
            let _ = Command::new("playerctl").arg("play").output();
        }
    }
    let _ = fs::remove_file(MEDIA_STATE_FILE);

    set_state("stopped")?;
    let _ = fs::remove_file(CONTROL_SOCKET);
    println!("Transcription confirmed - typed successfully");

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
            println!("  Engine: running");
        } else {
            println!("  Engine: NOT running");
        }

        if is_process_running("voice-dictation gui") {
            println!("  GUI: running");
        } else {
            println!("  GUI: NOT running");
        }
    }
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

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon => {
            dictation_engine::run()?;
        }
        Commands::Gui => {
            dictation_gui::run()?;
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
