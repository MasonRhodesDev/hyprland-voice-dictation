use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use serde::Deserialize;
use std::fs;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use vosk::{Model, Recognizer};

pub mod control_ipc;
pub mod ipc;
mod keyboard;

use control_ipc::{ControlMessage, ControlServer};
use keyboard::KeyboardInjector;

const CONTROL_SOCKET_PATH: &str = "/tmp/voice-dictation-control.sock";
const AUDIO_SOCKET_PATH: &str = "/tmp/voice-dictation.sock";

#[derive(Debug, Deserialize)]
struct Config {
    daemon: DaemonConfig,
}

#[derive(Debug, Deserialize)]
struct DaemonConfig {
    audio_device: String,
    sample_rate: String,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default = "default_preview_model")]
    preview_model: String,
    #[serde(default = "default_preview_model_custom_path")]
    preview_model_custom_path: String,
    #[serde(default = "default_final_model")]
    final_model: String,
    #[serde(default = "default_final_model_custom_path")]
    final_model_custom_path: String,
}

fn default_language() -> String { "en".to_string() }
fn default_preview_model() -> String { "vosk-model-en-us-daanzu-20200905-lgraph".to_string() }
fn default_preview_model_custom_path() -> String { 
    std::env::var("HOME")
        .map(|h| format!("{}/.config/voice-dictation/models/", h))
        .unwrap_or_else(|_| "./models/".to_string())
}
fn default_final_model() -> String { "vosk-model-en-us-0.22".to_string() }
fn default_final_model_custom_path() -> String { 
    std::env::var("HOME")
        .map(|h| format!("{}/.config/voice-dictation/models/", h))
        .unwrap_or_else(|_| "./models/".to_string())
}

fn load_config() -> Result<Config> {
    let home = std::env::var("HOME")?;
    let config_path = format!("{}/.config/voice-dictation/config.toml", home);
    
    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", config_path, e))?;
    
    let config: Config = toml::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
    
    Ok(config)
}

struct VoskEngine {
    recognizer: Arc<Mutex<Recognizer>>,
    accumulated_text: Arc<Mutex<String>>,
    audio_buffer: Arc<Mutex<Vec<i16>>>,
}

pub fn remove_duplicate_suffix(accumulated: &str, new_chunk: &str) -> String {
    let acc_words: Vec<&str> = accumulated.split_whitespace().collect();
    let new_words: Vec<&str> = new_chunk.split_whitespace().collect();

    if acc_words.is_empty() || new_words.is_empty() {
        return new_chunk.to_string();
    }

    for overlap_len in (1..=acc_words.len().min(new_words.len())).rev() {
        let acc_suffix = &acc_words[acc_words.len() - overlap_len..];
        let new_prefix = &new_words[..overlap_len];

        if acc_suffix == new_prefix {
            return new_words[overlap_len..].join(" ");
        }
    }

    new_chunk.to_string()
}

impl VoskEngine {
    fn new(model_path: &str, sample_rate: u32) -> Result<Self> {
        info!("Loading Vosk model from {}", model_path);
        let model =
            Model::new(model_path).ok_or_else(|| anyhow::anyhow!("Failed to load model"))?;
        let mut recognizer = Recognizer::new(&model, sample_rate as f32)
            .ok_or_else(|| anyhow::anyhow!("Failed to create recognizer"))?;

        let silence = vec![0i16; sample_rate as usize / 10];
        let _ = recognizer.accept_waveform(&silence);

        Ok(Self {
            recognizer: Arc::new(Mutex::new(recognizer)),
            accumulated_text: Arc::new(Mutex::new(String::new())),
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        let mut audio_buffer = self.audio_buffer.lock().unwrap();
        audio_buffer.extend_from_slice(samples);
        drop(audio_buffer);

        let mut recognizer = self.recognizer.lock().unwrap();
        let state = recognizer.accept_waveform(samples)?;

        if state == vosk::DecodingState::Finalized {
            let result = recognizer.result();
            if let Some(finalized) = result.single() {
                let text = finalized.text.to_string().trim().to_string();
                if !text.is_empty() {
                    let mut accumulated = self.accumulated_text.lock().unwrap();

                    let deduplicated = remove_duplicate_suffix(&accumulated, &text);

                    if !deduplicated.is_empty() {
                        if !accumulated.is_empty() {
                            accumulated.push(' ');
                        }
                        accumulated.push_str(&deduplicated);
                        info!("Accumulated chunk: '{}'", deduplicated);
                    }
                }
            }
        }

        Ok(())
    }

    fn run_correction_pass(&self, accurate_model: &Model, sample_rate: u32) -> Result<String> {
        info!("Running correction pass with accurate model...");

        let mut accurate_recognizer = Recognizer::new(accurate_model, sample_rate as f32)
            .ok_or_else(|| anyhow::anyhow!("Failed to create accurate recognizer"))?;

        let audio_buffer = self.audio_buffer.lock().unwrap();

        const CHUNK_SIZE: usize = 8000;
        for chunk in audio_buffer.chunks(CHUNK_SIZE) {
            accurate_recognizer.accept_waveform(chunk)?;
        }

        let result = accurate_recognizer.final_result();
        if let Some(text) = result.single().map(|r| r.text.to_string()) {
            Ok(text.trim().to_string())
        } else {
            Ok(String::new())
        }
    }

    fn get_current_full_text(&self) -> Result<String> {
        let mut recognizer = self.recognizer.lock().unwrap();
        let accumulated = self.accumulated_text.lock().unwrap();

        let partial_result = recognizer.partial_result();
        let partial = partial_result.partial.to_string().trim().to_string();

        if partial.is_empty() {
            Ok(accumulated.clone())
        } else if accumulated.is_empty() {
            Ok(partial)
        } else {
            Ok(format!("{} {}", accumulated, partial))
        }
    }

    fn get_final_result(&self) -> Result<String> {
        let mut recognizer = self.recognizer.lock().unwrap();
        let mut accumulated = self.accumulated_text.lock().unwrap();

        let result = recognizer.final_result();
        if let Some(final_chunk) = result.single() {
            let text = final_chunk.text.to_string().trim().to_string();
            if !text.is_empty() {
                if !accumulated.is_empty() {
                    accumulated.push(' ');
                }
                accumulated.push_str(&text);
            }
        }

        Ok(accumulated.clone())
    }
}

struct AudioCapture {
    stream: Option<Stream>,
}

impl AudioCapture {
    fn new(tx: mpsc::UnboundedSender<Vec<i16>>, device_name: Option<&str>, sample_rate: u32) -> Result<Self> {
        let host = cpal::default_host();
        
        info!("Available audio input devices from cpal:");
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if let Ok(name) = device.name() {
                    info!("  - '{}'", name);
                }
            }
        }
        
        let device = if let Some(name) = device_name {
            info!("Searching for configured device: '{}'", name);
            if name == "default" {
                info!("Using default audio input device");
                host.default_input_device().ok_or_else(|| anyhow::anyhow!("No default input device"))?
            } else {
                info!("Searching for audio device: {}", name);
                let mut found_device = None;
                
                for device in host.input_devices()? {
                    if let Ok(device_name) = device.name() {
                        if device_name == name {
                            found_device = Some(device);
                            break;
                        }
                    }
                }
                
                found_device.ok_or_else(|| {
                    warn!("Configured device '{}' not found, falling back to default", name);
                    anyhow::anyhow!("Audio device '{}' not found", name)
                }).or_else(|_| {
                    host.default_input_device().ok_or_else(|| anyhow::anyhow!("No input device available"))
                })?
            }
        } else {
            info!("No device configured, using default");
            host.default_input_device().ok_or_else(|| anyhow::anyhow!("No default input device"))?
        };

        info!("Using input device: {}", device.name()?);

        let config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let samples: Vec<i16> =
                    data.iter().map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16).collect();
                let _ = tx.send(samples);
            },
            |err| error!("Audio stream error: {}", err),
            None,
        )?;

        Ok(Self { stream: Some(stream) })
    }

    fn start(&self) -> Result<()> {
        if let Some(stream) = &self.stream {
            stream.play()?;
            info!("Audio capture started");
        }
        Ok(())
    }
}

#[tokio::main]
pub async fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    info!("Starting preview-based Vosk dictation engine");

    let config = load_config().unwrap_or_else(|e| {
        warn!("Failed to load config: {}, using defaults", e);
        Config {
            daemon: DaemonConfig {
                audio_device: "default".to_string(),
                sample_rate: "16000".to_string(),
                language: default_language(),
                preview_model: default_preview_model(),
                preview_model_custom_path: default_preview_model_custom_path(),
                final_model: default_final_model(),
                final_model_custom_path: default_final_model_custom_path(),
            }
        }
    });

    let sample_rate: u32 = config.daemon.sample_rate.parse()
        .unwrap_or_else(|_| {
            warn!("Invalid sample_rate '{}', defaulting to 16000", config.daemon.sample_rate);
            16000
        });

    info!("Config loaded: audio_device={}, sample_rate={}, language={}", 
          config.daemon.audio_device, sample_rate, config.daemon.language);

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    
    let preview_model_path = if config.daemon.preview_model == "custom" {
        config.daemon.preview_model_custom_path.replace("$HOME", &home)
    } else {
        let base_path = format!("{}/.config/voice-dictation/models", home);
        format!("{}/{}", base_path, config.daemon.preview_model)
    };
    
    let final_model_path = if config.daemon.final_model == "custom" {
        config.daemon.final_model_custom_path.replace("$HOME", &home)
    } else {
        let base_path = format!("{}/.config/voice-dictation/models", home);
        format!("{}/{}", base_path, config.daemon.final_model)
    };

    let (audio_tx, mut audio_rx) = mpsc::unbounded_channel();

    let audio_device = if config.daemon.audio_device.is_empty() {
        None
    } else {
        let device_str = config.daemon.audio_device.as_str();
        let device_name = if let Some(idx) = device_str.find(" (") {
            &device_str[..idx]
        } else {
            device_str
        };
        Some(device_name)
    };

    let capture = AudioCapture::new(audio_tx, audio_device, sample_rate)?;
    capture.start()?;
    info!("Audio capture started - buffering...");

    info!("Loading fast model for live preview from: {}", preview_model_path);
    let engine = Arc::new(VoskEngine::new(&preview_model_path, sample_rate)?);
    let keyboard = Arc::new(KeyboardInjector::new(10, 50));

    info!("Preloading accurate model in background from: {}", final_model_path);
    let accurate_model_handle = tokio::spawn(async move { Model::new(&final_model_path) });

    let audio_ipc = Arc::new(ipc::IpcServer::new(AUDIO_SOCKET_PATH.to_string()));
    audio_ipc.start_server();

    let mut control_server = ControlServer::new(CONTROL_SOCKET_PATH).await?;

    info!("Ready - waiting for GUI to connect");

    control_server.broadcast(&ControlMessage::Ready).await?;

    info!("Recording... (waiting for Confirm command)");

    let mut startup_buffer = Vec::new();
    while let Ok(samples) = audio_rx.try_recv() {
        startup_buffer.push(samples);
    }

    if !startup_buffer.is_empty() {
        info!("Processing {} buffered audio chunks from startup", startup_buffer.len());
        for samples in startup_buffer {
            if let Err(e) = engine.process_audio(&samples) {
                error!("Processing buffered audio error: {}", e);
            }
        }
    }

    let engine_clone = Arc::clone(&engine);
    let audio_ipc_clone = Arc::clone(&audio_ipc);
    let audio_task = tokio::spawn(async move {
        let mut buffer = Vec::new();

        while let Some(samples) = audio_rx.recv().await {
            let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

            buffer.extend_from_slice(&samples_f32);

            while buffer.len() >= 512 {
                let chunk: Vec<f32> = buffer.drain(..512).collect();
                audio_ipc_clone.broadcast_samples(&chunk).await;
            }

            if let Err(e) = engine_clone.process_audio(&samples) {
                error!("Processing error: {}", e);
            }
        }
    });

    let engine_clone = Arc::clone(&engine);
    let control_server_shared = Arc::new(tokio::sync::Mutex::new(control_server));
    let control_clone_for_preview = Arc::clone(&control_server_shared);
    let preview_task = tokio::spawn(async move {
        let mut check_interval = tokio::time::interval(std::time::Duration::from_millis(200));

        loop {
            check_interval.tick().await;

            let mut server = control_clone_for_preview.lock().await;
            server.try_accept().await;
            drop(server);

            match engine_clone.get_current_full_text() {
                Ok(text_curr) => {
                    let mut server = control_clone_for_preview.lock().await;
                    let _ = server
                        .broadcast(&ControlMessage::TranscriptionUpdate {
                            text: text_curr,
                            is_final: false,
                        })
                        .await;
                }
                Err(e) => error!("Failed to get text: {}", e),
            }
        }
    });

    loop {
        let mut server = control_server_shared.lock().await;
        server.try_accept().await;

        if let Some(ControlMessage::Confirm) = server.receive_from_any().await {
            info!("Received Confirm command");
            drop(server);
            break;
        }
        drop(server);

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    info!("Stopping recording...");

    audio_task.abort();
    preview_task.abort();

    let fast_result = engine.get_final_result()?;
    info!("Fast model result: '{}'", fast_result);

    let mut server = control_server_shared.lock().await;
    server
        .broadcast(&ControlMessage::TranscriptionUpdate {
            text: fast_result.clone(),
            is_final: true,
        })
        .await?;
    drop(server);

    if !fast_result.is_empty() {
        let mut server = control_server_shared.lock().await;
        server.broadcast(&ControlMessage::ProcessingStarted).await?;
        drop(server);

        info!("Waiting for accurate model to finish loading...");
        let accurate_model = accurate_model_handle
            .await?
            .ok_or_else(|| anyhow::anyhow!("Failed to load accurate model"))?;

        info!("Running correction pass...");
        let accurate_result = engine.run_correction_pass(&accurate_model, sample_rate)?;
        info!("Accurate model result: '{}'", accurate_result);

        info!("Typing final text...");
        keyboard.type_text(&accurate_result).await?;
        info!("✓ Typed!");

        let mut server = control_server_shared.lock().await;
        server.broadcast(&ControlMessage::Complete).await?;
        drop(server);

        tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
    } else {
        info!("No text to type");

        let mut server = control_server_shared.lock().await;
        server.broadcast(&ControlMessage::Complete).await?;
        drop(server);

        tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_duplicate_suffix_no_overlap() {
        let result = remove_duplicate_suffix("hello world", "foo bar");
        assert_eq!(result, "foo bar");
    }

    #[test]
    fn test_remove_duplicate_suffix_full_overlap() {
        let result = remove_duplicate_suffix("hello world", "hello world");
        assert_eq!(result, "");
    }

    #[test]
    fn test_remove_duplicate_suffix_partial_overlap() {
        let result = remove_duplicate_suffix("hello world", "world this is new");
        assert_eq!(result, "this is new");
    }

    #[test]
    fn test_remove_duplicate_suffix_multi_word_overlap() {
        let result = remove_duplicate_suffix("the quick brown", "quick brown fox");
        assert_eq!(result, "fox");
    }

    #[test]
    fn test_remove_duplicate_suffix_empty_accumulated() {
        let result = remove_duplicate_suffix("", "hello world");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_remove_duplicate_suffix_empty_new() {
        let result = remove_duplicate_suffix("hello world", "");
        assert_eq!(result, "");
    }

    #[test]
    fn test_remove_duplicate_suffix_both_empty() {
        let result = remove_duplicate_suffix("", "");
        assert_eq!(result, "");
    }

    #[test]
    fn test_remove_duplicate_suffix_single_word_overlap() {
        let result = remove_duplicate_suffix("test", "test again");
        assert_eq!(result, "again");
    }

    #[test]
    fn test_remove_duplicate_suffix_no_match_similar() {
        let result = remove_duplicate_suffix("hello world", "hello universe");
        assert_eq!(result, "hello universe");
    }

    #[test]
    fn test_remove_duplicate_suffix_longer_overlap() {
        let result = remove_duplicate_suffix("one two three four", "two three four five six");
        assert_eq!(result, "five six");
    }
}
