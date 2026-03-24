use anyhow::{Context, Result};
use hound::WavReader;
use serde::Deserialize;
use std::fs;
use std::path::Path;

use dictation_engine::model_selector::ModelSpec;
use dictation_engine::post_processing::Pipeline;

#[derive(Debug, Deserialize)]
struct Config {
    daemon: DaemonConfig,
}

#[derive(Debug, Deserialize)]
struct DaemonConfig {
    #[serde(default = "default_model", alias = "preview_model")]
    model: String,
    #[serde(default = "default_enable_acronyms")]
    enable_acronyms: bool,
    #[serde(default = "default_enable_punctuation")]
    enable_punctuation: bool,
    #[serde(default = "default_enable_grammar")]
    enable_grammar: bool,
}

fn default_model() -> String {
    "parakeet:default".to_string()
}
fn default_enable_acronyms() -> bool {
    true
}
fn default_enable_punctuation() -> bool {
    true
}
fn default_enable_grammar() -> bool {
    true
}

fn load_config() -> Result<Config> {
    let home = std::env::var("HOME").context("HOME not set")?;
    let config_path = std::path::PathBuf::from(home)
        .join(".config")
        .join("voice-dictation")
        .join("config.toml");

    if !config_path.exists() {
        return Ok(Config {
            daemon: DaemonConfig {
                model: default_model(),
                enable_acronyms: true,
                enable_punctuation: true,
                enable_grammar: true,
            },
        });
    }

    let content = fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: Config = toml::from_str(&content).context("parsing config.toml")?;
    Ok(config)
}

pub fn rerun_on_wav(wav_path: &Path) -> Result<String> {
    let config = load_config()?;

    let spec = ModelSpec::parse(&config.daemon.model)?;
    let engine = spec.create_engine(16000)?;

    let mut reader = WavReader::open(wav_path)
        .with_context(|| format!("opening WAV {}", wav_path.display()))?;
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<std::result::Result<_, _>>()
        .context("reading WAV samples")?;

    engine.process_audio(&samples)?;
    let raw = engine.get_final_result()?;

    let pipeline = Pipeline::from_config(
        config.daemon.enable_acronyms,
        config.daemon.enable_punctuation,
        config.daemon.enable_grammar,
    );
    let processed = pipeline.process(&raw)?;

    Ok(processed)
}
