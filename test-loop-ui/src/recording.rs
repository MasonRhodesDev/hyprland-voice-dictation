use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

const DEBUG_DIR: &str = "/tmp/voice-dictation-debug";

#[derive(Debug, Deserialize, Clone)]
pub struct AudioMetadata {
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub sample_rate: u32,
    pub sample_count: usize,
    pub devices: Vec<String>,
    pub active_device: Option<String>,
    pub preview_text: String,
    pub final_text: String,
    pub preview_engine: String,
    pub accurate_engine: String,
    pub same_model_used: bool,
}

#[derive(Debug, Clone)]
pub struct Recording {
    pub wav_path: PathBuf,
    pub json_path: PathBuf,
    pub metadata: AudioMetadata,
}

pub fn list_recordings() -> Result<Vec<Recording>> {
    let debug_dir = Path::new(DEBUG_DIR);
    if !debug_dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<_> = fs::read_dir(debug_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .collect();

    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut recordings = Vec::new();
    for entry in entries {
        let json_path = entry.path();
        let wav_path = json_path.with_extension("wav");
        if !wav_path.exists() {
            continue;
        }
        let content = fs::read_to_string(&json_path)
            .with_context(|| format!("reading {}", json_path.display()))?;
        let metadata: AudioMetadata = serde_json::from_str(&content)
            .with_context(|| format!("parsing {}", json_path.display()))?;
        recordings.push(Recording {
            wav_path,
            json_path,
            metadata,
        });
    }

    Ok(recordings)
}

pub fn most_recent_recording() -> Result<Option<Recording>> {
    let mut recordings = list_recordings()?;
    Ok(recordings.pop())
}

pub fn recording_by_path(wav_path: &Path) -> Result<Recording> {
    let json_path = wav_path.with_extension("json");
    let content = fs::read_to_string(&json_path)
        .with_context(|| format!("reading {}", json_path.display()))?;
    let metadata: AudioMetadata = serde_json::from_str(&content)
        .with_context(|| format!("parsing {}", json_path.display()))?;
    Ok(Recording {
        wav_path: wav_path.to_owned(),
        json_path,
        metadata,
    })
}
