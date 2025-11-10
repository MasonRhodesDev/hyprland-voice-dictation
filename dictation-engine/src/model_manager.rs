use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

const WHISPER_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Whisper model information.
pub struct WhisperModelInfo {
    pub filename: String,
    pub size_mb: u64,
}

impl WhisperModelInfo {
    /// Get model info for common Whisper models.
    pub fn get(model_name: &str) -> Option<Self> {
        match model_name {
            "ggml-tiny.en.bin" => Some(Self {
                filename: model_name.to_string(),
                size_mb: 75,
            }),
            "ggml-base.en.bin" => Some(Self {
                filename: model_name.to_string(),
                size_mb: 142,
            }),
            "ggml-small.en.bin" => Some(Self {
                filename: model_name.to_string(),
                size_mb: 466,
            }),
            "ggml-medium.en.bin" => Some(Self {
                filename: model_name.to_string(),
                size_mb: 1500,
            }),
            _ => None,
        }
    }
}

/// Check if a Whisper model exists at the specified path.
pub fn model_exists(model_path: &Path) -> bool {
    model_path.exists() && model_path.is_file()
}

/// Download a Whisper model from Hugging Face.
///
/// # Arguments
/// * `model_name` - Model filename (e.g., "ggml-small.en.bin")
/// * `dest_dir` - Destination directory (will be created if missing)
///
/// # Returns
/// * `Ok(PathBuf)` - Path to the downloaded model
/// * `Err` - Download or I/O error
pub fn download_whisper_model(model_name: &str, dest_dir: &Path) -> Result<PathBuf> {
    let model_info = WhisperModelInfo::get(model_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown Whisper model: {}", model_name))?;

    // Create destination directory
    fs::create_dir_all(dest_dir)?;

    let dest_path = dest_dir.join(&model_info.filename);

    // Check if already downloaded
    if model_exists(&dest_path) {
        info!("Model already exists: {}", dest_path.display());
        return Ok(dest_path);
    }

    // Download URL
    let url = format!("{}/{}", WHISPER_BASE_URL, model_info.filename);

    info!("Downloading Whisper model: {} (~{}MB)", model_name, model_info.size_mb);
    info!("From: {}", url);
    info!("To: {}", dest_path.display());

    // Download with progress bar
    let response = reqwest::blocking::get(&url)
        .map_err(|e| anyhow::anyhow!("Failed to download model: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Download failed with status: {}",
            response.status()
        ));
    }

    let total_size = response.content_length().unwrap_or(model_info.size_mb * 1024 * 1024);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Downloading {}", model_name));

    // Write to temp file first with progress
    let temp_path = dest_dir.join(format!("{}.tmp", model_info.filename));
    let mut dest_file = fs::File::create(&temp_path)?;

    use std::io::copy;
    let mut reader = pb.wrap_read(response);
    copy(&mut reader, &mut dest_file)?;

    pb.finish_with_message(format!("✓ Downloaded {}", model_name));

    // Atomic rename
    fs::rename(&temp_path, &dest_path)?;

    info!("✓ Model downloaded successfully: {}", dest_path.display());

    Ok(dest_path)
}

/// Ensure a Whisper model is available, downloading if necessary.
///
/// # Arguments
/// * `model_name` - Model filename
/// * `model_dir` - Model directory (will expand $HOME)
///
/// # Returns
/// * `Ok(PathBuf)` - Full path to the model file
/// * `Err` - If download failed or model unavailable
pub fn ensure_whisper_model(model_name: &str, model_dir: &str) -> Result<PathBuf> {
    // Expand $HOME and ~ in path
    let expanded_dir = shellexpand::full(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to expand path: {}", e))?
        .to_string();
    let dir_path = Path::new(&expanded_dir);

    let model_path = dir_path.join(model_name);

    if model_exists(&model_path) {
        info!("✓ Whisper model found: {}", model_path.display());
        return Ok(model_path);
    }

    warn!("Whisper model not found: {}", model_path.display());
    info!("Auto-downloading model (this may take a few minutes)...");

    download_whisper_model(model_name, dir_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info() {
        let info = WhisperModelInfo::get("ggml-small.en.bin");
        assert!(info.is_some());
        assert_eq!(info.unwrap().size_mb, 466);
    }

    #[test]
    fn test_unknown_model() {
        let info = WhisperModelInfo::get("invalid-model.bin");
        assert!(info.is_none());
    }

    #[test]
    fn test_model_exists() {
        // Test with a path that definitely doesn't exist
        let exists = model_exists(Path::new("/nonexistent/model.bin"));
        assert_eq!(exists, false);
    }
}
