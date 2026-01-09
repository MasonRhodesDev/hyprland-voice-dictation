//! GPU detection for automatic CUDA acceleration
//!
//! Detects CUDA availability at runtime to auto-enable GPU acceleration
//! for Whisper without requiring user configuration.

use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use tracing::info;

/// Cached CUDA detection result
static CUDA_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check if CUDA is available for GPU acceleration
///
/// Results are cached after the first call for performance.
pub fn cuda_available() -> bool {
    *CUDA_AVAILABLE.get_or_init(|| detect_cuda())
}

/// Perform actual CUDA detection
fn detect_cuda() -> bool {
    // Method 1: Check for nvidia-smi (most reliable)
    if Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        info!("CUDA detected via nvidia-smi");
        return true;
    }

    // Method 2: Check for CUDA libraries
    let cuda_paths = [
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/lib/x86_64-linux-gnu/libcudart.so",
        "/usr/lib64/libcudart.so",
        "/opt/cuda/lib64/libcudart.so",
    ];

    for path in &cuda_paths {
        if Path::new(path).exists() {
            info!("CUDA detected via library: {}", path);
            return true;
        }
    }

    // Method 3: Check environment variables
    if std::env::var("CUDA_HOME").is_ok() || std::env::var("CUDA_PATH").is_ok() {
        info!("CUDA detected via environment variable");
        return true;
    }

    info!("CUDA not detected, using CPU");
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_detection_runs() {
        // Just verify it doesn't panic
        let _ = cuda_available();
    }

    #[test]
    fn test_cuda_detection_cached() {
        // Verify caching works (second call should be instant)
        let result1 = cuda_available();
        let result2 = cuda_available();
        assert_eq!(result1, result2);
    }
}
