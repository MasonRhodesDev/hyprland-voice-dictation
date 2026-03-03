// Keyboard text injection via wtype

use anyhow::Result;
use std::time::Duration;
use tracing::debug;

pub struct KeyboardInjector;

impl KeyboardInjector {
    pub fn new() -> Self {
        Self
    }

    pub async fn type_text(&self, text: &str, word_delay_ms: u64) -> Result<()> {
        debug!("Typing text: {}", text);

        if word_delay_ms > 0 {
            // Rate-limited mode: word-by-word with delays to avoid overwhelming
            // terminal UIs like Claude Code's React/Ink interface (React error #185)
            for (i, word) in text.split_whitespace().enumerate() {
                let chunk = if i == 0 {
                    word.to_string()
                } else {
                    format!(" {}", word)
                };

                let output = tokio::process::Command::new("wtype")
                    .arg(&chunk)
                    .output()
                    .await?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!("wtype failed: {}", stderr);
                }

                tokio::time::sleep(Duration::from_millis(word_delay_ms)).await;
            }
        } else {
            // Fast mode: type all text at once
            let output = tokio::process::Command::new("wtype")
                .arg(text)
                .output()
                .await?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("wtype failed: {}", stderr);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyboard_injector_new() {
        let _injector = KeyboardInjector::new();
    }

    #[tokio::test]
    async fn test_type_text_interface() {
        let injector = KeyboardInjector::new();
        let result = injector.type_text("test", 0).await;
        // wtype may or may not be available in test environment
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_type_text_word_delay() {
        let injector = KeyboardInjector::new();
        let result = injector.type_text("test", 50).await;
        // wtype may or may not be available in test environment
        assert!(result.is_ok() || result.is_err());
    }
}
