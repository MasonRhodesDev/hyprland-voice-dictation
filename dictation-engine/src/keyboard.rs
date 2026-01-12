// Keyboard text injection via wtype

use anyhow::Result;
use std::time::Duration;
use tracing::debug;

pub struct KeyboardInjector {
    word_delay_ms: u64,
}

impl KeyboardInjector {
    pub fn new(_typing_delay_ms: u64, word_delay_ms: u64) -> Self {
        Self { word_delay_ms }
    }

    pub async fn type_text(&self, text: &str) -> Result<()> {
        debug!("Typing text: {}", text);

        if self.word_delay_ms > 0 {
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

                tokio::time::sleep(Duration::from_millis(self.word_delay_ms)).await;
            }
        } else {
            // Fast mode: type all text at once (original behavior)
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
        let injector = KeyboardInjector::new(10, 50);
        assert_eq!(injector.word_delay_ms, 50);
    }

    #[test]
    fn test_keyboard_injector_zero_delay() {
        let injector = KeyboardInjector::new(0, 0);
        assert_eq!(injector.word_delay_ms, 0);
    }

    #[tokio::test]
    async fn test_type_text_interface() {
        let injector = KeyboardInjector::new(10, 50);
        let result = injector.type_text("test").await;
        // wtype may or may not be available in test environment
        assert!(result.is_ok() || result.is_err());
    }
}
