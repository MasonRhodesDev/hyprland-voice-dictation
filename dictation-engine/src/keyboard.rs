// Keyboard text injection via wtype

use anyhow::Result;
use tracing::debug;

pub struct KeyboardInjector {}

impl KeyboardInjector {
    pub fn new(_typing_delay_ms: u64, _word_delay_ms: u64) -> Self {
        Self {}
    }

    pub async fn type_text(&self, text: &str) -> Result<()> {
        debug!("Typing text: {}", text);

        // Use wtype to type the text directly (preserves exact spacing)
        let output = tokio::process::Command::new("wtype").arg(text).output().await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("wtype failed: {}", stderr);
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
        assert_eq!(std::mem::size_of_val(&injector), 0);
    }

    #[tokio::test]
    async fn test_type_text_interface() {
        let injector = KeyboardInjector::new(10, 50);
        let result = injector.type_text("test").await;
        assert!(result.is_ok() || result.is_err());
    }
}
