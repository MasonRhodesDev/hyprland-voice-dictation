// Keyboard text injection via wtype

use anyhow::Result;
use tokio::time::{sleep, Duration};
use tracing::debug;

pub struct KeyboardInjector {
    typing_delay_ms: u64,
    word_delay_ms: u64,
}

impl KeyboardInjector {
    pub fn new(typing_delay_ms: u64, word_delay_ms: u64) -> Self {
        Self {
            typing_delay_ms,
            word_delay_ms,
        }
    }
    
    pub async fn type_text(&self, text: &str) -> Result<()> {
        debug!("Typing text: {}", text);
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            self.type_word(word).await?;
            
            // Add space between words (except last word)
            if i < words.len() - 1 {
                self.type_word(" ").await?;
                sleep(Duration::from_millis(self.word_delay_ms)).await;
            }
        }
        
        debug!("Finished typing {} words", words.len());
        Ok(())
    }
    
    async fn type_word(&self, word: &str) -> Result<()> {
        // Use wtype to inject text
        let output = tokio::process::Command::new("wtype")
            .arg(word)
            .output()
            .await?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("wtype failed for '{}': {}", word, stderr);
        }
        
        sleep(Duration::from_millis(self.typing_delay_ms)).await;
        Ok(())
    }
}
