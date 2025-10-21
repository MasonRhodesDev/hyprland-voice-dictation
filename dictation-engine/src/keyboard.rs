// Keyboard text injection via wtype

use anyhow::Result;
use std::process::Command;
use tokio::time::{sleep, Duration};

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
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            self.type_word(word).await?;
            
            // Add space between words (except last word)
            if i < words.len() - 1 {
                self.type_word(" ").await?;
                sleep(Duration::from_millis(self.word_delay_ms)).await;
            }
        }
        
        Ok(())
    }
    
    async fn type_word(&self, word: &str) -> Result<()> {
        // Use wtype to inject text
        let output = Command::new("wtype")
            .arg(word)
            .output()?;
        
        if !output.status.success() {
            anyhow::bail!("wtype failed: {:?}", output.stderr);
        }
        
        sleep(Duration::from_millis(self.typing_delay_ms)).await;
        Ok(())
    }
}
