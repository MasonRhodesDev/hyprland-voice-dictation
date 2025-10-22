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
        let output = tokio::process::Command::new("wtype")
            .arg(text)
            .output()
            .await?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("wtype failed: {}", stderr);
        }
        
        Ok(())
    }
    
    pub async fn _delete_chars(&self, count: usize) -> Result<()> {
        debug!("Deleting {} chars", count);
        
        // Use wtype to send backspace key presses
        let output = tokio::process::Command::new("wtype")
            .arg("-k")
            .arg("BackSpace")
            .arg("-P")
            .arg(count.to_string())
            .output()
            .await?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("wtype backspace failed: {}", stderr);
        }
        
        Ok(())
    }
    

}
