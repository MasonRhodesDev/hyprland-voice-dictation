use anyhow::{Context, Result};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

fn data_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME not set")?;
    Ok(PathBuf::from(home).join(".local").join("share").join("voice-dictation"))
}

pub fn append_substitution(spoken: &str, replacement: &str) -> Result<()> {
    let dir = data_dir()?;
    fs::create_dir_all(&dir)?;
    let path = dir.join("substitutions.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    writeln!(file, "{} -> {}", spoken, replacement)?;
    Ok(())
}

pub fn append_hotword(word: &str, score: f32) -> Result<()> {
    let dir = data_dir()?;
    fs::create_dir_all(&dir)?;
    let path = dir.join("hotwords.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    writeln!(file, "{} {}", word, score)?;
    Ok(())
}
