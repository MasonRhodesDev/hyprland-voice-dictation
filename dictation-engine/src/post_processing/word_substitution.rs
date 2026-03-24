use super::TextProcessor;
use crate::user_dictionary::UserDictionary;
use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Word substitution post-processor.
///
/// Maps spoken phrases to desired output text (e.g., "shay moy" → "chezmoi").
/// Substitutions are loaded from `~/.local/share/voice-dictation/substitutions.txt`.
///
/// File format:
/// ```text
/// # Comments start with #
/// shay moy -> chezmoi
/// cube cuttle -> kubectl
/// ```
#[derive(Clone)]
pub struct WordSubstitutionProcessor {
    /// Substitution entries: (spoken words split into vec, replacement string).
    /// Sorted by descending spoken-form word count (longest match first).
    entries: Arc<RwLock<Vec<(Vec<String>, String)>>>,
    /// Optional user dictionary to register replacement words with.
    user_dict: Option<Arc<UserDictionary>>,
}

impl WordSubstitutionProcessor {
    /// Create a new word substitution processor, loading from the default substitutions file.
    ///
    /// Replacement words are added to the user dictionary so grammar checking won't flag them.
    pub fn new(user_dict: Option<Arc<UserDictionary>>) -> Result<Self> {
        let path = Self::get_substitutions_path()?;
        let entries = Self::load_substitutions(&path)?;

        // Register replacement words in user dictionary
        if let Some(ref dict) = user_dict {
            for (_, replacement) in &entries {
                for word in replacement.split_whitespace() {
                    let _ = dict.add(word);
                }
            }
        }

        Ok(Self {
            entries: Arc::new(RwLock::new(entries)),
            user_dict,
        })
    }

    /// Create an empty processor with no substitutions (fallback).
    pub fn empty() -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            user_dict: None,
        }
    }

    /// Reload substitutions from disk and re-sync to user dictionary.
    pub fn reload(&self) -> Result<()> {
        let path = Self::get_substitutions_path()?;
        let new_entries = Self::load_substitutions(&path)?;

        // Register replacement words in user dictionary
        if let Some(ref dict) = self.user_dict {
            for (_, replacement) in &new_entries {
                for word in replacement.split_whitespace() {
                    let _ = dict.add(word);
                }
            }
        }

        let mut entries = self
            .entries
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        *entries = new_entries;
        Ok(())
    }

    /// Returns the file path for the substitutions file (for the watcher).
    pub fn watch_path() -> PathBuf {
        Self::get_substitutions_path().unwrap_or_default()
    }

    /// Returns the path to the substitutions file.
    pub fn get_substitutions_path() -> Result<PathBuf> {
        let data_dir = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;

        let path = data_dir
            .join("voice-dictation")
            .join("substitutions.txt");

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        Ok(path)
    }

    /// Parse a substitutions file.
    ///
    /// Returns entries sorted by descending spoken-form word count (longest match first).
    pub fn load_substitutions(path: &Path) -> Result<Vec<(Vec<String>, String)>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(path)?;
        let mut entries: Vec<(Vec<String>, String)> = content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .filter_map(|line| {
                let parts: Vec<&str> = line.splitn(2, "->").collect();
                if parts.len() != 2 {
                    return None;
                }
                let spoken = parts[0].trim();
                let replacement = parts[1].trim();
                if spoken.is_empty() || replacement.is_empty() {
                    return None;
                }
                let spoken_words: Vec<String> = spoken
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
                    .collect();
                Some((spoken_words, replacement.to_string()))
            })
            .collect();

        // Sort by descending spoken-form word count (longest match first)
        entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(entries)
    }
}

impl TextProcessor for WordSubstitutionProcessor {
    fn process(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let entries = self
            .entries
            .read()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        if entries.is_empty() {
            return Ok(text.to_string());
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result = Vec::new();
        let mut i = 0;

        while i < words.len() {
            let mut matched = false;

            // Try each substitution entry (longest first due to sorting)
            for (spoken_words, replacement) in entries.iter() {
                let len = spoken_words.len();
                if i + len > words.len() {
                    continue;
                }

                // Case-insensitive comparison
                let is_match = words[i..i + len]
                    .iter()
                    .zip(spoken_words.iter())
                    .all(|(input, spoken)| input.to_lowercase() == *spoken);

                if is_match {
                    result.push(replacement.clone());
                    i += len;
                    matched = true;
                    break;
                }
            }

            if !matched {
                result.push(words[i].to_string());
                i += 1;
            }
        }

        Ok(result.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_processor(entries: Vec<(&str, &str)>) -> WordSubstitutionProcessor {
        let entries: Vec<(Vec<String>, String)> = entries
            .into_iter()
            .map(|(spoken, replacement)| {
                let words: Vec<String> = spoken
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
                    .collect();
                (words, replacement.to_string())
            })
            .collect();
        // Sort by descending word count
        let mut entries = entries;
        entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        WordSubstitutionProcessor {
            entries: Arc::new(RwLock::new(entries)),
            user_dict: None,
        }
    }

    #[test]
    fn test_empty_string() {
        let processor = make_processor(vec![("shay moy", "chezmoi")]);
        let result = processor.process("").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_single_word_substitution() {
        let processor = make_processor(vec![("kube", "kubectl")]);
        let result = processor.process("run kube apply").unwrap();
        assert_eq!(result, "run kubectl apply");
    }

    #[test]
    fn test_multi_word_substitution() {
        let processor = make_processor(vec![("shay moy", "chezmoi")]);
        let result = processor.process("open shay moy config").unwrap();
        assert_eq!(result, "open chezmoi config");
    }

    #[test]
    fn test_case_insensitivity() {
        let processor = make_processor(vec![("shay moy", "chezmoi")]);
        let result = processor.process("open Shay Moy config").unwrap();
        assert_eq!(result, "open chezmoi config");
    }

    #[test]
    fn test_no_false_positives_on_partial_match() {
        let processor = make_processor(vec![("shay moy", "chezmoi")]);
        // "shay" alone should NOT trigger the substitution
        let result = processor.process("shay is a name").unwrap();
        assert_eq!(result, "shay is a name");
    }

    #[test]
    fn test_multiple_substitutions_in_one_sentence() {
        let processor = make_processor(vec![
            ("shay moy", "chezmoi"),
            ("cube cuttle", "kubectl"),
        ]);
        let result = processor.process("use shay moy and cube cuttle together").unwrap();
        assert_eq!(result, "use chezmoi and kubectl together");
    }

    #[test]
    fn test_no_substitutions_passthrough() {
        let processor = WordSubstitutionProcessor::empty();
        let result = processor.process("hello world").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_longest_match_first() {
        let processor = make_processor(vec![
            ("ch ez moy", "chezmoi"),
            ("ch ez", "chez"),
        ]);
        // "ch ez moy" should match first (3 words) over "ch ez" (2 words)
        let result = processor.process("run ch ez moy now").unwrap();
        assert_eq!(result, "run chezmoi now");
    }

    #[test]
    fn test_file_parsing() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# Pronunciation corrections").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "shay moy -> chezmoi").unwrap();
        writeln!(file, "ch ez moy -> chezmoi").unwrap();
        writeln!(file, "cube cuttle -> kubectl").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "# Another comment").unwrap();
        writeln!(file, "malformed line without arrow").unwrap();
        writeln!(file, " -> empty spoken").unwrap();
        writeln!(file, "empty replacement ->").unwrap();
        file.flush().unwrap();

        let entries =
            WordSubstitutionProcessor::load_substitutions(file.path()).unwrap();

        // Should have 3 valid entries (malformed/empty lines skipped)
        assert_eq!(entries.len(), 3);

        // Should be sorted by descending word count
        assert_eq!(entries[0].0.len(), 3); // "ch ez moy"
        assert_eq!(entries[1].0.len(), 2); // "shay moy" or "cube cuttle"
        assert_eq!(entries[2].0.len(), 2);

        // Verify content
        assert_eq!(entries[0].1, "chezmoi");
    }

    #[test]
    fn test_file_parsing_nonexistent() {
        let entries = WordSubstitutionProcessor::load_substitutions(
            Path::new("/tmp/nonexistent_substitutions_test.txt"),
        )
        .unwrap();
        assert!(entries.is_empty());
    }
}
