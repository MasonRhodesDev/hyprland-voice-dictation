use anyhow::Result;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Manages user-defined words for spell checking.
///
/// Combines application-specific words with system Hunspell personal dictionary.
/// Supports hot-reload via file watching.
pub struct UserDictionary {
    /// Application-specific words (read-write)
    app_words: Arc<RwLock<HashSet<String>>>,
    /// System Hunspell personal dictionary words (read-only, but reloadable)
    system_words: Arc<RwLock<HashSet<String>>>,
    /// Path to application word list
    app_words_path: PathBuf,
    /// Path to system Hunspell dictionary (if available)
    system_dict_path: Option<PathBuf>,
}

impl UserDictionary {
    /// Create new user dictionary.
    ///
    /// Loads from:
    /// 1. ~/.local/share/voice-dictation/user_words.txt (app-specific)
    /// 2. ~/.hunspell_LANG (system Hunspell personal dictionary)
    pub fn new() -> Result<Self> {
        let app_words_path = Self::get_app_words_path()?;
        let app_words = Self::load_app_words(&app_words_path)?;

        let system_dict_path = Self::get_hunspell_personal_dict_path();
        let system_words = if let Some(ref path) = system_dict_path {
            Self::load_system_words_from_path(path).unwrap_or_default()
        } else {
            HashSet::new()
        };

        Ok(Self {
            app_words: Arc::new(RwLock::new(app_words)),
            system_words: Arc::new(RwLock::new(system_words)),
            app_words_path,
            system_dict_path,
        })
    }

    /// Get paths to watch for changes.
    ///
    /// Returns vector of paths that should be monitored for dictionary updates.
    pub fn watch_paths(&self) -> Vec<PathBuf> {
        let mut paths = vec![self.app_words_path.clone()];
        if let Some(ref system_path) = self.system_dict_path {
            paths.push(system_path.clone());
        }
        paths
    }

    /// Check if word exists in user dictionaries.
    pub fn contains(&self, word: &str) -> bool {
        let word_lower = word.to_lowercase();

        // Check app words
        if let Ok(app_words) = self.app_words.read() {
            if app_words.contains(&word_lower) {
                return true;
            }
        }

        // Check system words
        if let Ok(system_words) = self.system_words.read() {
            if system_words.contains(&word_lower) {
                return true;
            }
        }

        false
    }

    /// Add word to application dictionary.
    pub fn add(&self, word: &str) -> Result<()> {
        let word_lower = word.trim().to_lowercase();

        if word_lower.is_empty() {
            return Ok(());
        }

        // Add to app words
        {
            let mut app_words = self
                .app_words
                .write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            app_words.insert(word_lower);
        }

        // Persist to disk
        self.save()
    }

    /// Remove word from application dictionary.
    pub fn remove(&self, word: &str) -> Result<()> {
        let word_lower = word.to_lowercase();

        {
            let mut app_words = self
                .app_words
                .write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            app_words.remove(&word_lower);
        }

        self.save()
    }

    /// Get all words from application dictionary.
    pub fn app_words(&self) -> Vec<String> {
        self.app_words
            .read()
            .map(|words| {
                let mut list: Vec<_> = words.iter().cloned().collect();
                list.sort();
                list
            })
            .unwrap_or_default()
    }

    /// Reload app-specific dictionary from disk.
    pub fn reload_app_words(&self) -> Result<()> {
        let words = Self::load_app_words(&self.app_words_path)?;
        let mut app_words = self
            .app_words
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        *app_words = words;
        Ok(())
    }

    /// Reload system Hunspell dictionary from disk.
    pub fn reload_system_words(&self) -> Result<()> {
        if let Some(ref path) = self.system_dict_path {
            let words = Self::load_system_words_from_path(path).unwrap_or_default();
            let mut system_words = self
                .system_words
                .write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *system_words = words;
        }
        Ok(())
    }

    /// Reload both dictionaries from disk.
    pub fn reload_all(&self) -> Result<()> {
        self.reload_app_words()?;
        self.reload_system_words()?;
        Ok(())
    }

    // Private methods

    fn get_app_words_path() -> Result<PathBuf> {
        let data_dir = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;

        let path = data_dir.join("voice-dictation").join("user_words.txt");

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        Ok(path)
    }

    fn load_app_words(path: &Path) -> Result<HashSet<String>> {
        if !path.exists() {
            return Ok(HashSet::new());
        }

        let content = fs::read_to_string(path)?;
        let words = content
            .lines()
            .map(|line| line.trim().to_lowercase())
            .filter(|line| !line.is_empty())
            .collect();

        Ok(words)
    }

    fn load_system_words_from_path(path: &Path) -> Result<HashSet<String>> {
        let content = fs::read_to_string(path)?;
        let words = content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with('*'))
            .map(|line| {
                // Remove affixation suffix (e.g., "word/flag" -> "word")
                line.split('/').next().unwrap_or(line).to_lowercase()
            })
            .collect();

        Ok(words)
    }

    fn get_hunspell_personal_dict_path() -> Option<PathBuf> {
        use std::env;

        let home = dirs::home_dir()?;

        // Try environment variables for locale
        let locale = env::var("DICTIONARY")
            .or_else(|_| env::var("LC_ALL"))
            .or_else(|_| env::var("LC_MESSAGES"))
            .or_else(|_| env::var("LANG"))
            .ok()
            .and_then(|s| s.split('.').next().map(String::from));

        if let Some(loc) = locale {
            let path = home.join(format!(".hunspell_{}", loc));
            if path.exists() {
                return Some(path);
            }
        }

        // Fallback to default
        let default_path = home.join(".hunspell_default");
        if default_path.exists() {
            Some(default_path)
        } else {
            None
        }
    }

    fn save(&self) -> Result<()> {
        let app_words = self
            .app_words
            .read()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        let mut words: Vec<_> = app_words.iter().cloned().collect();
        words.sort();

        let content = words.join("\n");
        fs::write(&self.app_words_path, content)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_new_creates_directory() {
        let dict = UserDictionary::new();
        assert!(dict.is_ok());
    }

    #[test]
    fn test_add_and_contains() {
        let dict = UserDictionary::new().unwrap();
        assert!(!dict.contains("testword"));

        dict.add("testword").unwrap();
        assert!(dict.contains("testword"));
        assert!(dict.contains("TestWord")); // Case insensitive
    }

    #[test]
    fn test_remove() {
        let dict = UserDictionary::new().unwrap();
        dict.add("testword").unwrap();
        assert!(dict.contains("testword"));

        dict.remove("testword").unwrap();
        assert!(!dict.contains("testword"));
    }

    #[test]
    fn test_app_words_sorted() {
        let dict = UserDictionary::new().unwrap();
        dict.add("zebra").unwrap();
        dict.add("apple").unwrap();
        dict.add("monkey").unwrap();

        let words = dict.app_words();
        assert_eq!(words, vec!["apple", "monkey", "zebra"]);
    }

    #[test]
    fn test_empty_word_ignored() {
        let dict = UserDictionary::new().unwrap();
        assert!(dict.add("").is_ok());
        assert!(dict.add("   ").is_ok());
    }
}
