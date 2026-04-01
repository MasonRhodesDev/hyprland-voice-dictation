//! Persistent correction store with frequency counting and auto-promotion.
//!
//! Corrections are stored in `~/.local/share/voice-dictation/corrections.json`.
//! When a correction reaches the auto-promote threshold, it's appended to
//! `substitutions.txt` in the existing `spoken -> replacement` format.

use crate::types::{CorrectionPair, CorrectionRecord, CorrectionStats, MonitorConfig};
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;
use tracing::{debug, info, warn};

/// Versioned store format for forward compatibility.
#[derive(Debug, Serialize, Deserialize)]
struct StoreFile {
    version: u32,
    corrections: Vec<CorrectionRecord>,
}

impl Default for StoreFile {
    fn default() -> Self {
        Self {
            version: 1,
            corrections: Vec::new(),
        }
    }
}

/// Persistent correction store with frequency counting and auto-promotion.
pub struct CorrectionStore {
    records: Vec<CorrectionRecord>,
    config: MonitorConfig,
}

impl CorrectionStore {
    /// Load the correction store from disk. Creates an empty store if the file doesn't exist.
    pub fn load(config: &MonitorConfig) -> Result<Self> {
        let records = if config.store_path.exists() {
            let content = fs::read_to_string(&config.store_path)?;
            let store_file: StoreFile = serde_json::from_str(&content).unwrap_or_else(|e| {
                warn!(
                    "Failed to parse corrections file: {} — starting fresh",
                    e
                );
                StoreFile::default()
            });
            store_file.corrections
        } else {
            Vec::new()
        };

        debug!("Loaded {} correction records", records.len());
        Ok(Self {
            records,
            config: config.clone(),
        })
    }

    /// Create an empty store with the given config (for testing).
    pub fn empty(config: MonitorConfig) -> Self {
        Self {
            records: Vec::new(),
            config,
        }
    }

    /// Save the correction store to disk.
    pub fn save(&self) -> Result<()> {
        let store_file = StoreFile {
            version: 1,
            corrections: self.records.clone(),
        };

        // Ensure parent directory exists
        if let Some(parent) = self.config.store_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&store_file)?;
        fs::write(&self.config.store_path, json)?;
        debug!("Saved {} correction records", self.records.len());
        Ok(())
    }

    /// Record a correction pair. Increments count if seen before.
    /// Returns true if this triggered auto-promotion to substitutions.txt.
    pub fn record_correction(&mut self, pair: CorrectionPair) -> Result<bool> {
        let now = Utc::now();

        // Normalize: lowercase comparison for matching
        let orig_lower = pair.original.to_lowercase();
        let corr_lower = pair.corrected.to_lowercase();

        // Look for existing record with same original → corrected mapping
        let existing_idx = self.records.iter().position(|r| {
            r.original.to_lowercase() == orig_lower && r.corrected.to_lowercase() == corr_lower
        });

        if let Some(idx) = existing_idx {
            self.records[idx].count += 1;
            self.records[idx].last_seen = now;

            info!(
                "Correction count updated: '{}' → '{}' (count: {})",
                self.records[idx].original, self.records[idx].corrected, self.records[idx].count
            );

            // Check if we should auto-promote
            if self.records[idx].count >= self.config.auto_promote_threshold
                && !self.records[idx].promoted
            {
                self.records[idx].promoted = true;
                promote_to_substitution_file(
                    &self.config.substitutions_path,
                    &self.records[idx],
                )?;
                self.save()?;
                return Ok(true);
            }

            self.save()?;
            return Ok(false);
        }

        // New correction
        info!(
            "New correction recorded: '{}' → '{}'",
            pair.original, pair.corrected
        );

        let mut record = CorrectionRecord {
            original: pair.original,
            corrected: pair.corrected,
            count: 1,
            first_seen: now,
            last_seen: now,
            promoted: false,
        };

        // Check immediate promotion (threshold of 1)
        if self.config.auto_promote_threshold <= 1 {
            record.promoted = true;
            promote_to_substitution_file(&self.config.substitutions_path, &record)?;
            self.records.push(record);
            self.save()?;
            return Ok(true);
        }

        self.records.push(record);
        self.save()?;
        Ok(false)
    }

    /// Look up all corrections matching a given original phrase.
    pub fn lookup(&self, original: &str) -> Vec<&CorrectionRecord> {
        let orig_lower = original.to_lowercase();
        self.records
            .iter()
            .filter(|r| r.original.to_lowercase() == orig_lower)
            .collect()
    }

    /// Get aggregate statistics about the correction store.
    pub fn stats(&self) -> CorrectionStats {
        let total_corrections = self.records.len();
        let total_observations: u32 = self.records.iter().map(|r| r.count).sum();
        let promoted_count = self.records.iter().filter(|r| r.promoted).count();
        let pending_count = self
            .records
            .iter()
            .filter(|r| !r.promoted && r.count > 0)
            .count();

        CorrectionStats {
            total_corrections,
            total_observations,
            promoted_count,
            pending_count,
        }
    }

}

/// Append a correction as a substitution rule to substitutions.txt.
///
/// Format: `original -> corrected` (same as WordSubstitutionProcessor expects).
/// The daemon's existing file watcher will detect the change and hot-reload.
fn promote_to_substitution_file(
    substitutions_path: &Path,
    record: &CorrectionRecord,
) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = substitutions_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Check if this substitution already exists in the file
    if substitutions_path.exists() {
        let content = fs::read_to_string(substitutions_path)?;
        let needle = format!(
            "{} -> {}",
            record.original.to_lowercase(),
            record.corrected
        );
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.to_lowercase().starts_with(&needle.to_lowercase()) {
                debug!(
                    "Substitution already exists: {} -> {}",
                    record.original, record.corrected
                );
                return Ok(());
            }
        }
    }

    // Append the new substitution
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(substitutions_path)?;

    writeln!(file)?; // Ensure we start on a new line
    writeln!(
        file,
        "# Auto-promoted correction (seen {} times)",
        record.count
    )?;
    writeln!(file, "{} -> {}", record.original, record.corrected)?;

    info!(
        "Auto-promoted correction to substitutions: '{}' → '{}' (after {} observations)",
        record.original, record.corrected, record.count
    );

    Ok(())
}

/// Load the substitutions file and parse entries (for compatibility testing).
pub fn load_substitutions_from_file(path: &Path) -> Result<Vec<(String, String)>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let content = fs::read_to_string(path)?;
    let entries: Vec<(String, String)> = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(2, "->").collect();
            if parts.len() == 2 {
                let spoken = parts[0].trim().to_string();
                let replacement = parts[1].trim().to_string();
                if !spoken.is_empty() && !replacement.is_empty() {
                    Some((spoken, replacement))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::TempDir;

    fn make_config_in(dir: &Path) -> MonitorConfig {
        MonitorConfig {
            enabled: true,
            monitor_duration_secs: 60,
            auto_promote_threshold: 3,
            store_path: dir.join("corrections.json"),
            substitutions_path: dir.join("substitutions.txt"),
        }
    }

    fn make_pair(original: &str, corrected: &str) -> CorrectionPair {
        CorrectionPair {
            original: original.to_string(),
            corrected: corrected.to_string(),
            context_before: String::new(),
            context_after: String::new(),
            window_class: "test-app".to_string(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_new_correction_recorded() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config);

        let promoted = store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        assert!(!promoted);

        let records = store.lookup("cash");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].count, 1);
        assert!(!records[0].promoted);
    }

    #[test]
    fn test_duplicate_increments_count() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config);

        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();

        let records = store.lookup("cash");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].count, 3);
    }

    #[test]
    fn test_auto_promotion_at_threshold() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config.clone());

        // Record 2 times — no promotion yet
        assert!(!store
            .record_correction(make_pair("cash", "cache"))
            .unwrap());
        assert!(!store
            .record_correction(make_pair("cash", "cache"))
            .unwrap());

        // Third time — should trigger promotion
        let promoted = store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        assert!(promoted);

        // Verify substitutions.txt was written
        let subs = load_substitutions_from_file(&config.substitutions_path).unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].0, "cash");
        assert_eq!(subs[0].1, "cache");
    }

    #[test]
    fn test_substitution_format_compatible() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config_in(dir.path());
        config.auto_promote_threshold = 1; // Promote immediately
        let mut store = CorrectionStore::empty(config.clone());

        store
            .record_correction(make_pair("shay moy", "chezmoi"))
            .unwrap();

        // Read raw file and verify format
        let content = fs::read_to_string(&config.substitutions_path).unwrap();
        assert!(
            content.contains("shay moy -> chezmoi"),
            "File should contain 'shay moy -> chezmoi', got: {}",
            content
        );

        // Verify parseable by our loader (equivalent to WordSubstitutionProcessor::load_substitutions)
        let subs = load_substitutions_from_file(&config.substitutions_path).unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].0, "shay moy");
        assert_eq!(subs[0].1, "chezmoi");
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());

        // Create store, add corrections, save
        {
            let mut store = CorrectionStore::empty(config.clone());
            store
                .record_correction(make_pair("cash", "cache"))
                .unwrap();
            store
                .record_correction(make_pair("hear", "here"))
                .unwrap();
        }

        // Load from disk and verify
        let store = CorrectionStore::load(&config).unwrap();
        assert_eq!(store.records.len(), 2);

        let cash_records = store.lookup("cash");
        assert_eq!(cash_records.len(), 1);
        assert_eq!(cash_records[0].corrected, "cache");

        let hear_records = store.lookup("hear");
        assert_eq!(hear_records.len(), 1);
        assert_eq!(hear_records[0].corrected, "here");
    }

    #[test]
    fn test_lookup_by_original() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config);

        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        store
            .record_correction(make_pair("cash", "stash"))
            .unwrap();
        store
            .record_correction(make_pair("hear", "here"))
            .unwrap();

        let cash = store.lookup("cash");
        assert_eq!(cash.len(), 2);

        let hear = store.lookup("hear");
        assert_eq!(hear.len(), 1);

        let none = store.lookup("nonexistent");
        assert!(none.is_empty());
    }

    #[test]
    fn test_stats() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config);

        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        store
            .record_correction(make_pair("hear", "here"))
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_corrections, 2); // "cash→cache" and "hear→here"
        assert_eq!(stats.total_observations, 4); // 3 + 1
        assert_eq!(stats.promoted_count, 1); // "cash→cache" promoted at threshold 3
        assert_eq!(stats.pending_count, 1); // "hear→here" pending
    }

    #[test]
    fn test_no_duplicate_promotions() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config.clone());

        // Record 3 times to trigger promotion
        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        let promoted = store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        assert!(promoted);

        // Record 2 more times — should NOT promote again
        let promoted = store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        assert!(!promoted);
        let promoted = store
            .record_correction(make_pair("cash", "cache"))
            .unwrap();
        assert!(!promoted);

        // substitutions.txt should have exactly one entry
        let subs = load_substitutions_from_file(&config.substitutions_path).unwrap();
        assert_eq!(subs.len(), 1);
    }

    #[test]
    fn test_case_insensitive_matching() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config);

        store
            .record_correction(make_pair("Cash", "cache"))
            .unwrap();
        store
            .record_correction(make_pair("cash", "Cache"))
            .unwrap();

        // Should match as same correction (case-insensitive)
        let records = store.lookup("cash");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].count, 2);
    }

    #[test]
    fn test_empty_store_load_nonexistent() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let store = CorrectionStore::load(&config).unwrap();
        assert_eq!(store.records.len(), 0);
        assert_eq!(store.stats().total_corrections, 0);
    }
}
