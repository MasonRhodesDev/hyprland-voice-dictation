//! Persistent correction store with frequency counting and auto-promotion.
//!
//! Corrections are stored in `~/.local/share/voice-dictation/corrections.json`.
//! When a correction reaches the auto-promote threshold, it's appended to
//! `substitutions.txt` in the existing `spoken -> replacement` format.

use crate::types::{BlockedPair, CorrectionPair, CorrectionRecord, CorrectionStats, MonitorConfig};
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
    /// Rejected pairs — never re-recorded or promoted again.
    #[serde(default)]
    blocklist: Vec<BlockedPair>,
}

impl Default for StoreFile {
    fn default() -> Self {
        Self { version: 1, corrections: Vec::new(), blocklist: Vec::new() }
    }
}

/// Persistent correction store with frequency counting and auto-promotion.
pub struct CorrectionStore {
    records: Vec<CorrectionRecord>,
    blocklist: Vec<BlockedPair>,
    config: MonitorConfig,
}

impl CorrectionStore {
    /// Load the correction store from disk. Creates an empty store if the file doesn't exist.
    ///
    /// Unpromoted records whose `last_seen` is older than `config.max_age_days`
    /// are pruned (aged out) on load.
    pub fn load(config: &MonitorConfig) -> Result<Self> {
        let store_file = if config.store_path.exists() {
            let content = fs::read_to_string(&config.store_path)?;
            serde_json::from_str::<StoreFile>(&content).unwrap_or_else(|e| {
                warn!("Failed to parse corrections file: {} — starting fresh", e);
                StoreFile::default()
            })
        } else {
            StoreFile::default()
        };

        let mut store = Self {
            records: store_file.corrections,
            blocklist: store_file.blocklist,
            config: config.clone(),
        };

        // Age out stale unpromoted pairs
        let cutoff = Utc::now() - chrono::Duration::days(i64::from(config.max_age_days));
        let before = store.records.len();
        store.records.retain(|r| r.promoted || r.last_seen >= cutoff);
        let pruned = before - store.records.len();
        if pruned > 0 {
            info!(
                "Pruned {} unpromoted correction(s) older than {} days",
                pruned, config.max_age_days
            );
            store.save()?;
        }

        debug!(
            "Loaded {} correction records ({} blocklisted pairs)",
            store.records.len(),
            store.blocklist.len()
        );
        Ok(store)
    }

    /// Create an empty store with the given config (for testing).
    pub fn empty(config: MonitorConfig) -> Self {
        Self { records: Vec::new(), blocklist: Vec::new(), config }
    }

    /// Save the correction store to disk.
    pub fn save(&self) -> Result<()> {
        let store_file = StoreFile {
            version: 1,
            corrections: self.records.clone(),
            blocklist: self.blocklist.clone(),
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

    /// Re-read records and blocklist from disk, discarding in-memory state.
    ///
    /// The daemon holds a long-lived in-memory store, but the `corrections`
    /// CLI (`clear`, `remove`, `edit`) mutates corrections.json directly. Without
    /// this, the daemon's stale in-memory copy clobbers those edits on its next
    /// save. The daemon's file watcher calls this when the file changes on disk.
    pub fn reload(&mut self) -> Result<()> {
        let fresh = Self::load(&self.config)?;
        self.records = fresh.records;
        self.blocklist = fresh.blocklist;
        Ok(())
    }

    /// Record a correction pair. Increments count if seen before.
    /// Returns true if this triggered auto-promotion to substitutions.txt.
    ///
    /// Blocklisted pairs (previously removed by the user) are silently dropped.
    pub fn record_correction(&mut self, pair: CorrectionPair) -> Result<bool> {
        let now = Utc::now();

        // Normalize: lowercase comparison for matching
        let orig_lower = pair.original.to_lowercase();
        let corr_lower = pair.corrected.to_lowercase();

        if self.is_blocked(&orig_lower, &corr_lower) {
            debug!(
                "Correction '{}' → '{}' is blocklisted, ignoring",
                pair.original, pair.corrected
            );
            return Ok(false);
        }

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
                promote_to_substitution_file(&self.config.substitutions_path, &self.records[idx])?;
                self.save()?;
                return Ok(true);
            }

            self.save()?;
            return Ok(false);
        }

        // New correction
        info!("New correction recorded: '{}' → '{}'", pair.original, pair.corrected);

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
        self.records.iter().filter(|r| r.original.to_lowercase() == orig_lower).collect()
    }

    /// All correction records, in store order.
    pub fn records(&self) -> &[CorrectionRecord] {
        &self.records
    }

    /// All blocklisted (rejected) pairs.
    pub fn blocklist(&self) -> &[BlockedPair] {
        &self.blocklist
    }

    /// Check whether a pair is blocklisted (case-insensitive).
    fn is_blocked(&self, orig_lower: &str, corr_lower: &str) -> bool {
        self.blocklist.iter().any(|b| b.original == orig_lower && b.corrected == corr_lower)
    }

    /// Remove all corrections matching a given original phrase (case-insensitive).
    ///
    /// Removed pairs are added to the persisted blocklist so they are never
    /// re-recorded or promoted again. Returns the removed records.
    pub fn remove(&mut self, original: &str) -> Result<Vec<CorrectionRecord>> {
        let orig_lower = original.to_lowercase();
        let now = Utc::now();

        let (removed, kept): (Vec<CorrectionRecord>, Vec<CorrectionRecord>) =
            self.records.drain(..).partition(|r| r.original.to_lowercase() == orig_lower);
        self.records = kept;

        for record in &removed {
            let corr_lower = record.corrected.to_lowercase();
            if !self.is_blocked(&orig_lower, &corr_lower) {
                self.blocklist.push(BlockedPair {
                    original: orig_lower.clone(),
                    corrected: corr_lower,
                    blocked_at: now,
                });
            }
            info!(
                "Removed and blocklisted correction: '{}' → '{}'",
                record.original, record.corrected
            );
        }

        if !removed.is_empty() {
            self.save()?;
        }
        Ok(removed)
    }

    /// Remove all correction records (the blocklist is kept).
    ///
    /// Unlike `remove()`, cleared pairs are NOT blocklisted — clearing means
    /// "start fresh", so the same corrections can be learned again.
    pub fn clear(&mut self) -> Result<usize> {
        let count = self.records.len();
        self.records.clear();
        if count > 0 {
            info!("Cleared {} correction record(s)", count);
        }
        self.save()?;
        Ok(count)
    }

    /// Get aggregate statistics about the correction store.
    pub fn stats(&self) -> CorrectionStats {
        let total_corrections = self.records.len();
        let total_observations: u32 = self.records.iter().map(|r| r.count).sum();
        let promoted_count = self.records.iter().filter(|r| r.promoted).count();
        let pending_count = self.records.iter().filter(|r| !r.promoted && r.count > 0).count();

        CorrectionStats { total_corrections, total_observations, promoted_count, pending_count }
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
        let needle = format!("{} -> {}", record.original.to_lowercase(), record.corrected);
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.to_lowercase().starts_with(&needle.to_lowercase()) {
                debug!("Substitution already exists: {} -> {}", record.original, record.corrected);
                return Ok(());
            }
        }
    }

    // Append the new substitution
    let mut file = fs::OpenOptions::new().create(true).append(true).open(substitutions_path)?;

    writeln!(file)?; // Ensure we start on a new line
    writeln!(file, "# Auto-promoted correction (seen {} times)", record.count)?;
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
            max_age_days: 30,
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

        let promoted = store.record_correction(make_pair("cash", "cache")).unwrap();
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

        store.record_correction(make_pair("cash", "cache")).unwrap();
        store.record_correction(make_pair("cash", "cache")).unwrap();
        store.record_correction(make_pair("cash", "cache")).unwrap();

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
        assert!(!store.record_correction(make_pair("cash", "cache")).unwrap());
        assert!(!store.record_correction(make_pair("cash", "cache")).unwrap());

        // Third time — should trigger promotion
        let promoted = store.record_correction(make_pair("cash", "cache")).unwrap();
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

        store.record_correction(make_pair("shay moy", "chezmoi")).unwrap();

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
            store.record_correction(make_pair("cash", "cache")).unwrap();
            store.record_correction(make_pair("hear", "here")).unwrap();
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

        store.record_correction(make_pair("cash", "cache")).unwrap();
        store.record_correction(make_pair("cash", "stash")).unwrap();
        store.record_correction(make_pair("hear", "here")).unwrap();

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

        store.record_correction(make_pair("cash", "cache")).unwrap();
        store.record_correction(make_pair("cash", "cache")).unwrap();
        store.record_correction(make_pair("cash", "cache")).unwrap();
        store.record_correction(make_pair("hear", "here")).unwrap();

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
        store.record_correction(make_pair("cash", "cache")).unwrap();
        store.record_correction(make_pair("cash", "cache")).unwrap();
        let promoted = store.record_correction(make_pair("cash", "cache")).unwrap();
        assert!(promoted);

        // Record 2 more times — should NOT promote again
        let promoted = store.record_correction(make_pair("cash", "cache")).unwrap();
        assert!(!promoted);
        let promoted = store.record_correction(make_pair("cash", "cache")).unwrap();
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

        store.record_correction(make_pair("Cash", "cache")).unwrap();
        store.record_correction(make_pair("cash", "Cache")).unwrap();

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

    #[test]
    fn test_remove_blocklists_pair() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config.clone());

        store.record_correction(make_pair("cash", "cache")).unwrap();
        let removed = store.remove("cash").unwrap();
        assert_eq!(removed.len(), 1);
        assert!(store.lookup("cash").is_empty());
        assert_eq!(store.blocklist().len(), 1);

        // Blocklisted pair is never re-recorded (case-insensitive)...
        for _ in 0..5 {
            assert!(!store.record_correction(make_pair("Cash", "Cache")).unwrap());
        }
        assert!(store.lookup("cash").is_empty());

        // ...so it can never reach the promote threshold either
        let subs = load_substitutions_from_file(&config.substitutions_path).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_is_noop() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config);

        store.record_correction(make_pair("cash", "cache")).unwrap();
        let removed = store.remove("nonexistent").unwrap();
        assert!(removed.is_empty());
        assert!(store.blocklist().is_empty());
        assert_eq!(store.records().len(), 1);
    }

    #[test]
    fn test_blocklist_persists_across_reload() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());

        {
            let mut store = CorrectionStore::empty(config.clone());
            store.record_correction(make_pair("cash", "cache")).unwrap();
            store.remove("cash").unwrap();
        }

        let mut store = CorrectionStore::load(&config).unwrap();
        assert_eq!(store.blocklist().len(), 1);
        assert!(!store.record_correction(make_pair("cash", "cache")).unwrap());
        assert!(store.lookup("cash").is_empty());
    }

    #[test]
    fn test_reload_picks_up_external_clear() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());

        // Daemon's long-lived in-memory store learns (and persists) a pair.
        let mut daemon_store = CorrectionStore::empty(config.clone());
        daemon_store.record_correction(make_pair("cash", "cache")).unwrap();
        assert_eq!(daemon_store.records().len(), 1);

        // A separate process (the `corrections` CLI) clears the file on disk.
        {
            let mut cli_store = CorrectionStore::load(&config).unwrap();
            cli_store.clear().unwrap();
        }

        // Without reload the daemon would clobber the clear on its next save;
        // reload discards the stale in-memory copy and matches disk.
        daemon_store.reload().unwrap();
        assert!(daemon_store.records().is_empty());
    }

    #[test]
    fn test_clear_keeps_blocklist() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path());
        let mut store = CorrectionStore::empty(config.clone());

        store.record_correction(make_pair("cash", "cache")).unwrap();
        store.record_correction(make_pair("hear", "here")).unwrap();
        store.remove("cash").unwrap();

        let cleared = store.clear().unwrap();
        assert_eq!(cleared, 1); // only "hear" was left
        assert!(store.records().is_empty());
        assert_eq!(store.blocklist().len(), 1);

        // Cleared (non-blocklisted) pairs can be learned again
        store.record_correction(make_pair("hear", "here")).unwrap();
        assert_eq!(store.lookup("hear").len(), 1);

        // Blocklist survives clear + reload
        let store = CorrectionStore::load(&config).unwrap();
        assert_eq!(store.blocklist().len(), 1);
        assert_eq!(store.records().len(), 1);
    }

    #[test]
    fn test_aging_prunes_old_unpromoted_on_load() {
        let dir = TempDir::new().unwrap();
        let config = make_config_in(dir.path()); // max_age_days: 30
        let old = Utc::now() - chrono::Duration::days(40);
        let recent = Utc::now();

        let make_record = |original: &str, last_seen, promoted| CorrectionRecord {
            original: original.to_string(),
            corrected: "x".to_string(),
            count: 1,
            first_seen: old,
            last_seen,
            promoted,
        };

        // Note: no blocklist field — also exercises loading the old file format
        let file = serde_json::json!({
            "version": 1,
            "corrections": [
                make_record("stale-pending", old, false),
                make_record("old-promoted", old, true),
                make_record("fresh-pending", recent, false),
            ],
        });
        fs::write(&config.store_path, serde_json::to_string_pretty(&file).unwrap()).unwrap();

        let store = CorrectionStore::load(&config).unwrap();
        assert!(store.lookup("stale-pending").is_empty(), "stale unpromoted should be pruned");
        assert_eq!(store.lookup("old-promoted").len(), 1, "promoted records never age out");
        assert_eq!(store.lookup("fresh-pending").len(), 1, "recent records are kept");

        // Pruning was persisted
        let reloaded = CorrectionStore::load(&config).unwrap();
        assert_eq!(reloaded.records().len(), 2);
    }
}
