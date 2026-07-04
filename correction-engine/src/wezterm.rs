//! wezterm-native correction detection backend.
//!
//! AT-SPI2 cannot see wezterm at all (wezterm has no accessibility
//! implementation upstream), but wezterm's mux CLI gives full read access to
//! pane contents: `wezterm cli list --format json` for pane metadata and
//! `wezterm cli get-text --pane-id N` for the rendered text (no ANSI escapes
//! by default). This module snapshots the focused pane right after text
//! injection, then polls the pane text for the monitoring window and diffs
//! revisions of the injected span to extract [`CorrectionPair`] records into
//! the shared [`CorrectionStore`].
//!
//! # How it works
//!
//! 1. **Socket discovery**: as a systemd user service the daemon does not
//!    inherit `WEZTERM_UNIX_SOCKET`, so we scan
//!    `$XDG_RUNTIME_DIR/wezterm/gui-sock-*`, probe each socket with a real
//!    `connect()` (stale sockets refuse), and pick the most recently modified
//!    live one. All CLI invocations pass `--no-auto-start` so a stale or
//!    missing socket can never cause us to spawn a `wezterm-mux-server`.
//! 2. **Span location**: the injected text is tokenized on whitespace and the
//!    token sequence is searched in the pane text (last occurrence wins, since
//!    the freshest prompt is at the bottom). Terminal reflow can hard-wrap a
//!    word across a line boundary, so the matcher tolerates one injected token
//!    being split across multiple consecutive pane tokens.
//! 3. **Revision diffing**: every ~2s the pane text is re-fetched and a
//!    token-level LCS diff against the baseline snapshot is computed. Change
//!    regions overlapping the injected span are converted into synthetic
//!    insert/delete [`TextChangeEvent`]s and fed through the existing
//!    [`crate::diff::extract_corrections`] machinery, so pair extraction,
//!    append/prepend skipping and word-level insert handling behave exactly
//!    like the AT-SPI backend. Recording goes through
//!    [`CorrectionStore::record_correction`], respecting the blocklist and
//!    aging rules.
//!
//! # Honest limitations
//!
//! This is a heuristic, scrollback-based backend — it observes rendered cells,
//! not an editing buffer:
//!
//! - **Multiline prompts and TUIs** (nvim inside wezterm, fzf, etc.) redraw
//!   the whole screen and reflow arbitrarily. We detect this as diff churn
//!   (most of the pane changed) or as failure to locate the injected span, and
//!   bail out with a log line instead of recording garbage pairs.
//! - **Readline editing** shows up as a full-line rewrite between two polls,
//!   not as discrete insert/delete events. The token-level LCS isolates the
//!   changed words, so word-level pairs are still extracted where possible,
//!   but character-precise edit positions are lost.
//! - **Hard wrapping** is only heuristically undone: a token split across a
//!   line boundary is re-joined during matching, but the baseline span text
//!   (used for correction context) keeps the pane's rendition.
//! - **Scroll-out**: once the injected text scrolls out of the viewport the
//!   whole span appears deleted. This is detected (`SpanLost`) and monitoring
//!   ends with whatever the last good poll saw, rather than recording a bogus
//!   "everything was deleted" pair.
//! - **Focused-pane inference**: `wezterm cli list` marks the active pane per
//!   tab, not the globally focused one, so the injected text itself is used to
//!   confirm which candidate pane received the injection.

use crate::store::CorrectionStore;
use crate::types::{
    CorrectionPair, InjectionContext, MonitorConfig, TextChangeEvent, TextChangeOp,
};
use anyhow::{Context as AnyhowContext, Result};
use async_trait::async_trait;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// Hyprland window class of wezterm — used by the daemon to route injections
/// to this backend instead of the AT-SPI monitor.
pub const WEZTERM_WINDOW_CLASS: &str = "org.wezfurlong.wezterm";

/// How often the pane text is re-fetched during the monitoring window.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Timeout for a single `wezterm cli` invocation.
const CLI_TIMEOUT: Duration = Duration::from_secs(5);

/// Consecutive failed/churny polls before we give up on the window.
const MAX_CONSECUTIVE_BAD_POLLS: u32 = 3;

/// If more than this fraction of the baseline tokens changed in one poll,
/// treat it as a full-screen redraw (TUI) rather than an edit.
const CHURN_FRACTION: f64 = 0.5;

/// Churn detection only kicks in above this many baseline tokens, so tiny
/// panes with a single prompt line don't trip it.
const CHURN_MIN_TOKENS: usize = 20;

/// Diff inputs larger than this (tokens, after prefix/suffix trimming) are
/// treated as churn instead of running a quadratic LCS.
const MAX_DIFF_TOKENS: usize = 3000;

// ---------------------------------------------------------------------------
// Socket discovery
// ---------------------------------------------------------------------------

/// Find a live wezterm GUI mux socket.
///
/// Preference order:
/// 1. `WEZTERM_UNIX_SOCKET` if set and live (covers running inside wezterm).
/// 2. The most recently modified live `gui-sock-*` under
///    `$XDG_RUNTIME_DIR/wezterm/` (covers the systemd-user-service case where
///    the env var is not inherited). Stale sockets from crashed instances are
///    skipped because they refuse connections.
pub fn discover_socket() -> Option<PathBuf> {
    if let Ok(env_sock) = std::env::var("WEZTERM_UNIX_SOCKET") {
        let path = PathBuf::from(env_sock);
        if socket_is_live(&path) {
            return Some(path);
        }
        debug!("WEZTERM_UNIX_SOCKET set but socket is not live: {}", path.display());
    }

    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let sock_dir = Path::new(&runtime_dir).join("wezterm");

    let mut live: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&sock_dir).ok()?.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("gui-sock-") {
            continue;
        }
        if !socket_is_live(&path) {
            debug!("Skipping stale wezterm socket: {}", path.display());
            continue;
        }
        let mtime = entry.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
        live.push((mtime, path));
    }

    live.sort();
    let chosen = live.pop().map(|(_, p)| p);
    if let Some(ref p) = chosen {
        debug!("Discovered live wezterm socket: {}", p.display());
    }
    chosen
}

/// A socket is live iff something is accepting connections on it.
/// Unix-domain connects resolve immediately (success or ECONNREFUSED),
/// so a blocking probe is fine.
fn socket_is_live(path: &Path) -> bool {
    std::os::unix::net::UnixStream::connect(path).is_ok()
}

// ---------------------------------------------------------------------------
// wezterm CLI client
// ---------------------------------------------------------------------------

/// Subset of `wezterm cli list --format json` we care about.
#[derive(Debug, Clone, Deserialize)]
pub struct PaneEntry {
    pub pane_id: u64,
    #[serde(default)]
    pub is_active: bool,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub workspace: String,
}

/// Abstraction over the wezterm mux so the monitor is testable without a live
/// wezterm.
#[async_trait]
pub trait WeztermClient: Send + Sync {
    async fn list_panes(&self) -> Result<Vec<PaneEntry>>;
    async fn get_text(&self, pane_id: u64) -> Result<String>;
}

/// Production client that shells out to `wezterm cli --no-auto-start` with an
/// explicitly discovered socket.
pub struct WeztermCli {
    socket: PathBuf,
}

impl WeztermCli {
    /// Discover a live socket; `None` if no wezterm GUI instance is running.
    pub fn discover() -> Option<Self> {
        discover_socket().map(|socket| Self { socket })
    }

    /// Use an explicit socket path (must already be validated by the caller).
    pub fn with_socket(socket: PathBuf) -> Self {
        Self { socket }
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    async fn run(&self, args: &[&str]) -> Result<String> {
        let mut cmd = tokio::process::Command::new("wezterm");
        cmd.arg("cli")
            .arg("--no-auto-start")
            .args(args)
            .env("WEZTERM_UNIX_SOCKET", &self.socket)
            .stdin(std::process::Stdio::null());

        let output = tokio::time::timeout(CLI_TIMEOUT, cmd.output())
            .await
            .context("wezterm cli timed out")?
            .context("failed to spawn wezterm cli")?;

        if !output.status.success() {
            anyhow::bail!(
                "wezterm cli {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

#[async_trait]
impl WeztermClient for WeztermCli {
    async fn list_panes(&self) -> Result<Vec<PaneEntry>> {
        let stdout = self.run(&["list", "--format", "json"]).await?;
        serde_json::from_str(&stdout).context("failed to parse wezterm cli list output")
    }

    async fn get_text(&self, pane_id: u64) -> Result<String> {
        self.run(&["get-text", "--pane-id", &pane_id.to_string()]).await
    }
}

// ---------------------------------------------------------------------------
// Pure text machinery: normalization, span location, revision diffing
// ---------------------------------------------------------------------------

/// A token range (`[start, end)`) within a tokenized pane snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Whitespace-normalize into tokens. This is what makes matching survive
/// terminal soft-wrapping and the trailing-space padding `get-text` emits.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

/// Try to match all of `needle` starting at `haystack[pos]`, tolerating a
/// needle token being hard-wrapped across consecutive haystack tokens.
/// Returns the exclusive haystack end index on success.
fn match_needle_at(haystack: &[String], pos: usize, needle: &[String]) -> Option<usize> {
    let mut h = pos;
    for n in needle {
        if h >= haystack.len() {
            return None;
        }
        if &haystack[h] == n {
            h += 1;
            continue;
        }
        // Hard-wrap tolerance: the needle token may have been split across
        // line boundaries into several consecutive pane tokens.
        let mut acc = String::new();
        let mut j = h;
        loop {
            if j >= haystack.len() {
                return None;
            }
            acc.push_str(&haystack[j]);
            j += 1;
            if &acc == n {
                break;
            }
            if !n.starts_with(acc.as_str()) {
                return None;
            }
        }
        h = j;
    }
    Some(h)
}

/// Locate the *last* occurrence of the injected token sequence in the pane.
/// Last occurrence because the injection lands at the freshest prompt (bottom
/// of the pane), while shell echo / scrollback may contain older copies.
pub fn locate_span(haystack: &[String], needle: &[String]) -> Option<Span> {
    if needle.is_empty() || haystack.is_empty() {
        return None;
    }
    for start in (0..haystack.len()).rev() {
        if let Some(end) = match_needle_at(haystack, start, needle) {
            return Some(Span { start, end });
        }
    }
    None
}

/// A contiguous change between two token sequences:
/// `a[a_start..a_end]` was replaced by `b[b_start..b_end]`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DiffRegion {
    a_start: usize,
    a_end: usize,
    b_start: usize,
    b_end: usize,
}

/// Token-level diff via common prefix/suffix trimming + LCS on the middle.
/// Returns `None` if the middle is too large to diff (treated as churn).
fn token_diff(a: &[String], b: &[String]) -> Option<Vec<DiffRegion>> {
    // Trim common prefix
    let mut prefix = 0;
    while prefix < a.len() && prefix < b.len() && a[prefix] == b[prefix] {
        prefix += 1;
    }
    // Trim common suffix
    let mut suffix = 0;
    while suffix < a.len() - prefix
        && suffix < b.len() - prefix
        && a[a.len() - 1 - suffix] == b[b.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let am = &a[prefix..a.len() - suffix];
    let bm = &b[prefix..b.len() - suffix];

    if am.is_empty() && bm.is_empty() {
        return Some(Vec::new());
    }
    if am.len() > MAX_DIFF_TOKENS || bm.len() > MAX_DIFF_TOKENS {
        return None;
    }

    // LCS DP over the middle
    let n = am.len();
    let m = bm.len();
    let mut dp = vec![0u32; (n + 1) * (m + 1)];
    let idx = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[idx(i, j)] = if am[i] == bm[j] {
                dp[idx(i + 1, j + 1)] + 1
            } else {
                dp[idx(i + 1, j)].max(dp[idx(i, j + 1)])
            };
        }
    }

    // Backtrack, emitting maximal unmatched regions
    let mut regions = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    let (mut ra, mut rb) = (0usize, 0usize); // region start candidates
    let mut in_region = false;
    while i < n || j < m {
        if i < n && j < m && am[i] == bm[j] {
            if in_region {
                regions.push(DiffRegion {
                    a_start: prefix + ra,
                    a_end: prefix + i,
                    b_start: prefix + rb,
                    b_end: prefix + j,
                });
                in_region = false;
            }
            i += 1;
            j += 1;
        } else {
            if !in_region {
                ra = i;
                rb = j;
                in_region = true;
            }
            if j >= m || (i < n && dp[idx(i + 1, j)] >= dp[idx(i, j + 1)]) {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
    if in_region {
        regions.push(DiffRegion {
            a_start: prefix + ra,
            a_end: prefix + i,
            b_start: prefix + rb,
            b_end: prefix + j,
        });
    }
    Some(regions)
}

/// Outcome of diffing one polled revision against the baseline snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevisionOutcome {
    /// Correction pairs extracted (possibly empty: nothing relevant changed).
    Corrections(Vec<CorrectionPair>),
    /// The injected span disappeared entirely (scrolled out, cleared, Ctrl-C).
    SpanLost,
    /// Most of the pane changed at once — a TUI redraw, not an edit.
    Churn,
}

/// Char offset of `token_idx` (relative to `span.start`) within the
/// space-joined span text.
fn span_char_offset(baseline: &[String], span: Span, token_idx: usize) -> i32 {
    baseline[span.start..token_idx].iter().map(|t| t.chars().count() as i32 + 1).sum()
}

/// Diff one revision of the pane against the baseline and extract correction
/// pairs for changes that fall inside the injected span.
///
/// Reuses [`crate::diff::extract_corrections`]: change regions are converted
/// into synthetic delete/insert [`TextChangeEvent`]s positioned within the
/// normalized span text, so pairing, boundary-append skipping and word-level
/// insert handling are identical to the AT-SPI backend.
pub fn diff_revision(
    baseline: &[String],
    span: Span,
    current: &[String],
    context: &InjectionContext,
) -> RevisionOutcome {
    let Some(regions) = token_diff(baseline, current) else {
        return RevisionOutcome::Churn;
    };

    // Churn guard: a TUI taking over the pane rewrites nearly everything.
    let changed_baseline: usize = regions.iter().map(|r| r.a_end - r.a_start).sum();
    if baseline.len() >= CHURN_MIN_TOKENS
        && (changed_baseline as f64) / (baseline.len() as f64) > CHURN_FRACTION
    {
        return RevisionOutcome::Churn;
    }

    // Span-lost guard: every span token deleted and nothing inserted in range.
    let mut span_tokens_deleted = 0usize;
    let mut span_tokens_inserted = 0usize;
    for r in &regions {
        let ca = r.a_start.max(span.start);
        let cb = r.a_end.min(span.end);
        if ca < cb {
            span_tokens_deleted += cb - ca;
            span_tokens_inserted += r.b_end - r.b_start;
        }
    }
    if !span.is_empty() && span_tokens_deleted == span.len() && span_tokens_inserted == 0 {
        return RevisionOutcome::SpanLost;
    }

    // Convert overlapping regions into synthetic events against the
    // normalized span text.
    let span_text: String = baseline[span.start..span.end].join(" ");
    let now = Instant::now();
    let mut events: Vec<TextChangeEvent> = Vec::new();

    for r in &regions {
        // Overlap test uses a closed range so pure inserts at the span
        // boundaries are still forwarded — extract_corrections itself skips
        // pure appends/prepends at the exact edges.
        if r.a_end < span.start || r.a_start > span.end {
            continue;
        }
        let ca = r.a_start.max(span.start);
        let cb = r.a_end.min(span.end);
        let deleted: String = baseline[ca..cb].join(" ");
        let inserted: String = current[r.b_start..r.b_end].join(" ");
        let pos = span_char_offset(baseline, span, ca);

        if !deleted.is_empty() {
            events.push(TextChangeEvent {
                operation: TextChangeOp::Delete,
                start_pos: pos,
                length: deleted.chars().count() as i32,
                text: deleted,
                timestamp: now,
                source_app: "wezterm".to_string(),
            });
        }
        if !inserted.is_empty() {
            events.push(TextChangeEvent {
                operation: TextChangeOp::Insert,
                start_pos: pos,
                length: inserted.chars().count() as i32,
                text: inserted,
                timestamp: now,
                source_app: "wezterm".to_string(),
            });
        }
    }

    if events.is_empty() {
        return RevisionOutcome::Corrections(Vec::new());
    }

    // Positions are relative to the normalized span text, so extraction must
    // run against that text rather than the raw injected string.
    let span_context = InjectionContext { text: span_text, ..context.clone() };
    let pairs = crate::diff::extract_corrections(&span_context, &events);
    RevisionOutcome::Corrections(pairs)
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

/// wezterm-native counterpart of [`crate::CorrectionMonitor`].
///
/// Same conceptual lifecycle: [`WeztermMonitor::start_monitoring`] spawns a
/// detached task that watches the injected text for `monitor_duration_secs`
/// and records detected [`CorrectionPair`]s into the [`CorrectionStore`].
pub struct WeztermMonitor {
    /// `None` means "discover a live socket at monitoring time" — this keeps
    /// daemon startup independent of wezterm's lifetime.
    client: Option<Arc<dyn WeztermClient>>,
    store: Arc<Mutex<CorrectionStore>>,
    config: MonitorConfig,
    poll_interval: Duration,
}

impl WeztermMonitor {
    /// Production constructor with its own store instance.
    pub fn new(config: MonitorConfig) -> Result<Self> {
        let store = CorrectionStore::load(&config)?;
        Ok(Self {
            client: None,
            store: Arc::new(Mutex::new(store)),
            config,
            poll_interval: DEFAULT_POLL_INTERVAL,
        })
    }

    /// Production constructor sharing a store with another monitor (avoids
    /// two in-memory copies of corrections.json clobbering each other).
    pub fn with_shared_store(store: Arc<Mutex<CorrectionStore>>, config: MonitorConfig) -> Self {
        Self { client: None, store, config, poll_interval: DEFAULT_POLL_INTERVAL }
    }

    /// Test constructor with an injected client and store.
    pub fn with_client(
        client: Arc<dyn WeztermClient>,
        store: CorrectionStore,
        config: MonitorConfig,
    ) -> Self {
        Self {
            client: Some(client),
            store: Arc::new(Mutex::new(store)),
            config,
            poll_interval: DEFAULT_POLL_INTERVAL,
        }
    }

    pub fn set_poll_interval(&mut self, interval: Duration) {
        self.poll_interval = interval;
    }

    /// Whether a live wezterm GUI socket is reachable right now.
    pub fn is_available() -> bool {
        discover_socket().is_some()
    }

    /// Start monitoring the injected text for corrections. Mirrors
    /// [`crate::CorrectionMonitor::start_monitoring`].
    pub fn start_monitoring(
        &self,
        context: InjectionContext,
    ) -> tokio::task::JoinHandle<Vec<CorrectionPair>> {
        let client = self.client.clone();
        let store = Arc::clone(&self.store);
        let config = self.config.clone();
        let poll_interval = self.poll_interval;

        tokio::spawn(async move {
            match Self::run_monitoring(client, store, config, poll_interval, context).await {
                Ok(pairs) => pairs,
                Err(e) => {
                    error!("wezterm correction monitoring failed: {}", e);
                    Vec::new()
                }
            }
        })
    }

    async fn run_monitoring(
        client: Option<Arc<dyn WeztermClient>>,
        store: Arc<Mutex<CorrectionStore>>,
        config: MonitorConfig,
        poll_interval: Duration,
        context: InjectionContext,
    ) -> Result<Vec<CorrectionPair>> {
        let injected_tokens = tokenize(&context.text);
        if injected_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let client: Arc<dyn WeztermClient> = match client {
            Some(c) => c,
            None => Arc::new(
                WeztermCli::discover()
                    .context("no live wezterm socket found — is wezterm running?")?,
            ),
        };

        // Let the compositor/pane render the freshly typed text.
        tokio::time::sleep(Duration::from_millis(300)).await;

        let Some((pane_id, baseline, span)) =
            Self::locate_injection(client.as_ref(), &injected_tokens).await?
        else {
            warn!(
                "wezterm: could not locate injected text in any pane \
                 (TUI / multiline prompt / heavy reflow?) — skipping correction monitoring"
            );
            return Ok(Vec::new());
        };

        info!(
            "wezterm correction monitoring started for {}s (pane {}, span {} tokens)",
            config.monitor_duration_secs,
            pane_id,
            span.len()
        );

        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(config.monitor_duration_secs);
        let mut last_pairs: Vec<CorrectionPair> = Vec::new();
        let mut consecutive_bad = 0u32;

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            tokio::time::sleep(remaining.min(poll_interval)).await;

            let text = match client.get_text(pane_id).await {
                Ok(t) => t,
                Err(e) => {
                    debug!("wezterm get-text failed during poll: {}", e);
                    consecutive_bad += 1;
                    if consecutive_bad >= MAX_CONSECUTIVE_BAD_POLLS {
                        warn!("wezterm: pane unreadable, ending monitoring early");
                        break;
                    }
                    continue;
                }
            };

            match diff_revision(&baseline, span, &tokenize(&text), &context) {
                RevisionOutcome::Corrections(pairs) => {
                    consecutive_bad = 0;
                    last_pairs = pairs;
                }
                RevisionOutcome::SpanLost => {
                    // Scrolled out / cleared — later polls can't recover it.
                    debug!("wezterm: injected span no longer visible, ending monitoring");
                    break;
                }
                RevisionOutcome::Churn => {
                    consecutive_bad += 1;
                    debug!("wezterm: pane churn detected ({}), likely TUI redraw", consecutive_bad);
                    if consecutive_bad >= MAX_CONSECUTIVE_BAD_POLLS {
                        warn!(
                            "wezterm: persistent pane churn (TUI took over?) — \
                             ending monitoring without recording"
                        );
                        // Whatever we extracted before the TUI appeared may be
                        // stale; keep it only if a later poll re-confirmed it,
                        // which it didn't — so drop.
                        last_pairs.clear();
                        break;
                    }
                }
            }
        }

        info!("wezterm correction monitoring ended. {} correction(s) detected.", last_pairs.len());

        if last_pairs.is_empty() {
            return Ok(Vec::new());
        }

        let mut store = store.lock().await;
        for pair in &last_pairs {
            match store.record_correction(pair.clone()) {
                Ok(true) => info!(
                    "Correction auto-promoted to substitution: '{}' → '{}'",
                    pair.original, pair.corrected
                ),
                Ok(false) => {}
                Err(e) => warn!("Failed to record correction: {}", e),
            }
        }

        Ok(last_pairs)
    }

    /// Find which pane received the injection: prefer active panes, but
    /// confirm by actually locating the injected token sequence in the text.
    async fn locate_injection(
        client: &dyn WeztermClient,
        injected_tokens: &[String],
    ) -> Result<Option<(u64, Vec<String>, Span)>> {
        for attempt in 0..3 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(700)).await;
            }

            let panes = client.list_panes().await?;
            let mut candidates: Vec<&PaneEntry> = panes.iter().filter(|p| p.is_active).collect();
            if candidates.is_empty() {
                candidates = panes.iter().collect();
            }

            let mut found: Option<(u64, Vec<String>, Span)> = None;
            for pane in candidates {
                let text = match client.get_text(pane.pane_id).await {
                    Ok(t) => t,
                    Err(e) => {
                        debug!("wezterm get-text failed for pane {}: {}", pane.pane_id, e);
                        continue;
                    }
                };
                let tokens = tokenize(&text);
                if let Some(span) = locate_span(&tokens, injected_tokens) {
                    if let Some((first_pane, _, _)) = &found {
                        warn!(
                            "wezterm: injected text matched multiple panes; \
                             using the first match (pane {})",
                            first_pane
                        );
                        break;
                    }
                    found = Some((pane.pane_id, tokens, span));
                }
            }
            if found.is_some() {
                return Ok(found);
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;

    fn toks(s: &str) -> Vec<String> {
        tokenize(s)
    }

    fn make_context(text: &str) -> InjectionContext {
        InjectionContext {
            text: text.to_string(),
            timestamp: Utc::now(),
            instant: Instant::now(),
            window_class: WEZTERM_WINDOW_CLASS.to_string(),
            window_title: "wezterm".to_string(),
        }
    }

    fn test_config(dir: &Path) -> MonitorConfig {
        MonitorConfig {
            enabled: true,
            monitor_duration_secs: 1,
            auto_promote_threshold: 3,
            max_age_days: 30,
            store_path: dir.join("corrections.json"),
            substitutions_path: dir.join("substitutions.txt"),
        }
    }

    // --- tokenize / normalization ---

    #[test]
    fn test_tokenize_collapses_whitespace_and_padding() {
        // get-text pads lines with trailing spaces and terminals soft-wrap
        let text = "hello   world  \n   this is\t a prompt   \n";
        assert_eq!(toks(text), vec!["hello", "world", "this", "is", "a", "prompt"]);
    }

    #[test]
    fn test_tokenize_empty() {
        assert!(toks("   \n  \t ").is_empty());
    }

    // --- span location ---

    #[test]
    fn test_locate_span_simple() {
        let hay = toks("prompt $ echo hello world and more");
        let needle = toks("hello world");
        let span = locate_span(&hay, &needle).unwrap();
        assert_eq!((span.start, span.end), (3, 5));
    }

    #[test]
    fn test_locate_span_prefers_last_occurrence() {
        // Older copy in scrollback + fresh copy at the prompt
        let hay = toks("hello world output line $ hello world");
        let needle = toks("hello world");
        let span = locate_span(&hay, &needle).unwrap();
        assert_eq!((span.start, span.end), (5, 7));
    }

    #[test]
    fn test_locate_span_wrapped_word() {
        // "configuration" hard-wrapped across a line boundary:
        // pane renders it as "configu" / "ration" on separate lines.
        let hay = toks("$ update the configu\nration file now");
        let needle = toks("update the configuration file");
        let span = locate_span(&hay, &needle).unwrap();
        assert_eq!(span.start, 1);
        // The wrapped token consumed two haystack tokens.
        assert_eq!(span.end, 6);
    }

    #[test]
    fn test_locate_span_word_split_three_ways() {
        let hay = toks("$ super\ncali\nfragilistic done");
        let needle = toks("supercalifragilistic done");
        let span = locate_span(&hay, &needle).unwrap();
        assert_eq!((span.start, span.end), (1, 5));
    }

    #[test]
    fn test_locate_span_not_found() {
        let hay = toks("$ ls -la");
        let needle = toks("hello world");
        assert!(locate_span(&hay, &needle).is_none());
    }

    #[test]
    fn test_locate_span_empty_needle() {
        assert!(locate_span(&toks("a b c"), &[]).is_none());
    }

    // --- token_diff ---

    #[test]
    fn test_token_diff_identical() {
        let a = toks("a b c");
        assert_eq!(token_diff(&a, &a).unwrap(), Vec::new());
    }

    #[test]
    fn test_token_diff_single_replacement() {
        let a = toks("prompt $ the cash is here");
        let b = toks("prompt $ the cache is here");
        let regions = token_diff(&a, &b).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!((regions[0].a_start, regions[0].a_end), (3, 4));
        assert_eq!((regions[0].b_start, regions[0].b_end), (3, 4));
    }

    #[test]
    fn test_token_diff_insert_and_delete() {
        let a = toks("one two three four");
        let b = toks("one three four five");
        let regions = token_diff(&a, &b).unwrap();
        assert_eq!(regions.len(), 2);
        // "two" deleted
        assert_eq!((regions[0].a_start, regions[0].a_end), (1, 2));
        assert_eq!(regions[0].b_start, regions[0].b_end);
        // "five" appended
        assert_eq!(regions[1].a_start, regions[1].a_end);
        assert_eq!((regions[1].b_start, regions[1].b_end), (3, 4));
    }

    // --- diff_revision → pair extraction ---

    #[test]
    fn test_diff_revision_word_replacement() {
        let context = make_context("the cash is here");
        let baseline = toks(
            "scrollback stuff above the pane area filled with old output lines \
                             plus some more filler tokens to pass churn minimum \
                             $ the cash is here",
        );
        let span = locate_span(&baseline, &toks("the cash is here")).unwrap();
        let current = toks(
            "scrollback stuff above the pane area filled with old output lines \
                            plus some more filler tokens to pass churn minimum \
                            $ the cache is here",
        );

        let outcome = diff_revision(&baseline, span, &current, &context);
        let RevisionOutcome::Corrections(pairs) = outcome else {
            panic!("expected Corrections, got {:?}", outcome);
        };
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].original, "cash");
        assert_eq!(pairs[0].corrected, "cache");
        assert_eq!(pairs[0].window_class, WEZTERM_WINDOW_CLASS);
    }

    #[test]
    fn test_diff_revision_unchanged() {
        let context = make_context("hello world");
        let baseline = toks("$ hello world");
        let span = locate_span(&baseline, &toks("hello world")).unwrap();

        let outcome = diff_revision(&baseline, span, &baseline, &context);
        assert_eq!(outcome, RevisionOutcome::Corrections(Vec::new()));
    }

    #[test]
    fn test_diff_revision_ignores_new_output_after_span() {
        // User pressed Enter: new output appears after the injected text.
        let context = make_context("git status");
        let baseline = toks("$ git status");
        let span = locate_span(&baseline, &toks("git status")).unwrap();
        let current = toks("$ git status On branch main nothing to commit $");

        let outcome = diff_revision(&baseline, span, &current, &context);
        assert_eq!(outcome, RevisionOutcome::Corrections(Vec::new()));
    }

    #[test]
    fn test_diff_revision_full_line_rewrite_extracts_word_pair() {
        // Readline editing rewrites the whole line between polls; the LCS
        // must still isolate the single changed word.
        let context = make_context("please install the parakeet model today");
        let baseline = toks("$ please install the parakeet model today");
        let span =
            locate_span(&baseline, &toks("please install the parakeet model today")).unwrap();
        let current = toks("$ please install the parquet model today");

        let RevisionOutcome::Corrections(pairs) =
            diff_revision(&baseline, span, &current, &context)
        else {
            panic!("expected Corrections");
        };
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].original, "parakeet");
        assert_eq!(pairs[0].corrected, "parquet");
    }

    #[test]
    fn test_diff_revision_multi_word_replacement() {
        let context = make_context("say hi to shay moy for me");
        let baseline = toks("$ say hi to shay moy for me");
        let span = locate_span(&baseline, &toks("say hi to shay moy for me")).unwrap();
        let current = toks("$ say hi to chezmoi for me");

        let RevisionOutcome::Corrections(pairs) =
            diff_revision(&baseline, span, &current, &context)
        else {
            panic!("expected Corrections");
        };
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].original, "shay moy");
        assert_eq!(pairs[0].corrected, "chezmoi");
    }

    #[test]
    fn test_diff_revision_partial_word_deletion() {
        let context = make_context("remove this extra word");
        let baseline = toks("$ remove this extra word");
        let span = locate_span(&baseline, &toks("remove this extra word")).unwrap();
        let current = toks("$ remove this word");

        let RevisionOutcome::Corrections(pairs) =
            diff_revision(&baseline, span, &current, &context)
        else {
            panic!("expected Corrections");
        };
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].original, "extra");
        assert_eq!(pairs[0].corrected, "");
    }

    #[test]
    fn test_diff_revision_span_lost_on_clear() {
        let context = make_context("hello world");
        let baseline = toks("old output line one two three $ hello world");
        let span = locate_span(&baseline, &toks("hello world")).unwrap();
        // Pane cleared to a bare prompt; small pane so churn guard stays out.
        let current = toks("old output line one two three $");

        let outcome = diff_revision(&baseline, span, &current, &context);
        assert_eq!(outcome, RevisionOutcome::SpanLost);
    }

    #[test]
    fn test_diff_revision_churn_on_tui_redraw() {
        let context = make_context("hello world");
        let baseline = toks(
            "line1 aa bb line2 cc dd line3 ee ff line4 gg hh line5 ii jj \
             line6 kk ll line7 mm nn $ hello world",
        );
        let span = locate_span(&baseline, &toks("hello world")).unwrap();
        // nvim took over: everything replaced
        let current = toks(
            "NORMAL main.rs 1,1 All ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \
             ~ ~ ~ ~ ~ ~ ~ ~ set nu",
        );

        let outcome = diff_revision(&baseline, span, &current, &context);
        assert_eq!(outcome, RevisionOutcome::Churn);
    }

    #[test]
    fn test_span_char_offset() {
        let baseline = toks("$ the cash is here");
        let span = Span { start: 1, end: 5 }; // "the cash is here"
        assert_eq!(span_char_offset(&baseline, span, 1), 0); // "the"
        assert_eq!(span_char_offset(&baseline, span, 2), 4); // "cash"
        assert_eq!(span_char_offset(&baseline, span, 4), 12); // "here"
    }

    // --- monitor end-to-end with a scripted client ---

    struct MockClient {
        panes: Vec<PaneEntry>,
        /// Revisions handed out per get_text call; last one repeats.
        revisions: StdMutex<VecDeque<String>>,
        last: StdMutex<String>,
    }

    impl MockClient {
        fn new(revisions: Vec<&str>) -> Self {
            let mut queue: VecDeque<String> = revisions.into_iter().map(str::to_string).collect();
            let last = queue.back().cloned().unwrap_or_default();
            // Keep the last revision out of the queue so it repeats forever.
            if queue.len() > 1 {
                queue.pop_back();
            }
            Self {
                panes: vec![PaneEntry {
                    pane_id: 7,
                    is_active: true,
                    title: "fish".to_string(),
                    workspace: "default".to_string(),
                }],
                revisions: StdMutex::new(queue),
                last: StdMutex::new(last),
            }
        }
    }

    #[async_trait]
    impl WeztermClient for MockClient {
        async fn list_panes(&self) -> Result<Vec<PaneEntry>> {
            Ok(self.panes.clone())
        }

        async fn get_text(&self, _pane_id: u64) -> Result<String> {
            let mut q = self.revisions.lock().unwrap();
            match q.pop_front() {
                Some(rev) => Ok(rev),
                None => Ok(self.last.lock().unwrap().clone()),
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_monitor_records_correction_via_mock_client() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = test_config(dir.path());
        let store = CorrectionStore::empty(config.clone());

        let client = Arc::new(MockClient::new(vec![
            // Baseline snapshot at injection time
            "$ the cash is here",
            // User corrected it
            "$ the cache is here",
        ]));

        let monitor = WeztermMonitor::with_client(client, store, config);
        let context = make_context("the cash is here");
        let pairs = monitor.start_monitoring(context).await.unwrap();

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].original, "cash");
        assert_eq!(pairs[0].corrected, "cache");

        // Recorded into the store (respecting blocklist path)
        let stats = {
            let store = monitor.store.lock().await;
            store.stats()
        };
        assert_eq!(stats.total_corrections, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn test_monitor_bails_when_injection_not_found() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = test_config(dir.path());
        let store = CorrectionStore::empty(config.clone());

        // Pane never shows the injected text (e.g. nvim swallowed it)
        let client = Arc::new(MockClient::new(vec!["NORMAL ~ ~ ~ main.rs"]));

        let monitor = WeztermMonitor::with_client(client, store, config);
        let pairs = monitor.start_monitoring(make_context("hello world")).await.unwrap();
        assert!(pairs.is_empty());

        let stats = {
            let store = monitor.store.lock().await;
            store.stats()
        };
        assert_eq!(stats.total_corrections, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_monitor_blocklisted_pair_not_recorded() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = test_config(dir.path());
        let mut store = CorrectionStore::empty(config.clone());
        // Reject the pair up-front: record then remove → blocklisted
        store
            .record_correction(CorrectionPair {
                original: "cash".to_string(),
                corrected: "cache".to_string(),
                context_before: String::new(),
                context_after: String::new(),
                window_class: WEZTERM_WINDOW_CLASS.to_string(),
                timestamp: Utc::now(),
            })
            .unwrap();
        store.remove("cash").unwrap();

        let client = Arc::new(MockClient::new(vec!["$ the cash is here", "$ the cache is here"]));
        let monitor = WeztermMonitor::with_client(client, store, config);
        let pairs = monitor.start_monitoring(make_context("the cash is here")).await.unwrap();

        // The pair is still *detected*...
        assert_eq!(pairs.len(), 1);
        // ...but the blocklist keeps it out of the store.
        let stats = {
            let store = monitor.store.lock().await;
            store.stats()
        };
        assert_eq!(stats.total_corrections, 0);
    }

    // --- live integration (read-only; requires a running wezterm GUI) ---

    #[tokio::test]
    #[ignore = "requires a live wezterm GUI instance; read-only probe"]
    async fn live_socket_list_and_get_text() {
        let Some(cli) = WeztermCli::discover() else {
            eprintln!("no live wezterm socket — skipping");
            return;
        };
        println!("socket: {}", cli.socket().display());

        let panes = cli.list_panes().await.expect("list_panes failed");
        assert!(!panes.is_empty(), "expected at least one pane");
        println!("{} pane(s); active: {:?}", panes.len(), panes.iter().find(|p| p.is_active));

        let pane = panes.iter().find(|p| p.is_active).unwrap_or(&panes[0]);
        let text = cli.get_text(pane.pane_id).await.expect("get_text failed");
        assert!(!text.is_empty(), "expected pane text");
        for line in text.lines().take(5) {
            println!("| {}", line.trim_end());
        }
    }
}
