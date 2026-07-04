//! Time-boxed `zwp_input_method_v2` diagnostic probe (daemon-as-IME prototype).
//!
//! Binds the compositor's input-method slot for a fixed number of seconds and
//! logs every IME event delivered by the compositor (activate / deactivate /
//! surrounding_text / text_change_cause / content_type / done / unavailable),
//! tagging each with the focused window class from `hyprctl activewindow -j`.
//!
//! This exists to answer, live on a Hyprland session, whether the
//! input-method-v2 read path (surrounding_text — correction detection) and
//! write path (commit_string — in-place fixes) work against real clients
//! (GTK/zenity, Chromium/Electron with `--enable-wayland-ime`, wezterm).
//!
//! Safety properties, by construction:
//! - Time-boxed: the wayland connection is dropped when the deadline passes;
//!   there is no daemon integration.
//! - Never requests a keyboard grab (`grab_keyboard` is never called), so the
//!   user's typing is unaffected while the probe runs.
//! - `commit_string` is only ever sent when BOTH `--commit` and
//!   `--commit-class` are given AND the focused window class matches
//!   `--commit-class` exactly. Intended for throwaway targets
//!   (e.g. `zenity --entry`) only.
//! - If the compositor reports `unavailable` (another IME holds the slot),
//!   the probe records that fact and exits immediately — no retry loop.

use serde::Serialize;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use wayland_client::{
    delegate_noop,
    globals::{registry_queue_init, GlobalListContents},
    protocol::{wl_registry, wl_seat::WlSeat},
    Connection, Dispatch, Proxy, QueueHandle, WEnum,
};
use wayland_protocols_misc::zwp_input_method_v2::client::{
    zwp_input_method_manager_v2::ZwpInputMethodManagerV2,
    zwp_input_method_v2::{Event as ImeEvent, ZwpInputMethodV2},
};

/// Options for a probe run.
#[derive(Debug, Clone)]
pub struct ProbeOptions {
    /// How long to hold the IME slot before disconnecting.
    pub secs: u64,
    /// Emit a machine-readable JSON summary on stdout (event lines go to
    /// stderr in this mode).
    pub json: bool,
    /// Text to commit into the focused editor for write-path validation.
    /// Requires `commit_class`; only sent while a window of that class is
    /// focused.
    pub commit_text: Option<String>,
    /// Exact window class the commit is restricted to (safety gate).
    pub commit_class: Option<String>,
}

/// One observed protocol event (or probe action), tagged with wall-clock
/// offset and the focused window at the time it was applied.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeRecord {
    /// Milliseconds since probe start.
    pub t_ms: u64,
    /// Focused window class per hyprctl at the surrounding `done`.
    pub class: Option<String>,
    /// Focused window title per hyprctl.
    pub title: Option<String>,
    #[serde(flatten)]
    pub kind: RecordKind,
}

/// The event payload of a [`ProbeRecord`].
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RecordKind {
    Activate,
    Deactivate,
    SurroundingText {
        text: String,
        cursor: u32,
        anchor: u32,
    },
    TextChangeCause {
        cause: String,
    },
    ContentType {
        hint: String,
        purpose: String,
    },
    Done {
        serial: u32,
    },
    Unavailable,
    /// Probe-initiated commit_string + commit (not a compositor event).
    Committed {
        text: String,
    },
}

/// Snapshot of the last surrounding_text seen for a window class.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SurroundingSnapshot {
    pub text: String,
    pub cursor: u32,
    pub anchor: u32,
}

/// Per-window-class findings, aggregated from the event stream.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ClassFindings {
    pub class: String,
    pub activations: u32,
    pub deactivations: u32,
    pub surrounding_text_events: u32,
    pub last_surrounding: Option<SurroundingSnapshot>,
    /// Unique text_change_cause labels in order of first appearance.
    pub change_causes: Vec<String>,
    /// Unique content_type labels ("hint / purpose") in order of first
    /// appearance.
    pub content_types: Vec<String>,
}

/// Outcome of the (optional) commit_string write-path test.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CommitOutcome {
    pub text: String,
    pub target_class: String,
    /// True if the probe actually sent commit_string (target was focused
    /// while active).
    pub committed: bool,
    /// True if a later surrounding_text for the same class contained the
    /// committed text (full read-back round trip).
    pub roundtrip_confirmed: bool,
}

/// Machine-readable summary of a probe run.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProbeSummary {
    pub duration_ms: u64,
    /// The compositor refused the IME slot (another IME is bound).
    pub unavailable: bool,
    pub total_done_events: u32,
    pub per_class: Vec<ClassFindings>,
    pub commit_test: Option<CommitOutcome>,
}

/// Build the aggregate summary from the raw record stream. Pure function so
/// it is unit-testable without a compositor.
pub fn build_summary(
    records: &[ProbeRecord],
    duration_ms: u64,
    commit_plan: Option<(&str, &str)>,
) -> ProbeSummary {
    let mut per_class: Vec<ClassFindings> = Vec::new();
    let mut unavailable = false;
    let mut total_done = 0u32;

    let class_of = |r: &ProbeRecord| r.class.clone().unwrap_or_else(|| "<unknown>".to_string());

    for rec in records {
        match &rec.kind {
            RecordKind::Unavailable => unavailable = true,
            RecordKind::Done { .. } => total_done += 1,
            _ => {}
        }
        let class = class_of(rec);
        let entry = match per_class.iter_mut().find(|c| c.class == class) {
            Some(e) => e,
            None => {
                per_class.push(ClassFindings {
                    class: class.clone(),
                    activations: 0,
                    deactivations: 0,
                    surrounding_text_events: 0,
                    last_surrounding: None,
                    change_causes: Vec::new(),
                    content_types: Vec::new(),
                });
                per_class.last_mut().expect("just pushed")
            }
        };
        match &rec.kind {
            RecordKind::Activate => entry.activations += 1,
            RecordKind::Deactivate => entry.deactivations += 1,
            RecordKind::SurroundingText { text, cursor, anchor } => {
                entry.surrounding_text_events += 1;
                entry.last_surrounding = Some(SurroundingSnapshot {
                    text: text.clone(),
                    cursor: *cursor,
                    anchor: *anchor,
                });
            }
            RecordKind::TextChangeCause { cause } => {
                if !entry.change_causes.contains(cause) {
                    entry.change_causes.push(cause.clone());
                }
            }
            RecordKind::ContentType { hint, purpose } => {
                let label = format!("{hint} / {purpose}");
                if !entry.content_types.contains(&label) {
                    entry.content_types.push(label);
                }
            }
            _ => {}
        }
    }

    let commit_test = commit_plan.map(|(text, target_class)| {
        let committed_at = records.iter().position(
            |r| matches!(&r.kind, RecordKind::Committed { text: t } if t == text),
        );
        let roundtrip_confirmed = committed_at.is_some_and(|idx| {
            records[idx + 1..].iter().any(|r| {
                r.class.as_deref() == Some(target_class)
                    && matches!(&r.kind, RecordKind::SurroundingText { text: t, .. } if t.contains(text))
            })
        });
        CommitOutcome {
            text: text.to_string(),
            target_class: target_class.to_string(),
            committed: committed_at.is_some(),
            roundtrip_confirmed,
        }
    });

    ProbeSummary { duration_ms, unavailable, total_done_events: total_done, per_class, commit_test }
}

/// Render a `WEnum` payload as a stable human label ("Other", "InputMethod",
/// "unknown(7)", ...).
fn wenum_label<T: std::fmt::Debug>(e: &WEnum<T>) -> String {
    match e {
        WEnum::Value(v) => format!("{v:?}"),
        WEnum::Unknown(u) => format!("unknown({u})"),
    }
}

#[derive(Debug, Default, Clone)]
struct PendingState {
    activate: bool,
    deactivate: bool,
    surrounding: Option<(String, u32, u32)>,
    change_cause: Option<String>,
    content_type: Option<(String, String)>,
}

struct ProbeState {
    start: Instant,
    json: bool,
    records: Vec<ProbeRecord>,
    pending: PendingState,
    active: bool,
    done_count: u32,
    unavailable: bool,
    commit_text: Option<String>,
    commit_class: Option<String>,
    committed: bool,
}

impl ProbeState {
    fn t_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    fn push(&mut self, class: &Option<String>, title: &Option<String>, kind: RecordKind) {
        let rec =
            ProbeRecord { t_ms: self.t_ms(), class: class.clone(), title: title.clone(), kind };
        let line = format!(
            "[ime-probe +{:>6}ms] class={:?} {}",
            rec.t_ms,
            rec.class.as_deref().unwrap_or("?"),
            match &rec.kind {
                RecordKind::Activate => "activate".to_string(),
                RecordKind::Deactivate => "deactivate".to_string(),
                RecordKind::SurroundingText { text, cursor, anchor } =>
                    format!("surrounding_text text={text:?} cursor={cursor} anchor={anchor}"),
                RecordKind::TextChangeCause { cause } => format!("text_change_cause={cause}"),
                RecordKind::ContentType { hint, purpose } =>
                    format!("content_type hint={hint} purpose={purpose}"),
                RecordKind::Done { serial } => format!("done serial={serial}"),
                RecordKind::Unavailable => "UNAVAILABLE (another IME holds the slot)".to_string(),
                RecordKind::Committed { text } => format!("-> commit_string({text:?}) + commit"),
            }
        );
        if self.json {
            eprintln!("{line}");
        } else {
            println!("{line}");
        }
        self.records.push(rec);
    }

    /// Apply the pending (double-buffered) state on `done`, recording each
    /// component tagged with the currently focused window.
    fn apply_done(&mut self, im: &ZwpInputMethodV2) {
        let (class, title) = focused_window();
        let pending = std::mem::take(&mut self.pending);

        if pending.activate {
            self.active = true;
            self.push(&class, &title, RecordKind::Activate);
        }
        if pending.deactivate {
            self.active = false;
            self.push(&class, &title, RecordKind::Deactivate);
        }
        if let Some((text, cursor, anchor)) = pending.surrounding {
            self.push(&class, &title, RecordKind::SurroundingText { text, cursor, anchor });
        }
        if let Some(cause) = pending.change_cause {
            self.push(&class, &title, RecordKind::TextChangeCause { cause });
        }
        if let Some((hint, purpose)) = pending.content_type {
            self.push(&class, &title, RecordKind::ContentType { hint, purpose });
        }
        self.done_count += 1;
        let serial = self.done_count;
        self.push(&class, &title, RecordKind::Done { serial });

        // Write-path test: commit only once, only while active, and only if
        // the focused window class matches the explicit safety gate.
        if !self.committed && self.active {
            if let (Some(text), Some(gate)) = (&self.commit_text, &self.commit_class) {
                if class.as_deref() == Some(gate.as_str()) {
                    im.commit_string(text.clone());
                    im.commit(serial);
                    self.committed = true;
                    let text = text.clone();
                    self.push(&class, &title, RecordKind::Committed { text });
                }
            }
        }
    }
}

/// Focused window (class, title) from Hyprland, best-effort.
fn focused_window() -> (Option<String>, Option<String>) {
    let Ok(out) = std::process::Command::new("hyprctl").args(["-j", "activewindow"]).output()
    else {
        return (None, None);
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&out.stdout) else {
        return (None, None);
    };
    (
        v.get("class").and_then(|c| c.as_str()).map(str::to_string),
        v.get("title").and_then(|t| t.as_str()).map(str::to_string),
    )
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for ProbeState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_registry::WlRegistry,
        _event: wl_registry::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

delegate_noop!(ProbeState: ignore WlSeat);
delegate_noop!(ProbeState: ZwpInputMethodManagerV2);

impl Dispatch<ZwpInputMethodV2, ()> for ProbeState {
    fn event(
        state: &mut Self,
        proxy: &ZwpInputMethodV2,
        event: ImeEvent,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            ImeEvent::Activate => state.pending.activate = true,
            ImeEvent::Deactivate => state.pending.deactivate = true,
            ImeEvent::SurroundingText { text, cursor, anchor } => {
                state.pending.surrounding = Some((text, cursor, anchor));
            }
            ImeEvent::TextChangeCause { cause } => {
                state.pending.change_cause = Some(wenum_label(&cause));
            }
            ImeEvent::ContentType { hint, purpose } => {
                state.pending.content_type = Some((wenum_label(&hint), wenum_label(&purpose)));
            }
            ImeEvent::Done => state.apply_done(proxy),
            ImeEvent::Unavailable => {
                let (class, title) = focused_window();
                state.unavailable = true;
                state.push(&class, &title, RecordKind::Unavailable);
            }
            _ => {}
        }
    }
}

/// Run the probe. Blocks for up to `opts.secs` seconds, then prints a summary
/// (JSON on stdout if `opts.json`).
pub fn run(opts: ProbeOptions) -> anyhow::Result<()> {
    if opts.commit_text.is_some() && opts.commit_class.is_none() {
        anyhow::bail!("--commit requires --commit-class (safety gate)");
    }

    let conn = Connection::connect_to_env()
        .map_err(|e| anyhow::anyhow!("failed to connect to wayland display: {e}"))?;
    let (globals, mut queue) = registry_queue_init::<ProbeState>(&conn)
        .map_err(|e| anyhow::anyhow!("wayland registry init failed: {e}"))?;
    let qh = queue.handle();

    let seat: WlSeat =
        globals.bind(&qh, 1..=4, ()).map_err(|e| anyhow::anyhow!("failed to bind wl_seat: {e}"))?;
    let manager: ZwpInputMethodManagerV2 = globals.bind(&qh, 1..=1, ()).map_err(|e| {
        anyhow::anyhow!("compositor does not expose zwp_input_method_manager_v2: {e}")
    })?;

    let mut state = ProbeState {
        start: Instant::now(),
        json: opts.json,
        records: Vec::new(),
        pending: PendingState::default(),
        active: false,
        done_count: 0,
        unavailable: false,
        commit_text: opts.commit_text.clone(),
        commit_class: opts.commit_class.clone(),
        committed: false,
    };

    let banner = format!(
        "[ime-probe] bound {} v1 on seat {:?}; holding IME slot for {}s (no keyboard grab)",
        ZwpInputMethodManagerV2::interface().name,
        seat.id(),
        opts.secs
    );
    if opts.json {
        eprintln!("{banner}");
    } else {
        println!("{banner}");
    }

    let input_method = manager.get_input_method(&seat, &qh, ());
    queue.flush()?;

    let deadline = state.start + Duration::from_secs(opts.secs);
    loop {
        queue.dispatch_pending(&mut state)?;
        if state.unavailable {
            break;
        }
        queue.flush()?;
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let remaining = deadline - now;
        // prepare_read is None when events are already pending; loop again.
        if let Some(guard) = queue.prepare_read() {
            let fd = guard.connection_fd().as_raw_fd();
            let mut pfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
            let timeout_ms = remaining.as_millis().min(250) as i32;
            let ret = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
            if ret > 0 {
                // Socket readable: pull events; dispatched on next loop turn.
                let _ = guard.read();
            }
            // ret == 0: timeout, guard dropped cancels the read.
        }
    }

    input_method.destroy();
    let _ = queue.flush();

    let duration_ms = state.start.elapsed().as_millis() as u64;
    let commit_plan = match (&opts.commit_text, &opts.commit_class) {
        (Some(t), Some(c)) => Some((t.as_str(), c.as_str())),
        _ => None,
    };
    let summary = build_summary(&state.records, duration_ms, commit_plan);

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_human_summary(&summary);
    }

    if summary.unavailable {
        anyhow::bail!(
            "compositor reported zwp_input_method_v2.unavailable — another IME holds the slot; \
             not retrying"
        );
    }
    Ok(())
}

fn print_human_summary(s: &ProbeSummary) {
    println!("\n=== ime-probe summary ({} ms) ===", s.duration_ms);
    println!("unavailable: {}", s.unavailable);
    println!("done events: {}", s.total_done_events);
    for c in &s.per_class {
        println!("- class {:?}:", c.class);
        println!(
            "    activations={} deactivations={} surrounding_text_events={}",
            c.activations, c.deactivations, c.surrounding_text_events
        );
        if let Some(sur) = &c.last_surrounding {
            println!(
                "    last surrounding: text={:?} cursor={} anchor={}",
                sur.text, sur.cursor, sur.anchor
            );
        }
        if !c.change_causes.is_empty() {
            println!("    change causes: {}", c.change_causes.join(", "));
        }
        if !c.content_types.is_empty() {
            println!("    content types: {}", c.content_types.join("; "));
        }
    }
    if let Some(ct) = &s.commit_test {
        println!(
            "commit test into class {:?}: committed={} roundtrip_confirmed={}",
            ct.target_class, ct.committed, ct.roundtrip_confirmed
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(t_ms: u64, class: Option<&str>, kind: RecordKind) -> ProbeRecord {
        ProbeRecord {
            t_ms,
            class: class.map(str::to_string),
            title: class.map(|c| format!("{c} window")),
            kind,
        }
    }

    #[test]
    fn summary_aggregates_per_class() {
        let records = vec![
            rec(10, Some("zenity"), RecordKind::Activate),
            rec(
                10,
                Some("zenity"),
                RecordKind::SurroundingText { text: "hi".into(), cursor: 2, anchor: 2 },
            ),
            rec(10, Some("zenity"), RecordKind::TextChangeCause { cause: "Other".into() }),
            rec(10, Some("zenity"), RecordKind::Done { serial: 1 }),
            rec(500, Some("zenity"), RecordKind::Deactivate),
            rec(500, Some("zenity"), RecordKind::Done { serial: 2 }),
            rec(900, Some("chromium"), RecordKind::Activate),
            rec(
                900,
                Some("chromium"),
                RecordKind::SurroundingText { text: "form".into(), cursor: 4, anchor: 4 },
            ),
            rec(900, Some("chromium"), RecordKind::Done { serial: 3 }),
        ];
        let s = build_summary(&records, 1000, None);
        assert!(!s.unavailable);
        assert_eq!(s.total_done_events, 3);
        assert_eq!(s.per_class.len(), 2);
        let zen = &s.per_class[0];
        assert_eq!(zen.class, "zenity");
        assert_eq!(zen.activations, 1);
        assert_eq!(zen.deactivations, 1);
        assert_eq!(zen.surrounding_text_events, 1);
        assert_eq!(zen.change_causes, vec!["Other".to_string()]);
        let chr = &s.per_class[1];
        assert_eq!(chr.class, "chromium");
        assert_eq!(
            chr.last_surrounding,
            Some(SurroundingSnapshot { text: "form".into(), cursor: 4, anchor: 4 })
        );
        assert!(s.commit_test.is_none());
    }

    #[test]
    fn change_causes_are_deduped_in_first_seen_order() {
        let records = vec![
            rec(1, Some("zenity"), RecordKind::TextChangeCause { cause: "Other".into() }),
            rec(2, Some("zenity"), RecordKind::TextChangeCause { cause: "InputMethod".into() }),
            rec(3, Some("zenity"), RecordKind::TextChangeCause { cause: "Other".into() }),
        ];
        let s = build_summary(&records, 10, None);
        assert_eq!(
            s.per_class[0].change_causes,
            vec!["Other".to_string(), "InputMethod".to_string()]
        );
    }

    #[test]
    fn commit_roundtrip_confirmed_by_later_surrounding_text() {
        let records = vec![
            rec(10, Some("zenity"), RecordKind::Activate),
            rec(10, Some("zenity"), RecordKind::Done { serial: 1 }),
            rec(11, Some("zenity"), RecordKind::Committed { text: "probe text".into() }),
            rec(
                90,
                Some("zenity"),
                RecordKind::SurroundingText { text: "probe text".into(), cursor: 10, anchor: 10 },
            ),
            rec(90, Some("zenity"), RecordKind::Done { serial: 2 }),
        ];
        let s = build_summary(&records, 100, Some(("probe text", "zenity")));
        let ct = s.commit_test.expect("commit test present");
        assert!(ct.committed);
        assert!(ct.roundtrip_confirmed);
    }

    #[test]
    fn commit_roundtrip_not_confirmed_without_readback() {
        let records = vec![
            rec(10, Some("zenity"), RecordKind::Activate),
            rec(10, Some("zenity"), RecordKind::Done { serial: 1 }),
            rec(11, Some("zenity"), RecordKind::Committed { text: "probe text".into() }),
        ];
        let s = build_summary(&records, 100, Some(("probe text", "zenity")));
        let ct = s.commit_test.expect("commit test present");
        assert!(ct.committed);
        assert!(!ct.roundtrip_confirmed);
    }

    #[test]
    fn commit_readback_from_other_class_does_not_count() {
        let records = vec![
            rec(11, Some("zenity"), RecordKind::Committed { text: "probe text".into() }),
            rec(
                90,
                Some("chromium"),
                RecordKind::SurroundingText { text: "probe text".into(), cursor: 10, anchor: 10 },
            ),
        ];
        let s = build_summary(&records, 100, Some(("probe text", "zenity")));
        assert!(!s.commit_test.expect("commit test present").roundtrip_confirmed);
    }

    #[test]
    fn commit_never_sent_reports_uncommitted() {
        let s = build_summary(&[], 100, Some(("probe text", "zenity")));
        let ct = s.commit_test.expect("commit test present");
        assert!(!ct.committed);
        assert!(!ct.roundtrip_confirmed);
    }

    #[test]
    fn unavailable_is_surfaced() {
        let records = vec![rec(5, Some("wezterm"), RecordKind::Unavailable)];
        let s = build_summary(&records, 10, None);
        assert!(s.unavailable);
    }

    #[test]
    fn unknown_class_is_bucketed() {
        let records = vec![rec(5, None, RecordKind::Activate)];
        let s = build_summary(&records, 10, None);
        assert_eq!(s.per_class[0].class, "<unknown>");
        assert_eq!(s.per_class[0].activations, 1);
    }

    #[test]
    fn summary_json_shape() {
        let records = vec![
            rec(10, Some("zenity"), RecordKind::Activate),
            rec(
                10,
                Some("zenity"),
                RecordKind::SurroundingText { text: "hi".into(), cursor: 2, anchor: 2 },
            ),
            rec(10, Some("zenity"), RecordKind::Done { serial: 1 }),
        ];
        let s = build_summary(&records, 42, Some(("x", "zenity")));
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(v["duration_ms"], 42);
        assert_eq!(v["unavailable"], false);
        assert_eq!(v["total_done_events"], 1);
        assert_eq!(v["per_class"][0]["class"], "zenity");
        assert_eq!(v["per_class"][0]["last_surrounding"]["text"], "hi");
        assert_eq!(v["commit_test"]["committed"], false);
    }

    #[test]
    fn record_json_uses_event_tag() {
        let r = rec(
            7,
            Some("zenity"),
            RecordKind::SurroundingText { text: "abc".into(), cursor: 3, anchor: 1 },
        );
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(v["event"], "surrounding_text");
        assert_eq!(v["cursor"], 3);
        assert_eq!(v["anchor"], 1);
        assert_eq!(v["class"], "zenity");
    }

    #[test]
    fn wenum_label_formats_unknown() {
        let unknown: WEnum<wayland_client::protocol::wl_seat::Capability> = WEnum::Unknown(999);
        assert_eq!(wenum_label(&unknown), "unknown(999)");
    }
}
