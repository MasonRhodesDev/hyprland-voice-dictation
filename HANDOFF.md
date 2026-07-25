# Crew handoff — hyprland-voice-dictation (voice-model / engine-event-contract)

Seeded from Claude session `a26e2dab-ddea-427d-a819-5b7d43f51860` (run in `~`, 2026-07-21).
Branch: **`feat/engine-event-contract`** (local-only — never pushed to origin). This
workspace has the committed work **plus** the uncommitted in-progress engine edits
transferred from `~/repos/hyprland-voice-dictation` (lib.rs, openai_engine.rs,
stream_engine.rs). The throwaway `dictation-engine/examples/` repro was intentionally
NOT carried over.

## What the session was doing
Evaluating cloud voice-to-text to replace/augment the local **parakeet** model, and
refactoring the engine to an event-emitting contract so a cloud engine can drop in.

## Decisions reached
- **Model: `openai:whisper-1`** is the right fit for how Mason actually dictates —
  long, pause-heavy sentences. Validated on real 20s pause-heavy utterances: full
  transcript, **no truncation, no invented words during pauses.**
- Rejected `gpt-4o-transcribe`: faster (~0.9s vs whisper's ~2s round trip) but
  **chops the sentence at the first pause** — unacceptable for Mason's speech pattern.
- **`gpt-realtime-whisper`** (WebSocket) remains a *future* option — would give live
  streaming preview AND pause-robustness, if worth the WebSocket build later.
- Caveat to watch: whisper can hallucinate on **very long dead-air (10s+)**; normal
  thinking pauses were fine.

## Done (committed on this branch)
- ✅ Event-emitting `StreamingEngine` contract + `LocalEngineDriver` (local parity:
  partials + correct finalize) — `baa3ba1`
- ✅ Daemon loop rewired to the event-emitting engine — `aadf1f2`
- ✅ OpenAI opt-in gating + OpenAI batch transcription (whisper-1) — `4613b07`
- ⏳ Uncommitted WIP in the 3 engine files: debug logging + whisper wiring.

## Remaining work (your tasks)
1. **Settle the default engine/model** — keep `openai:whisper-1` as the default, or
   make it a per-use toggle? (Realtime whisper = separate future WebSocket build.)
2. **Apply the GUI live-preview fix** — root cause: the custom pill overlay has **no
   text element**, so words never render while speaking. Fix = remove or patch that
   custom overlay so the live preview shows. (Not yet applied.)
3. **Cleanup / deliberate commit-or-revert:**
   - Strip the throwaway repro (`dictation-engine/examples/` — not in this workspace).
   - Keep the driver/openai logs as **debug-level** observability.
   - Decide on the test scaffolding left in `~/repos`: systemd drop-in pointing at the
     debug binary (`Restart=no`), imported API key, config pointed at OpenAI — commit
     intentionally or revert; don't ship the debug wiring.

## First move
Read `CURRENT_PLAN.md` / `GUI_BLOCKING_ISSUE.md` in the repo for prior context, then
confirm the whisper-1 default with Mason (task 1) before wiring the GUI fix (task 2).

---

## Session progress (echo, 2026-07-21)

### Task 1 — default engine/model: SETTLED
Mason: **keep parakeet as default**, and **consolidate to a single swappable model
interface** (one `model` key, parakeet default, any StreamingEngine toggled in its
place, ideally generic enough to add models without per-provider code). The daemon is
already single-model (`model`, serde alias `preview_model`); `final_model` was always a
dead key. Filed **bd hvd-eb0** for the full consolidation + generic-interface refactor.

### Task 2 — GUI live-preview: FIXED (validated, not yet seen live)
Root cause: the runtime UI `~/.config/voice-dictation/ui/dictation.slint` (a copy of
`style2-minimal`) declared the `text` property (so the contract check passed) but had
**no `Text` widget** in listening mode — words were set every tick and never rendered.
Fix: refactored into functional Slint components (`SpectrumBars`, `DotRing`,
`ListeningPill`, `ProcessingIndicator`, `ClosingIndicator`); added a centered live
transcription line under the spectrum; all modes centered in the surface. Compiles +
passes the 8-property contract. Same fix synced to the repo source
`slint-gui/ui/examples/style2-minimal.slint` (which was stale/failing the contract).

Key interaction: **whisper-1 is batch — emits no `Partial`s**, so it can't drive live
streaming words in the pill. For the fix to show words *while speaking*, the model must
be a streaming engine → parakeet.

### Task 3 — cleanup
- Throwaway repro: absent in this workspace. ✅
- Driver/openai debug logs: kept at debug-level (the staged WIP in the 3 engine files). ✅
- OpenAI test config: reverted → `~/.config/voice-dictation/config.toml` now single
  `model = "parakeet:default"` (was `preview_model = openai:whisper-1` + dead
  `final_model`). ✅
- systemd `branch-test.conf` drop-in (Restart=no, points at `~/repos` debug binary,
  RUST_LOG debug): left in place — it's Mason's temporary test harness with revert
  steps in its header. ⏳ his call.

### To see the fix live
The running daemon is the **`~/repos` debug build** with the *old* config in memory
(started 09:13, before these edits) and **Restart=no**. Restart it to pick up parakeet
+ the new pill UI:
`systemctl --user restart voice-dictation.service`

### Not committed/pushed (per handoff instruction — awaiting go-ahead)
- Staged: debug logging in lib.rs / openai_engine.rs / stream_engine.rs (inherited WIP)
- Unstaged: `slint-gui/ui/examples/style2-minimal.slint` (the componentized fix)
