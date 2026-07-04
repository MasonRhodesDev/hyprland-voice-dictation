# zwp_input_method_v2 correction backend — live validation findings

Date: 2026-07-03
Machine: live Hyprland session — Hyprland 0.55.0 (main @ 0aa7a843), Arch, wl_seat v9,
zenity 4.2.2 (GTK4), Chromium (system, `--enable-wayland-ime` via chromium-flags.conf),
Discord official tarball 1.0.145, wezterm, Claude Desktop.
IME slot free at test time (no fcitx5/ibus; `XMODIFIERS`/`GTK_IM_MODULE`/`QT_IM_MODULE` empty).

Prototype: `voice-dictation debug ime-probe` (`dictation-engine/src/ime_probe.rs`), a
time-boxed diagnostic that binds `zwp_input_method_manager_v2`, never grabs the keyboard,
and logs every IME event tagged with the focused window class from `hyprctl activewindow -j`.
Write-path testing (`--commit`) is hard-gated to an exact window class and was only ever
pointed at throwaway targets (`zenity --entry`, and a separate Chromium instance launched
with its own `--user-data-dir` and a distinct `--class=ime-probe-chromium` app id).

## Executive summary

| Question | Result |
| --- | --- |
| Hyprland accepts a second... any IME client binding | Yes — bind succeeded, `activate`/`done` flow immediately; `unavailable` never seen (slot was free) |
| Read path (`surrounding_text`) — GTK (zenity) | **Works.** Full text + cursor/anchor on every change, ~10 ms latency |
| Read path — Chromium (`--enable-wayland-ime`) | **Broken.** `activate`, `text_change_cause`, `content_type` arrive; `surrounding_text` is never sent, even with a seeded non-empty textarea |
| Read path — Discord (official tarball) | **Nothing.** Never activates — root cause: `--enable-wayland-ime` is NOT actually applied (see below) |
| Read path — wezterm | Activates and sends `surrounding_text`, but content is a placeholder (`"\u{FEFF}\n"`), not the terminal text |
| Read path — Claude Desktop | Nothing (expected negative control; bundled Electron without wayland-ime) |
| Write path (`commit_string`) — zenity | **Works**, protocol-confirmed round trip: committed text came back in the next `surrounding_text` 9 ms later |
| Write path — Chromium | **Works**, visually confirmed: committed text appeared in the page textarea at the caret |
| `text_change_cause` user-vs-IME discrimination | **Not reliable** (GTK): user keystrokes are also reported as `InputMethod`; only the initial state carried `Other` |

Verdict: **GO, with a split-capability design** — input-method-v2 is an excellent, safe
*write* channel everywhere tested (including Chromium/Electron), and a *read* channel for
GTK only. Correction *detection* for Chromium/Electron cannot come from
`surrounding_text` today. Details under "Go/no-go".

## Probe validation runs (captured live)

### Run 1 — zenity round trip (read + write + change-cause)

`debug ime-probe --secs 25 --json --commit "IME probe commit OK " --commit-class zenity`,
zenity focused via `hyprctl dispatch focuswindow`, then three keystrokes injected with
`hyprctl dispatch sendshortcut` to simulate user edits.

```
[ime-probe] bound zwp_input_method_manager_v2 v1 on seat ObjectId(wl_seat@3); holding IME slot for 25s (no keyboard grab)
[ime-probe +  1648ms] class="zenity" text_change_cause=InputMethod
[ime-probe +  1648ms] class="zenity" done serial=1
[ime-probe +  1658ms] class="zenity" activate
[ime-probe +  1658ms] class="zenity" surrounding_text text="" cursor=0 anchor=0
[ime-probe +  1658ms] class="zenity" text_change_cause=Other
[ime-probe +  1658ms] class="zenity" content_type hint=ContentHint(0x0) purpose=Normal
[ime-probe +  1658ms] class="zenity" done serial=2
[ime-probe +  1658ms] class="zenity" -> commit_string("IME probe commit OK ") + commit
[ime-probe +  1667ms] class="zenity" surrounding_text text="IME probe commit OK " cursor=20 anchor=20
[ime-probe +  1667ms] class="zenity" text_change_cause=InputMethod
[ime-probe +  1667ms] class="zenity" done serial=3
[ime-probe +  7026ms] class="zenity" surrounding_text text="IME probe commit OK u" cursor=21 anchor=21
[ime-probe +  7026ms] class="zenity" text_change_cause=InputMethod
...
[ime-probe +  9065ms] class="zenity" surrounding_text text="IME probe commit OK usr" cursor=23 anchor=23
[ime-probe +  9065ms] class="zenity" text_change_cause=InputMethod
```

JSON summary (excerpt): `"commit_test": { "committed": true, "roundtrip_confirmed": true }`;
zenity: 1 activation, 5 surrounding_text events, causes `["InputMethod", "Other"]`.

Answers:

- **2a (zenity):** yes — `activate` + `surrounding_text` with cursor/anchor on focus, and a
  fresh `surrounding_text` after every text change (~1 s cadence corresponds to the injected
  keystrokes; latency of the commit read-back was 9 ms).
- **2c (commit round trip):** yes — `commit_string` + `commit(serial)` inserted text and the
  next `surrounding_text` reflected it exactly (`cursor=20` after a 20-char commit).
- **2b (change cause):** **no reliable discrimination.** The keystrokes injected into zenity
  (`u`, `s`, `r` — real key events delivered by the compositor, indistinguishable from user
  typing from GTK's perspective) came back as `text_change_cause=InputMethod`, not `Other`.
  GTK routes all keystrokes through its IM context, so edits caused by typing are flagged as
  IM-caused; `Other` appeared only for the initial (programmatic) state. A correction monitor
  must therefore diff surrounding_text snapshots against its own commits instead of trusting
  `text_change_cause`.

### Run 2 — app focus sweep (read-only, no commits)

`debug ime-probe --secs 55 --json`; focus walked through zenity → new Chromium window on a
local `file://` form with an autofocused `<input>` → Discord → wezterm → Claude Desktop.

```
[ime-probe +  1468ms] class="zenity" activate
[ime-probe +  1468ms] class="zenity" surrounding_text text="" cursor=0 anchor=0
[ime-probe +  7245ms] class="chromium" activate
[ime-probe +  7245ms] class="chromium" text_change_cause=InputMethod
[ime-probe +  7245ms] class="chromium" done serial=4          <- NO surrounding_text
[ime-probe + 18114ms] class="discord" deactivate              <- chromium defocus; Discord NEVER activates
[ime-probe + 24051ms] class="org.wezfurlong.wezterm" surrounding_text text="\u{feff}\n" cursor=0 anchor=0
[ime-probe + 24060ms] class="org.wezfurlong.wezterm" activate
[ime-probe + 28060ms] class="claude-desktop" deactivate       <- wezterm defocus; Claude Desktop never activates
```

- **Chromium:** activates (text-input-v3 enabled and working at the enter/leave level) but
  sends no `surrounding_text` and, in this run, no `content_type`.
- **Discord:** no `activate` at all on window focus. Root-caused below.
- **wezterm:** *does* activate and *does* send `surrounding_text` — but the content is
  `"\u{FEFF}\n"` (a BOM + newline placeholder), not the terminal screen text. Read path is
  formally present, semantically useless for corrections (terminals have no editable buffer
  to expose — expected).
- **Claude Desktop:** nothing, as predicted (bundled Electron, no wayland-ime flag).

### Run 3 — Chromium with seeded text (conclusive read-path test)

Throwaway Chromium instance (`--user-data-dir=<scratch>`, `--class=ime-probe-chromium`,
inherits `--enable-wayland-ime` from chromium-flags.conf) opening a local page whose
`<textarea autofocus>` already contained `"seed surrounding text from chromium"`:

```
[ime-probe +   986ms] class="ime-probe-chromium" activate
[ime-probe +   986ms] class="ime-probe-chromium" done serial=2   <- no surrounding_text despite non-empty textarea
[ime-probe +   990ms] class="ime-probe-chromium" content_type hint=ContentHint(AutoCapitalization) purpose=Normal
```

Summary: `surrounding_text_events: 0`, `content_types: ["ContentHint(AutoCapitalization) / Normal"]`.
This removes the run-2 ambiguity (empty field): **Chromium does not send
`zwp_text_input_v3.set_surrounding_text` at all**, even when the focused editor has content.
It does send `set_content_type` (note: only *after* the first `done`, in a separate batch —
another minor ordering quirk). This matches the long-standing upstream gap in Chromium's
Ozone/Wayland text-input implementation: enter/leave, preedit, and commit work under
`--enable-wayland-ime`, but surrounding-text reporting has historically been absent or
partial on the text-input-v3 path (tracked upstream in the Wayland IME bug cluster around
crbug "wayland text input surrounding text"; behavior may improve in newer builds).

### Run 4 — Chromium write path (visual confirmation)

Same throwaway instance, `--commit "CHROMIUM COMMIT OK " --commit-class ime-probe-chromium`:

```
[ime-probe +     4ms] class="ime-probe-chromium" activate
[ime-probe +     4ms] class="ime-probe-chromium" -> commit_string("CHROMIUM COMMIT OK ") + commit
```

A `grim` screenshot of the window taken 5 s later shows the textarea content:

> `CHROMIUM COMMIT OK seed surrounding text from chromium`

The committed string was inserted at the caret (position 0). **The write path into
Chromium/Electron works** — protocol read-back (`roundtrip_confirmed`) stays `false` only
because Chromium never sends surrounding_text back.

## Protocol-level observations / interop quirks

1. **Initial `done` before `activate`:** on bind, Hyprland sends a first state batch
   (`text_change_cause` + `done serial=1`) with no `activate`, even when a text input is
   already focused; the real `activate` batch follows ~10 ms later. Consumers must not
   assume the first `done` implies an active state.
2. **Serial handling:** `commit(serial)` with serial = count of received `done` events was
   accepted by Hyprland in all runs (commit applied both to GTK and Chromium).
3. **`text_change_cause` is not a user-vs-IME oracle** (see run 1). Correction detection
   must diff surrounding_text against the IME's own recent commits.
4. **Attribution lag:** the probe tags events with the *currently* focused window at
   `done`-time, so the `deactivate` batch for app A is tagged with app B's class when focus
   has already moved. A production monitor should latch the class at `activate` time and
   pair `deactivate` with the previous activation.
5. **Chromium sends `content_type` in a separate batch after activation**, and only
   sometimes (run 3 yes, run 2 no) — likely dependent on field attributes.
6. **`unavailable` handling is implemented but was not exercised live** — the slot was free
   the whole time (that was the precondition). The probe exits with an error and no retry
   if it ever arrives.
7. **Discord finding (environment, not protocol):** the running Discord is the official
   tarball at `~/.config/discord/Discord`, launched with *no* Chromium flags — Wayland mode
   comes only from `ELECTRON_OZONE_PLATFORM_HINT=wayland` in its environment.
   `~/.config/discord-flags.conf` is read by the *Arch package* launcher, which is not what
   runs. So `--enable-wayland-ime` is silently not applied, and Discord never enables
   text-input-v3. This is fixable at launch-configuration level (wrapper script passing the
   flag), after which Discord should behave like the Chromium column (write path only).

## Go/no-go recommendation

**GO — build the IME monitor, but design it as a split-capability backend:**

- **Write path (universal):** `commit_string` is a clean, compositor-mediated injection
  channel that worked against every text-input-v3 client tested, including Chromium. For
  in-place fixes in GTK apps, `delete_surrounding_text` + `commit_string` can perform true
  replacements because cursor/anchor and text are known. For Chromium/Electron the daemon
  knows neither text nor cursor, so in-place *replacement* is limited to text the daemon
  itself just committed (track our own committed suffix and delete relative to the caret —
  needs live verification of `delete_surrounding_text` in Chromium before relying on it).
- **Read path (GTK-grade apps only):** surrounding_text gives everything correction
  detection needs (full text, cursor, anchor, ~10 ms latency) — but today that covers GTK
  apps, not the Chromium/Electron targets this project cares most about. For
  Chromium/Electron, correction *detection* must keep using another source (the existing
  clipboard/heuristic approaches; AT-SPI was already refuted separately), or wait on
  upstream Chromium surrounding-text support.
- **Exclusivity cost:** input-method-v2 allows one IME per seat. Holding it permanently
  means the user can never run fcitx5/ibus concurrently, and any crash-restart race needs
  care (`unavailable` => back off, never retry-loop). Recommend the daemon acquires the
  slot lazily (during/after dictation windows) rather than always-on, which also matches
  the time-boxed model validated here.
- **No keyboard grab, ever:** typing was verifiably unaffected in all four runs; the
  monitor must keep `grab_keyboard` out of scope.

## Limitations of this validation

- Single compositor/version (Hyprland 0.55.0 main); protocol behavior (initial-done quirk,
  serial semantics) may differ on other versions.
- `delete_surrounding_text` (the actual in-place *replacement* primitive) was not exercised
  — only insertion via `commit_string`. Next prototype step.
- Discord/wezterm write paths untested (no committing into real apps, per safety rules).
- `unavailable` path untested live (slot was free; would need fcitx5 running to exercise).
- Preedit flows (set_preedit_string) untested — not needed for corrections.
- Build gates: `cargo fmt --check`, `clippy -D warnings`, and `cargo test --workspace` all
  pass, but clippy/test/build ran with `--no-default-features --features
  voice-dictation/tray,voice-dictation/correction` (pipewire feature off): a *pre-existing*
  environment breakage — PipeWire 1.6.6 headers define `PW_ID_ANY` as `(uint32_t)(0xffffffff)`,
  a cast expression bindgen cannot evaluate, so a clean build of `pipewire-sys 0.10.0` fails
  with `cannot find value PW_ID_ANY in crate pw_sys` — blocks any fresh full-feature build
  on this machine (the main checkout only builds thanks to cached bindings). Unrelated to
  this branch; needs a pipewire-rs update or vendored patch.

## Follow-ups

1. Prototype `delete_surrounding_text`-based replacement against zenity and the throwaway
   Chromium instance (validates true in-place correction, not just insertion).
2. Fix Discord launch config so `--enable-wayland-ime` is actually applied (wrapper around
   the tarball binary; `discord-flags.conf` does not reach it), then re-run the sweep.
3. Track Chromium upstream surrounding-text support; re-run run 3 after Chromium updates —
   if it lands, the read path unifies and this backend can replace clipboard heuristics.
4. Design the monitor's lazy-acquire lifecycle (bind after dictation commit, release after
   correction window closes) + `unavailable` backoff.
5. Decide the pipewire-sys fix (bump/patch) so full-feature CI builds work again.
