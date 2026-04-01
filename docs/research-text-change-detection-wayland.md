# Technical Research Report: Detecting Text Changes in Arbitrary Application Windows on Wayland

**Date:** 2026-04-01
**Context:** Voice dictation tool for Wayland Linux (Rust). Investigating whether we can detect when a user corrects transcribed text in a target application, enabling a learning/feedback loop similar to Wispr Flow on macOS.

---

## Table of Contents

1. [Background: How macOS Does It](#1-background-how-macos-does-it)
2. [Wayland's Security Model](#2-waylands-security-model)
3. [Approach A: AT-SPI2 (Accessibility Framework)](#3-approach-a-at-spi2-accessibility-framework)
4. [Approach B: Input Method Framework](#4-approach-b-input-method-framework)
5. [Approach C: Clipboard / Primary Selection Monitoring](#5-approach-c-clipboard--primary-selection-monitoring)
6. [Approach D: Compositor-Specific APIs](#6-approach-d-compositor-specific-apis)
7. [Approach E: Screen Capture + OCR](#7-approach-e-screen-capture--ocr)
8. [Approach F: Alternative UX Designs](#8-approach-f-alternative-ux-designs)
9. [Feasibility Matrix](#9-feasibility-matrix)
10. [Recommendation](#10-recommendation)

---

## 1. Background: How macOS Does It

On macOS, Wispr Flow uses the **macOS Accessibility API** (AXObserver + AXValueChanged notifications). The macOS accessibility framework provides:

- `kAXValueChangedNotification` -- fires whenever text content in a focused field changes.
- `kAXSelectedTextChangedNotification` -- fires when text selection changes.
- `AXUIElementCopyAttributeValue` -- reads the current text value of any accessible text field.

This works because macOS requires all apps to expose their UI through the accessibility framework (it is deeply integrated into Cocoa/AppKit), and any app with Accessibility permission can observe changes system-wide. This is a mature, well-tested, OS-level capability.

**The core question: does Linux have an equivalent?** The answer is: partially, via AT-SPI2, but with significant caveats.

---

## 2. Wayland's Security Model

Wayland was designed to be fundamentally more secure than X11:

- **No shared buffers between clients.** Applications cannot read pixels from other windows.
- **No global input snooping.** Applications cannot intercept keystrokes destined for other windows.
- **No cross-client window introspection.** There is no equivalent to X11's `XGetWindowProperty` or `XQueryTree`.
- **Screen capture requires portal consent.** The `xdg-desktop-portal` service mediates screen sharing with explicit user permission per-session.

**Implication:** There is no Wayland protocol for reading text content from another application's window. The compositor intentionally does not expose this capability. Any solution must go through a side-channel: accessibility (AT-SPI2), input method protocol, or clipboard.

---

## 3. Approach A: AT-SPI2 (Accessibility Framework)

### 3.1 What Is AT-SPI2?

AT-SPI2 (Assistive Technology Service Provider Interface, version 2) is the Linux desktop accessibility framework. It runs over D-Bus and provides a way for assistive technology (AT) clients -- like screen readers -- to:

- Enumerate all accessible objects (the "accessibility tree") across all running applications
- Read text content from text fields
- Receive event notifications when UI state changes
- Send actions to UI elements (click, focus, edit text)

**This is the closest Linux equivalent to the macOS Accessibility API.**

### 3.2 The Text Interface

AT-SPI2 defines an `org.a11y.atspi.Text` interface that exposes:

| Method | Description |
|--------|-------------|
| `get_text(start, end)` | Read a range of text from a widget |
| `get_character_count()` | Total character count |
| `get_caret_offset()` | Current cursor position |
| `set_caret_offset()` | Move the cursor |
| `get_selection()` / `set_selection()` | Read/modify text selection |
| `get_string_at_offset()` | Get text at a given position with granularity (char, word, line, sentence) |
| `get_text_attributes()` | Read formatting attributes |

There is also an `org.a11y.atspi.EditableText` interface for text fields that support editing.

### 3.3 Text Change Events

AT-SPI2 fires the following events relevant to text change detection:

| Event | Data Carried |
|-------|-------------|
| `object:text-changed:insert` | start_pos (Unicode index), length, inserted text string |
| `object:text-changed:delete` | start_pos (Unicode index), length, deleted text string |
| `object:text-caret-moved` | new caret position (integer) |
| `object:text-selection-changed` | (no additional data) |
| `object:text-attributes-changed` | (no additional data) |

**This is exactly what we need.** The `text-changed:insert` and `text-changed:delete` events provide the position, length, and actual text of every insertion and deletion -- essentially a stream of diffs.

### 3.4 How to Listen (Rust)

The `atspi` Rust crate (from the Odilia screen reader project) provides a pure-Rust, async, zbus-based implementation:

```
# Cargo.toml
atspi = { version = "0.29", features = ["connection", "proxies", "tokio"] }
```

Key types:
- `AccessibilityConnection` -- connects to the AT-SPI2 bus
- `TextChangedEvent` -- struct with fields: `item` (ObjectRef), `operation` (Insert/Delete), `start_pos` (i32), `length` (i32), `text` (String)
- Events are received via an async event stream

The D-Bus interface is `org.a11y.atspi.Event.Object`, member `TextChanged`, match rule `object:text-changed`.

**Important:** Our project already uses `zbus = "5"` in `dictation-engine/Cargo.toml`, so the `atspi` crate (which also uses zbus) integrates cleanly.

### 3.5 Toolkit Support

| Toolkit | AT-SPI2 Support | Text Events | Notes |
|---------|----------------|-------------|-------|
| **GTK4** | Direct D-Bus implementation | Yes | Best support. Sources in `gtk/a11y`. |
| **GTK3** | Via ATK bridge | Yes | Mature, well-tested. |
| **Qt5** | Direct D-Bus implementation | Yes | Uses old Cache.GetItems signature but functional. |
| **Qt6** | Direct D-Bus implementation | Yes | Improved over Qt5. |
| **Chromium/Electron** | Hybrid (ATK + libatspi) | Yes, if enabled | Requires `--force-renderer-accessibility` or detection of screen reader. |
| **Firefox** | Via ATK bridge | Yes | Generally good support. |
| **LibreOffice** | Via ATK bridge | Yes | Good support for document editing. |
| **Java/Swing** | Via java-atk-wrapper | Partial | Intermediary layer, less reliable. |
| **Custom rendering** (games, etc.) | None | No | No accessibility tree exposed. |
| **Terminal emulators** | Varies | Partial | Some (GNOME Terminal, Kitty) expose text; many don't. |

### 3.6 Coverage Assessment

- **GTK and Qt apps** (the vast majority of native Linux desktop apps): ~85-90% coverage with good text events.
- **Electron apps** (VS Code, Slack, Discord, etc.): Work, but may require `--force-renderer-accessibility` flag or detection of a running screen reader. Chromium auto-enables accessibility when it detects AT-SPI clients listening for events.
- **Firefox and web apps**: Good support via ATK bridge.
- **Terminal emulators**: Inconsistent. Some expose the buffer as accessible text, some don't.
- **Apps with custom rendering** (e.g., games, some media players): No coverage.

**Estimated overall coverage: 70-85% of typical desktop workflows.**

### 3.7 Security and Permissions

The AT-SPI2 accessibility bus has a **fully permissive security model**:

```xml
<!-- From at-spi2-core/bus/accessibility.conf -->
<allow user="*"/>
<allow send_destination="*" eavesdrop="true"/>
<allow eavesdrop="true"/>
<allow own="*"/>
```

**Any process running as the current user can connect to the accessibility bus and listen to all events from all applications.** There is no permission prompt, no opt-in, no access control. This is both a feature (for our use case) and a known security concern (it means a malicious process could keylog via AT-SPI2).

This means: **no special permissions or user configuration needed** to listen for text change events, beyond the AT-SPI2 bus being active (it is by default on GNOME and most desktop environments).

### 3.8 Limitations

1. **Electron apps may not auto-enable accessibility.** Chromium-based apps lazy-load their accessibility tree. They check for AT-SPI clients at startup -- if Orca or another AT is already running, they enable it. If our tool starts after an Electron app, the app may not expose its tree. Workaround: set the environment variable `ACCESSIBILITY_ENABLED=1` or run Electron with `--force-renderer-accessibility`.

2. **Performance overhead.** When accessibility is enabled in Chromium/Electron, it builds and maintains the full accessibility tree, which can increase memory usage and slightly reduce performance.

3. **Not all text fields emit events reliably.** Some custom widgets or canvas-based UIs (e.g., Google Docs in a browser) may not fire `text-changed` events because the text is rendered in a canvas element, not a standard text input.

4. **Terminal emulators** have inconsistent support.

5. **AT-SPI2 is D-Bus-based, not Wayland-native.** It works alongside Wayland, not through it. This means it works the same whether you're on X11 or Wayland -- the Wayland security model neither helps nor hinders it.

6. **Future architecture changes.** The GNOME "Newton" project proposes a next-generation accessibility architecture that would be Wayland-native, push-based, and use AccessKit. The current plan is for clients to "infer events from changes to the tree, by comparing the old and new versions of updated nodes." This is still in early stages and would not break the text-change detection concept -- it might actually improve it.

### 3.9 Proof of Concept (Python, for quick validation)

```python
import pyatspi

def on_text_changed(event):
    print(f"[{event.type}] app={event.host_application.name}")
    print(f"  source: {event.source.name} (role={event.source.getRoleName()})")
    print(f"  position: {event.detail1}, length: {event.detail2}")
    print(f"  text: '{event.any_data}'")

pyatspi.Registry.registerEventListener(on_text_changed, "object:text-changed:insert")
pyatspi.Registry.registerEventListener(on_text_changed, "object:text-changed:delete")
pyatspi.Registry.start()
```

This will print every text insertion and deletion across all accessible applications.

---

## 4. Approach B: Input Method Framework

### 4.1 Wayland Text Input Protocol (zwp_text_input_v3)

The `zwp_text_input_v3` protocol mediates communication between applications and the compositor for input method purposes.

**Information flow from application to compositor:**
- `set_surrounding_text(text, cursor, anchor)` -- the application sends up to 4000 bytes of text around the cursor
- `set_text_change_cause(cause)` -- indicates why surrounding text changed (from IME or other reasons)
- `set_content_type(hint, purpose)` -- describes the type of text field
- `set_cursor_rectangle(x, y, width, height)` -- cursor position for IME popup placement

**Information flow from compositor to application:**
- `commit_string(text)` -- finalized text to insert
- `preedit_string(text, cursor_begin, cursor_end)` -- pre-edit composition text
- `delete_surrounding_text(before, after)` -- request to delete text around cursor

### 4.2 Input Method Protocol (zwp_input_method_v2)

If we implement a custom input method, the compositor sends us:
- `surrounding_text(text, cursor, anchor)` -- the text around the cursor in the focused field
- `text_change_cause(cause)` -- why the text changed
- `content_type(hint, purpose)` -- field description

### 4.3 Can We Use This?

**Theoretically:** If we implemented a custom IME that sits between the user and the application, we could:
1. Receive `surrounding_text` events from the application
2. Compare successive `surrounding_text` values to detect edits
3. Use `text_change_cause` to determine if the change was from the IME or from user editing

**Practically, this approach has severe limitations:**

1. **Only one IME can be active at a time.** If the user has fcitx5 or ibus configured, our tool would need to either replace it (breaking their existing input method) or integrate as a module within it.

2. **`surrounding_text` is limited to 4000 bytes** and only represents a window around the cursor, not the full text field content.

3. **Not all applications implement `text_input_v3`.** Many GTK3 apps and some older Qt apps don't send surrounding text. X11 apps running under XWayland definitely don't.

4. **`surrounding_text` is sent at the application's discretion.** The application must proactively call `set_surrounding_text` when text changes. Not all apps do this reliably.

5. **We would need to be the active input method.** The `surrounding_text` event is only sent to the currently bound input method, not to arbitrary clients.

### 4.4 Verdict

The IME approach could work as a supplementary signal (we could build a fcitx5 module), but it cannot be the primary mechanism for correction detection because:
- It requires the user to use our IME
- Coverage depends on per-app protocol support
- Only gives us a 4000-byte window, not full context

---

## 5. Approach C: Clipboard / Primary Selection Monitoring

### 5.1 Primary Selection

On Linux, selecting text with the mouse automatically copies it to the "primary selection" (middle-click paste). We can monitor this using the `wp_primary_selection` protocol or via `wl-paste --primary --watch`.

**Use case for correction detection:** If the user:
1. Selects the incorrectly transcribed text (primary selection now contains the old text)
2. Types the correction (replacing the selection)

We could detect step 1 by monitoring primary selection changes. However:
- We only get the selected text, not what it was replaced with.
- Many corrections are done without selecting first (e.g., backspace and retype).
- Not all compositors support primary selection monitoring for non-focused clients.

### 5.2 Clipboard (ext-data-control-v1)

The `ext-data-control-v1` protocol (replacing the deprecated `wlr-data-control-v1`) allows a privileged client to:
- Monitor clipboard changes (the `selection` event fires when a new selection is set)
- Monitor primary selection changes (the `primary_selection` event)
- Read clipboard content via data offers

Tools like `wl-paste --watch` and `cliphist` use this protocol.

**Limitations for correction detection:**
- Only captures what the user explicitly copies, not in-place edits.
- Cannot correlate clipboard content with specific text fields or corrections.
- Missing the "before" and "after" context needed to build a correction pair.

### 5.3 Verdict

Clipboard monitoring is useful as a **supplementary signal** (e.g., detect copy-paste correction patterns) but cannot serve as the primary mechanism.

---

## 6. Approach D: Compositor-Specific APIs

### 6.1 GNOME (Mutter)

- Mutter 48 added the `a11y-manager` D-Bus interface for AT-SPI keyboard event forwarding.
- No text introspection protocol beyond AT-SPI2.
- GNOME's roadmap is the "Newton" project (next-gen accessibility), which would be Wayland-native but is years from completion.

### 6.2 KDE (KWin)

- KWin 6.4 adopted the same keyboard accessibility interface as Mutter.
- No additional text introspection APIs.
- KDE's accessibility support relies entirely on Qt's AT-SPI2 bridge.

### 6.3 Hyprland

- No accessibility-specific protocols.
- `hyprctl` provides window metadata (class, title, address) but no text content.
- The project currently uses `hyprctl activewindow -j` for window targeting.

### 6.4 Sway

- wlroots-based, supports `wlr-data-control` for clipboard.
- No text introspection protocols.

### 6.5 COSMIC (System76)

- Implementing its own `cosmic-atspi-unstable-v1` Wayland protocol.
- Currently focused on keyboard grabs for screen readers (similar to GNOME's a11y-manager).
- Does not provide text introspection beyond standard AT-SPI2.

### 6.6 Verdict

No compositor provides text introspection beyond what AT-SPI2 already offers. The compositor-specific APIs are focused on keyboard capture for screen readers, not text content reading.

---

## 7. Approach E: Screen Capture + OCR

**Concept:** Use `xdg-desktop-portal` to capture the focused window, then OCR the text area.

**Assessment:**
- Requires user consent dialog for screen capture.
- Computationally expensive (continuous OCR).
- Low accuracy for detecting specific character-level edits.
- Cannot correlate OCR output with specific text fields.
- No reliable way to diff OCR output frame-to-frame.
- Massive performance overhead.

**Verdict:** Not viable for correction detection. This is an approach of last resort for any text-reading use case.

---

## 8. Approach F: Alternative UX Designs

Given that AT-SPI2 provides good-but-not-universal coverage, here are practical alternative and complementary UX approaches:

### 8.1 "Correction Mode" Hotkey

The user presses a hotkey (e.g., `Super+C`) after correcting text. The tool then:
1. Uses AT-SPI2 to read the current text from the focused widget
2. Compares it to what was originally dictated
3. Extracts the diff as a correction pair

**Advantages:** Works even if we missed the text-changed event. User intent is explicit.
**Disadvantage:** Requires user action.

### 8.2 Time-Window Monitoring

After injecting dictated text:
1. Start monitoring AT-SPI2 `text-changed` events for the target widget
2. Watch for a configurable time window (e.g., 30 seconds)
3. Any edits within that window to the region where we injected text are treated as corrections

**Advantages:** Fully automatic when AT-SPI2 works. Precise diffs.
**Disadvantage:** Limited to AT-SPI2-compatible apps.

### 8.3 Dedicated Correction UI

Present a small popup or panel showing the last N dictated phrases. The user can:
- Click a phrase to edit it
- The correction is captured directly in our UI

**Advantages:** 100% reliable, no dependency on target app. Works everywhere.
**Disadvantage:** Extra UI step, breaks flow.

### 8.4 Voice-Based Correction

"Change X to Y" voice commands (like Wispr Flow's approach):
- User says: "change cash to cache"
- We find-and-replace in the text we injected and apply the correction via keyboard injection
- The correction pair is logged

**Advantages:** Hands-free, no target app dependency.
**Disadvantage:** Requires NLU for command parsing. May conflict with dictation.

### 8.5 Clipboard-Based Correction Submission

User selects the corrected text, presses a hotkey. The tool:
1. Reads primary selection (the corrected text)
2. Matches it against recent dictation output
3. Computes the diff as a correction

**Advantages:** Simple, works with all apps (if they support primary selection).
**Disadvantage:** Requires user action, only captures the "after" not the "before."

### 8.6 Hybrid: AT-SPI2 + Hotkey Fallback

- **Primary:** AT-SPI2 time-window monitoring (automatic, best UX)
- **Fallback:** Correction hotkey for apps where AT-SPI2 doesn't work
- **Always available:** Voice correction commands and dedicated correction UI

---

## 9. Feasibility Matrix

| Approach | Coverage | Reliability | Permissions | Complexity | User Effort |
|----------|----------|-------------|-------------|------------|-------------|
| **AT-SPI2 events** | 70-85% | High (where supported) | None (open bus) | Medium | None (automatic) |
| **IME surrounding text** | 30-50% | Medium | Must be active IME | High | Config required |
| **Clipboard monitoring** | Supplementary only | Low for corrections | ext-data-control | Low | None |
| **Compositor APIs** | 0% (none exist) | N/A | N/A | N/A | N/A |
| **Screen capture + OCR** | ~95% visual | Very low for edits | User consent dialog | Very high | Consent per-session |
| **Correction hotkey** | 100% (user-driven) | High | None | Low | Per-correction |
| **Time-window + AT-SPI2** | 70-85% | High | None | Medium | None |
| **Voice correction** | 100% | Medium (NLU dependent) | None | Medium-High | Voice command |
| **Dedicated correction UI** | 100% | High | None | Medium | Per-correction |

---

## 10. Recommendation

### Primary Strategy: AT-SPI2 Time-Window Monitoring + Hybrid Fallbacks

**Phase 1: AT-SPI2 Integration (Recommended first implementation)**

1. Add the `atspi` crate to `dictation-engine/Cargo.toml` (compatible -- already uses zbus 5).
2. After injecting text via wtype, immediately:
   a. Identify the target widget via AT-SPI2 (correlate with `hyprctl activewindow` class).
   b. Register for `object:text-changed:insert` and `object:text-changed:delete` events filtered to that widget.
   c. Monitor for a 30-60 second window.
3. Any edits detected in the injected text region are captured as correction pairs (original transcription -> user's correction).
4. Store corrections in the existing user dictionary / learning library.

**Phase 2: Fallback Mechanisms**

5. Add a "correction mode" hotkey that uses AT-SPI2 to snapshot the current text field and diff against the last dictation.
6. Add voice correction commands ("change X to Y") -- these are compositor-independent and work everywhere.
7. Add a correction review panel in the existing Slint GUI.

**Phase 3: Enhanced Coverage**

8. Optionally implement a fcitx5 module that captures `surrounding_text` as a supplementary signal.
9. Set `ACCESSIBILITY_ENABLED=1` in the environment or document how users can enable it for Electron apps.

### Why AT-SPI2 Is the Right Primary Approach

1. **It provides exactly the right data**: insert/delete events with position, length, and text content -- the same primitives macOS provides via AXValueChanged.
2. **No permissions needed**: The accessibility bus is open. No user consent dialogs.
3. **Our project already uses zbus**: The `atspi` crate integrates with zero friction.
4. **It works on Wayland and X11**: AT-SPI2 runs over D-Bus independent of the display server.
5. **70-85% coverage of typical workflows**: GTK, Qt, Chromium/Electron (when accessibility enabled), Firefox, LibreOffice.
6. **Active ecosystem**: The Odilia screen reader project maintains the Rust crate. GNOME is investing in the next-gen Newton architecture.

### What We Cannot Achieve

Full parity with macOS is not possible today because:
- macOS has mandatory accessibility support in all AppKit/UIKit apps (close to 100% coverage).
- Linux AT-SPI2 is opt-in at the toolkit level and incomplete for custom-rendered UIs.
- Canvas-based web apps (Google Docs) and some terminal emulators will be blind spots.

The hybrid approach (AT-SPI2 automatic detection + hotkey/voice fallback) is the best practical solution and covers the large majority of real-world use cases.

---

## Sources

- [AT-SPI2 - freedesktop.org](https://www.freedesktop.org/wiki/Accessibility/AT-SPI2/)
- [AT-SPI Event Table - Linux Foundation](https://accessibility.linuxfoundation.org/a11yspecs/atspi/adoc/atspi-events.html)
- [TextChangedEvent in atspi Rust crate](https://docs.rs/atspi/latest/atspi/events/object/struct.TextChangedEvent.html)
- [atspi crate - crates.io](https://crates.io/crates/atspi)
- [Odilia screen reader (atspi crate authors)](https://github.com/odilia-app/atspi)
- [AT-SPI2 Text Interface](https://gnome.pages.gitlab.gnome.org/at-spi2-core/libatspi/iface.Text.html)
- [AT-SPI2 Toolkit Implementations](https://gnome.pages.gitlab.gnome.org/at-spi2-core/devel-docs/toolkits.html)
- [Next-gen Accessibility Architecture (Newton)](https://gnome.pages.gitlab.gnome.org/at-spi2-core/devel-docs/new-protocol.html)
- [AT-SPI2 Architecture](https://gnome.pages.gitlab.gnome.org/at-spi2-core/devel-docs/architecture.html)
- [Wayland Accessibility Notes](https://github.com/splondike/wayland-accessibility-notes)
- [zwp_text_input_v3 Protocol](https://wayland.app/protocols/text-input-unstable-v3)
- [zwp_input_method_v2 Protocol](https://wayland.app/protocols/input-method-unstable-v2)
- [ext-data-control-v1 Protocol](https://wayland.app/protocols/ext-data-control-v1)
- [COSMIC atspi Protocol](https://wayland.app/protocols/cosmic-atspi-unstable-v1)
- [Enhancing Screen-Reader Functionality in GNOME - LWN.net](https://lwn.net/Articles/1025127/)
- [Accessibility in Wayland - LWN.net](https://lwn.net/Articles/980811/)
- [Chromium Linux Accessibility](https://www.chromium.org/developers/accessibility/linux-accessibility/)
- [AccessKit](https://accesskit.dev/)
- [PyAtSpi2 Examples](https://lazka.github.io/pgi-docs/Atspi-2.0/classes/EventListener.html)
- [Atspi.EventListener.register](https://docs.gtk.org/atspi2/method.EventListener.register.html)
- [AT-SPI2 D-Bus accessibility.conf](https://github.com/tbsaunde/at-spi2-core/blob/master/bus/accessibility.conf)
- [Fcitx5 Wayland Input Method](https://fcitx-im.org/wiki/Using_Fcitx_5_on_Wayland)
- [Wayland Primary Selection Protocol](https://wayland.app/protocols/primary-selection-unstable-v1)
- [Hyprland Clipboard Managers](https://wiki.hypr.land/Useful-Utilities/Clipboard-Managers/)
