# Error Handling

## Patterns

This project uses two error handling approaches depending on context:

**`anyhow::Result<T>`** — in `dictation-engine` for internal library code where rich error context chains are useful.

**`Box<dyn std::error::Error>`** — in `src/main.rs` for CLI command functions where simple propagation is sufficient.

## Adding Context

When propagating errors, add context that helps the user understand what went wrong:

```rust
// Engine code (anyhow)
let model = Model::new(path)
    .ok_or_else(|| anyhow::anyhow!("Failed to load Vosk model from {}", path))?;

// CLI code (Box<dyn Error>)
send_confirm()
    .map_err(|e| format!("Failed to confirm recording: {}", e))?;
```

The `dbus_error_with_hint()` helper in `main.rs` wraps D-Bus errors with actionable recovery guidance:

```
Failed to communicate with daemon: <error>
Try: systemctl --user status voice-dictation
```

## User-Facing Error Messages

CLI errors should follow this pattern:
1. `eprintln!("Error: <what failed>")` — concise description
2. `eprintln!("<how to fix>")` — actionable recovery hint
3. `return Err("short reason".into())` — short reason for programmatic use

Example from `start_recording()`:
```rust
if !is_daemon_running() {
    eprintln!("Error: Daemon not running");
    eprintln!("Start the daemon with: systemctl --user start voice-dictation");
    eprintln!("Or run manually: voice-dictation daemon");
    return Err("Daemon not running".into());
}
```

## Contributor Guidelines

- Never silently swallow errors with `let _ = ...` unless the failure is truly inconsequential (e.g., cleanup that can't be retried).
- Use `?` for propagation in functions that return `Result`.
- Add `.map_err(|e| ...)` at system boundaries (D-Bus calls, file I/O, subprocess execution) to provide context.
- For `anyhow` errors, prefer `.context("description")` or `.with_context(|| format!(...))` over manual `map_err`.
- Lock poisoning errors (`Mutex::lock().map_err(...)`) should always be handled explicitly — a poisoned lock indicates a panic in another thread.
