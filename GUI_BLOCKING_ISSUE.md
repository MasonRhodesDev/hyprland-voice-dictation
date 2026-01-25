# GUI Blocking Issue

## Status
The GUI Shell creation blocks indefinitely on mason-desktop but works on mason-work.

## Problem
`layer_shika::Shell::from_file().build()` in `slint-gui/src/lib.rs:317` blocks forever and never returns.

## Evidence
1. **Logs show**: "Creating Shell from UI file..." but never "Shell created successfully"
2. **No Wayland protocol activity**: `WAYLAND_DEBUG=1` shows zero `wl_*` messages
3. **Blocking happens before Wayland connection**: The block occurs in layer-shika/slint initialization, not Wayland protocol
4. **Daemon continues anyway**: Due to design flaw where "Ready" signal is sent before Shell creation

## What Works
- All dependencies are installed correctly ✅
- Wayland environment variables are set ✅
- Binary has Wayland symbols compiled in ✅
- Slint version matches layer-shika requirements (1.14) ✅
- Works perfectly on mason-work machine ✅

## Investigation Needed
Since it works on mason-work, compare:

```bash
# On mason-work, run:
cd ~/repos/wayland-voice-dictation

# Check exact dependency versions
grep -A 3 '"layer-shika"' Cargo.lock
grep -A 3 '"slint"' Cargo.lock | head -10

# Check Hyprland version
hyprland version

# Check system libraries
pacman -Q | grep -E "(wayland|slint|hypr)"

# Check if there are any Hyprland layer/overlay configs
grep -i layer ~/.config/hypr/hyprland.conf 2>/dev/null || echo "No layer config"
```

## Possible Causes
1. Different layer-shika/slint versions between machines
2. Different Hyprland versions or configurations
3. Different system library versions (wayland-protocols, etc.)
4. Missing or incompatible runtime library on mason-desktop

## Design Issue to Fix
The GUI thread sends `GuiStatus::Ready` before creating the Shell (line 160 in slint-gui/src/lib.rs).
This should be moved to after successful Shell creation so the daemon knows if GUI initialization actually succeeded.

## Commits
- 6a226e2: Added comprehensive dependency checking
- 3a527ca: Fixed Wayland environment in systemd service
- 6be7a94: Fixed slint version mismatch (1.9 → 1.14)
- 9bf160f: Added detailed logging to track blocking point
