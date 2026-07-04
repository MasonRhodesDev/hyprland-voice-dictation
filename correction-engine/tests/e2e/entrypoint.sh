#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
    [ -n "${APP_PID:-}" ] && kill "$APP_PID" 2>/dev/null || true
    [ -n "${WESTON_PID:-}" ] && kill "$WESTON_PID" 2>/dev/null || true
    [ -n "${REGISTRYD_PID:-}" ] && kill "$REGISTRYD_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Fresh marker state for a race-free handshake with the Rust test.
rm -f /tmp/e2e-subscribed /tmp/e2e-ready /tmp/e2e-do-edit /tmp/e2e-edit-done /tmp/test-app.pid

# Wayland/weston needs XDG_RUNTIME_DIR for its socket; it is not set in the
# minimal container environment. Create a private one.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Start D-Bus session bus (dbus-launch is provided by dbus-x11).
eval "$(dbus-launch --sh-syntax)"
export DBUS_SESSION_BUS_ADDRESS

# Start AT-SPI2 registryd on that bus.
/usr/libexec/at-spi2-registryd &
REGISTRYD_PID=$!
sleep 1

# Start Weston with the headless backend so GTK4 has a compositor to map into.
export WAYLAND_DISPLAY=wayland-1
weston --backend=headless-backend.so --no-config --socket="$WAYLAND_DISPLAY" &
WESTON_PID=$!

# Wait for weston's Wayland socket to actually appear before launching GTK4.
for _ in $(seq 1 50); do
    [ -S "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ] && break
    sleep 0.2
done
if [ ! -S "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ]; then
    echo "FATAL: weston Wayland socket never appeared at $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" >&2
    exit 1
fi

# Ensure GTK4 emits AT-SPI2 accessibility events.
export GTK_A11Y=atspi
unset NO_AT_BRIDGE || true

# Launch the GTK4 test app in the background. It maps a window, sets the
# initial text, writes /tmp/e2e-ready, then waits for /tmp/e2e-do-edit before
# performing the in-place cash->cache edit.
python3 "$SCRIPT_DIR/test_app.py" &
APP_PID=$!

# Tell the Rust test it is running inside the real e2e container, so it runs
# the live AT-SPI2 assertions instead of skipping.
export CORRECTION_E2E=1

# Single-threaded so the marker-file handshake is deterministic. Capture the
# exit code without tripping `set -e` so cleanup still runs and we exit with it.
TEST_EXIT=0
cargo test --package correction-engine --test e2e_tests -- --nocapture --test-threads=1 \
    || TEST_EXIT=$?

exit "$TEST_EXIT"
