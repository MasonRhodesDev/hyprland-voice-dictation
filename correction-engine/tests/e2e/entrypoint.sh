#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    [ -n "${WESTON_PID:-}" ] && kill "$WESTON_PID" 2>/dev/null || true
    [ -n "${REGISTRYD_PID:-}" ] && kill "$REGISTRYD_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Start D-Bus session bus
eval "$(dbus-launch --sh-syntax)"
export DBUS_SESSION_BUS_ADDRESS

# Start AT-SPI2 registryd
/usr/libexec/at-spi2-registryd &
REGISTRYD_PID=$!
sleep 1

# Start Weston with headless backend
weston --backend=headless-backend.so --no-config &
WESTON_PID=$!
sleep 2

export WAYLAND_DISPLAY=wayland-1

# Run the e2e tests
cargo test --package correction-engine --test e2e_tests -- --nocapture
TEST_EXIT=$?

exit "$TEST_EXIT"
