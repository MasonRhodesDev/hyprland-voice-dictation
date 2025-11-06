#!/bin/bash

cd "$(dirname "$0")"

if command -v cargo-watch &> /dev/null; then
    echo "Using cargo-watch for hot reload..."
    exec cargo watch -x 'build -p monitor-label-poc' -s 'pkill -9 monitor-label-poc; ../target/debug/monitor-label-poc'
elif command -v inotifywait &> /dev/null; then
    echo "Using inotifywait for hot reload..."

    PID=""

    cleanup() {
        if [ ! -z "$PID" ]; then
            kill $PID 2>/dev/null
            wait $PID 2>/dev/null
        fi
        exit 0
    }

    trap cleanup SIGINT SIGTERM

    build_and_run() {
        if [ ! -z "$PID" ]; then
            kill $PID 2>/dev/null
            wait $PID 2>/dev/null
        fi

        cargo build -p monitor-label-poc 2>&1 | grep -E "Compiling|Finished|error|warning"

        if [ $? -eq 0 ]; then
            ../target/debug/monitor-label-poc &
            PID=$!
        fi
    }

    build_and_run

    inotifywait -m -r -e modify,create,delete --exclude '.*target.*' src/ Cargo.toml 2>/dev/null | while read -r directory events filename; do
        echo "Change detected: $filename"
        sleep 0.1
        build_and_run
    done
else
    echo "Error: Neither cargo-watch nor inotifywait found"
    echo ""
    echo "Install one of the following:"
    echo "  cargo install cargo-watch"
    echo "  sudo dnf install inotify-tools"
    exit 1
fi
