#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

podman build -t correction-engine-e2e -f "$SCRIPT_DIR/Containerfile" "$PROJECT_ROOT"
podman run --rm --security-opt label=disable --tmpfs /tmp correction-engine-e2e
