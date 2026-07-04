#!/usr/bin/env bash
set -euo pipefail

# repo root is three levels up: correction-engine/tests/e2e -> repo. The build
# context must be the repo root so the Containerfile's `COPY . .` lands the
# project at the paths the entrypoint expects (./correction-engine/...).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

podman build -t correction-engine-e2e -f "$SCRIPT_DIR/Containerfile" "$PROJECT_ROOT"
podman run --rm --security-opt label=disable --tmpfs /tmp correction-engine-e2e
