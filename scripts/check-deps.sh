#!/bin/bash

MISSING=()
OPTIONAL_MISSING=()

echo "Checking system dependencies..."
echo ""

command -v cargo &> /dev/null || MISSING+=("cargo")

for cmd in wtype pactl pkg-config python3; do
    command -v $cmd &> /dev/null || MISSING+=("$cmd")
done

# Optional build tools
command -v rpmbuild &> /dev/null || OPTIONAL_MISSING+=("rpmbuild (rpm-build)")

for lib in alsa fontconfig freetype2; do
    pkg-config --exists $lib 2>/dev/null || MISSING+=("$lib-devel")
done

if [ ! -d "models/vosk-model-small-en-us-0.15" ] || [ ! -d "models/vosk-model-en-us-0.22" ]; then
    echo "⚠️  Vosk models not found. Run: make deps"
    echo ""
fi

if [ ${#OPTIONAL_MISSING[@]} -gt 0 ]; then
    echo "⚠️  Optional dependencies missing: ${OPTIONAL_MISSING[*]}"
    echo "   These are only needed for building RPM packages"
    echo ""
fi

if [ ${#MISSING[@]} -eq 0 ]; then
    echo "✓ All required dependencies installed"
    exit 0
else
    echo "✗ Missing required dependencies: ${MISSING[*]}"
    echo ""
    echo "Fedora: sudo dnf install rust cargo wtype pipewire alsa-lib-devel fontconfig-devel freetype-devel python3"
    echo "Arch:   sudo pacman -S rust cargo wtype pipewire alsa-lib fontconfig freetype2 python"
    echo ""
    echo "For RPM packaging, also install: rpm-build"
    exit 1
fi
