#!/bin/bash

MISSING=()
OPTIONAL_MISSING=()
WARNINGS=()

echo "=== Checking System Dependencies ==="
echo ""

# Detect distro
if [ -f /etc/fedora-release ]; then
    DISTRO="fedora"
elif [ -f /etc/arch-release ]; then
    DISTRO="arch"
else
    DISTRO="unknown"
    WARNINGS+=("Unknown distro - package names may differ")
fi

# Check Rust version
echo "1. Checking Rust/Cargo..."
if command -v cargo &> /dev/null; then
    RUST_VERSION=$(rustc --version | awk '{print $2}')
    REQUIRED_VERSION="1.70.0"
    if printf '%s\n%s\n' "$REQUIRED_VERSION" "$RUST_VERSION" | sort -V -C; then
        echo "   ✓ Rust $RUST_VERSION (>= $REQUIRED_VERSION required)"
    else
        MISSING+=("rust (version >= $REQUIRED_VERSION, found $RUST_VERSION)")
    fi
else
    MISSING+=("rust/cargo")
fi
echo ""

# Check runtime dependencies
echo "2. Checking runtime dependencies..."
for cmd in wtype pkg-config; do
    if command -v $cmd &> /dev/null; then
        echo "   ✓ $cmd"
    else
        MISSING+=("$cmd")
    fi
done

# Check for audio tools (at least one needed)
if command -v pactl &> /dev/null; then
    echo "   ✓ pactl (PulseAudio/PipeWire)"
elif command -v pw-cli &> /dev/null; then
    echo "   ✓ pw-cli (PipeWire)"
else
    WARNINGS+=("No audio tool found (pactl or pw-cli) - audio device enumeration may not work")
fi

# Check for Wayland compositor
if [ -n "$WAYLAND_DISPLAY" ]; then
    echo "   ✓ Wayland compositor running ($WAYLAND_DISPLAY)"
elif [ -n "$DISPLAY" ]; then
    WARNINGS+=("X11 detected - Wayland compositor required for this application")
else
    WARNINGS+=("No display server detected - Wayland compositor required")
fi
echo ""

# Check build-time library dependencies
echo "3. Checking build libraries..."
declare -A LIB_CHECKS=(
    ["alsa"]="ALSA (audio)"
    ["fontconfig"]="Fontconfig (fonts)"
    ["freetype2"]="FreeType (font rendering)"
    ["wayland-client"]="Wayland client library"
    ["wayland-cursor"]="Wayland cursor library"
    ["wayland-protocols"]="Wayland protocols"
    ["xkbcommon"]="xkbcommon (keyboard)"
    ["xcursor"]="Xcursor library"
    ["egl"]="EGL (OpenGL)"
)

for lib in "${!LIB_CHECKS[@]}"; do
    if pkg-config --exists "$lib" 2>/dev/null; then
        echo "   ✓ ${LIB_CHECKS[$lib]}"
    else
        MISSING+=("$lib")
    fi
done

# Check optional PipeWire library (for native backend)
if pkg-config --exists "libpipewire-0.3" 2>/dev/null; then
    echo "   ✓ PipeWire native library (optional)"
else
    OPTIONAL_MISSING+=("libpipewire-0.3 (native audio backend)")
fi
echo ""

# Check optional build tools
echo "4. Checking optional tools..."
if command -v rpmbuild &> /dev/null; then
    echo "   ✓ rpmbuild (RPM packaging)"
else
    OPTIONAL_MISSING+=("rpmbuild (for RPM packaging)")
fi
echo ""

# Summary
if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo "⚠️  Warnings:"
    for warning in "${WARNINGS[@]}"; do
        echo "   - $warning"
    done
    echo ""
fi

if [ ${#OPTIONAL_MISSING[@]} -gt 0 ]; then
    echo "ℹ️  Optional dependencies missing:"
    for opt in "${OPTIONAL_MISSING[@]}"; do
        echo "   - $opt"
    done
    echo ""
fi

if [ ${#MISSING[@]} -eq 0 ]; then
    echo "✅ All required dependencies installed!"
    exit 0
else
    echo "❌ Missing required dependencies:"
    for dep in "${MISSING[@]}"; do
        echo "   - $dep"
    done
    echo ""

    # Provide distro-specific install commands
    case "$DISTRO" in
        fedora)
            echo "Install on Fedora:"
            echo "  sudo dnf install rust cargo wtype pipewire pkg-config \\"
            echo "    alsa-lib-devel fontconfig-devel freetype-devel \\"
            echo "    wayland-devel wayland-protocols-devel \\"
            echo "    libxkbcommon-devel libXcursor-devel mesa-libEGL-devel"
            ;;
        arch)
            echo "Install on Arch:"
            echo "  sudo pacman -S rust cargo wtype pipewire pkg-config \\"
            echo "    alsa-lib fontconfig freetype2 \\"
            echo "    wayland wayland-protocols \\"
            echo "    libxkbcommon libxcursor mesa"
            ;;
        *)
            echo "Install the missing dependencies for your distribution."
            echo "Required packages: rust, cargo, wtype, pkg-config, pipewire,"
            echo "  alsa-lib, fontconfig, freetype2, wayland, wayland-protocols,"
            echo "  xkbcommon, xcursor, EGL (mesa)"
            ;;
    esac
    echo ""
    echo "For RPM packaging, also install: rpm-build"
    exit 1
fi
