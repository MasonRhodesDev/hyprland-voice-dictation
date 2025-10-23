#!/bin/bash
set -e

VERSION="${VERSION:-${1:-0.1.0}}"
NAME="voice-dictation"
ARCH="amd64"

echo "=== Building DEB for $NAME v$VERSION ==="
echo ""

BUILD_DIR=$(mktemp -d)
PKG_DIR="$BUILD_DIR/${NAME}_${VERSION}_${ARCH}"

echo "1. Creating package structure..."
mkdir -p "$PKG_DIR"/{DEBIAN,usr/bin,usr/share/${NAME}/{scripts,models},usr/share/doc/${NAME}}

echo ""
echo "2. Building release binaries..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
cargo build --release

echo ""
echo "3. Copying binaries..."
cp target/release/dictation-engine "$PKG_DIR/usr/bin/"
cp target/release/dictation-gui "$PKG_DIR/usr/bin/"
chmod +x "$PKG_DIR/usr/bin/dictation-engine"
chmod +x "$PKG_DIR/usr/bin/dictation-gui"

echo ""
echo "4. Copying scripts..."
cp scripts/dictation-control "$PKG_DIR/usr/share/${NAME}/scripts/"
cp scripts/send_confirm.py "$PKG_DIR/usr/share/${NAME}/scripts/"
chmod +x "$PKG_DIR/usr/share/${NAME}/scripts/"*

echo ""
echo "5. Copying documentation..."
cp README.md "$PKG_DIR/usr/share/doc/${NAME}/"
cp LICENSE-MIT "$PKG_DIR/usr/share/doc/${NAME}/"
cp LICENSE-APACHE "$PKG_DIR/usr/share/doc/${NAME}/"

echo ""
echo "6. Creating control file..."
cat > "$PKG_DIR/DEBIAN/control" << EOF
Package: $NAME
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Depends: pipewire, python3
Recommends: wtype
Maintainer: Voice Dictation Contributors <noreply@example.com>
Description: Offline voice dictation for Linux with Wayland overlay
 Offline voice dictation system for Linux using Vosk speech recognition.
 Features a two-model approach with live preview and Wayland overlay showing
 audio spectrum and transcription.
 .
 This package requires Vosk models to be downloaded separately.
EOF

echo ""
echo "7. Creating postinst script..."
cat > "$PKG_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
echo ""
echo "Voice Dictation installed!"
echo ""
echo "To enable keybind, copy control scripts to ~/scripts/:"
echo "  mkdir -p ~/scripts"
echo "  cp /usr/share/voice-dictation/scripts/dictation-control ~/scripts/"
echo "  cp /usr/share/voice-dictation/scripts/send_confirm.py ~/scripts/"
echo ""
echo "Then add to Hyprland config:"
echo "  bind=\$Meh, V, exec, ~/scripts/dictation-control toggle"
echo ""
echo "Note: You need to download Vosk models separately (2GB):"
echo "  https://alphacephei.com/vosk/models"
echo ""
EOF
chmod +x "$PKG_DIR/DEBIAN/postinst"

echo ""
echo "8. Building package..."
dpkg-deb --build --root-owner-group "$PKG_DIR"

echo ""
echo "9. Moving package to current directory..."
mv "$PKG_DIR.deb" ./${NAME}_${VERSION}_${ARCH}.deb

echo ""
echo "10. Cleaning up..."
rm -rf "$BUILD_DIR"

echo ""
echo "✓ DEB build complete!"
echo ""
echo "Package: ./${NAME}_${VERSION}_${ARCH}.deb"
echo ""
echo "To install:"
echo "  sudo dpkg -i ${NAME}_${VERSION}_${ARCH}.deb"
echo "  sudo apt-get install -f  # Install dependencies if needed"
