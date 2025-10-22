#!/bin/bash
set -e

VERSION="0.1.0"
NAME="voice-dictation"

echo "=== Building RPM for $NAME v$VERSION ==="
echo ""

# Setup RPM build environment
echo "1. Setting up RPM build tree..."
mkdir -p ~/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Create source tarball
echo ""
echo "2. Creating source tarball..."
cd /home/mason/repos/voice-dictation-rust
tar --exclude='.git' \
    --exclude='target' \
    --exclude='*.log' \
    --exclude='rpmbuild' \
    -czf ~/rpmbuild/SOURCES/${NAME}-${VERSION}.tar.gz \
    --transform "s,^,${NAME}-${VERSION}/," \
    .

# Copy spec file
echo ""
echo "3. Copying spec file..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/voice-dictation.spec" ~/rpmbuild/SPECS/

# Add license files if missing
if [ ! -f LICENSE-MIT ] || [ ! -f LICENSE-APACHE ]; then
    echo ""
    echo "4. Creating license files..."
    cat > LICENSE-MIT << 'EOF'
MIT License

Copyright (c) 2024 Voice Dictation Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

    cat > LICENSE-APACHE << 'EOF'
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

See full license text at: https://www.apache.org/licenses/LICENSE-2.0
EOF
fi

# Build RPM
echo ""
echo "5. Building RPM (this may take several minutes)..."
cd ~/rpmbuild/SPECS
rpmbuild -ba voice-dictation.spec 2>&1 | tee /tmp/rpm-build.log || {
    echo ""
    echo "✗ RPM build failed. Check /tmp/rpm-build.log for details"
    exit 1
}

echo ""
echo "✓ RPM build complete!"
echo ""
echo "RPMs located at:"
ls -lh ~/rpmbuild/RPMS/x86_64/${NAME}-*.rpm 2>/dev/null || \
ls -lh ~/rpmbuild/RPMS/noarch/${NAME}-*.rpm 2>/dev/null || \
echo "No RPMs found - check build output for errors"
echo ""
echo "To install:"
echo "  sudo dnf install ~/rpmbuild/RPMS/x86_64/${NAME}-${VERSION}-1.*.rpm"
