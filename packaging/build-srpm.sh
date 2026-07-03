#!/bin/bash
# Build the SRPM (source tarball from a git tag + vendored cargo deps) and
# optionally submit it to COPR.
#
# Release flow (Fedora + Arch from the same tag):
#   1. Bump Cargo.toml [workspace.package] version + spec Version
#      (+ %changelog) + PKGBUILD pkgver — one commit.
#   2. git tag vX.Y.Z && git push --tags
#   CI does the rest: the release workflow builds and publishes the Arch
#   package, and COPR rebuilds the SRPM off its GitHub webhook via
#   .copr/Makefile (which runs this script with --head).
#
# NOTE: the RPM %build needs network for the ort crate's ONNX Runtime
# download — the COPR project must have "enable internet access" on (see
# the comment block in hyprland-voice-dictation.spec).
#
# This script stays fully usable locally:
#   --head builds from HEAD instead of the tag (testing only — never
#   submit a --head build); --copr does a manual COPR submit.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME=hyprland-voice-dictation
SPEC="$REPO/packaging/$NAME.spec"
SOURCES="${HOME}/rpmbuild/SOURCES"
COPR_PROJECT="${COPR_PROJECT:-$NAME}"

VER=$(sed -n 's/^Version:[[:space:]]*//p' "$SPEC")
# Workspace-aware: the released version lives in [workspace.package] of the
# root Cargo.toml (the root crate and dictation-engine inherit it).
CARGO_VER=$(awk -F'"' '/^\[workspace\.package\]/{ws=1; next} /^\[/{ws=0} ws && /^version[[:space:]]*=/{print $2; exit}' "$REPO/Cargo.toml")
PKGBUILD_VER=$(sed -n 's/^pkgver=//p' "$REPO/packaging/PKGBUILD")
# Cargo.lock's own entry for the root crate (guards a stale lock). The
# binary crate is named voice-dictation; the package is $NAME.
LOCK_VER=$(awk '/^name = "voice-dictation"$/{getline; gsub(/version = "|"/,""); print; exit}' "$REPO/Cargo.lock")
mismatch=""
[ "$CARGO_VER" = "$VER" ] || mismatch="$mismatch\n  Cargo.toml [workspace.package]=$CARGO_VER"
[ "$PKGBUILD_VER" = "$VER" ] || mismatch="$mismatch\n  PKGBUILD pkgver=$PKGBUILD_VER"
[ "$LOCK_VER" = "$VER" ] || mismatch="$mismatch\n  Cargo.lock=$LOCK_VER"
if [ -n "$mismatch" ]; then
    echo "ERROR: version mismatch (spec Version=$VER):$(printf "$mismatch")" >&2
    echo "Bump spec, Cargo.toml [workspace.package], PKGBUILD pkgver, and Cargo.lock together." >&2
    exit 1
fi

REF="v$VER"
if [ "${1:-}" = "--head" ]; then
    REF="HEAD"
    echo "WARNING: building from HEAD (testing only)"
    shift
elif ! git -C "$REPO" rev-parse -q --verify "refs/tags/$REF" >/dev/null; then
    echo "ERROR: tag $REF not found — tag the release first (or use --head to test)" >&2
    exit 1
fi

mkdir -p "$SOURCES"
echo "==> source tarball from $REF"
git -C "$REPO" archive --format=tar.gz --prefix="$NAME-$VER/" \
    -o "$SOURCES/$NAME-$VER.tar.gz" "$REF"

echo "==> vendoring cargo dependencies (crates.io + git: schema-tui, ksni)"
VENDOR_DIR=$(mktemp -d)
trap 'rm -rf "$VENDOR_DIR"' EXIT
git -C "$REPO" archive --prefix=src/ "$REF" | tar -x -C "$VENDOR_DIR"
# Vendor OUTSIDE the source tree: the repo's own vendor/ holds the
# layer-shika-adapters path patch and `cargo vendor` would delete it.
(cd "$VENDOR_DIR/src" && cargo vendor --locked ../vendor > ../vendor-config.toml)
# Keep only the git source-replacement sections from cargo vendor's config
# output; the spec's %cargo_prep already writes the crates-io replacement,
# and duplicating those TOML tables would break cargo's config parse.
awk '/^\[source\."git\+/{p=1} /^\[source\.crates-io\]/{p=0} /^\[source\.vendored-sources\]/{p=0} p' \
    "$VENDOR_DIR/vendor-config.toml" > "$VENDOR_DIR/vendor-git-sources.toml"
tar -cJf "$SOURCES/$NAME-$VER-vendor.tar.xz" -C "$VENDOR_DIR" vendor vendor-git-sources.toml

echo "==> building SRPM"
SRPM=$(rpmbuild -bs "$SPEC" | sed -n 's/^Wrote: //p')
echo "    $SRPM"
# Gating: a clean tree should pass (domain-term/spelling noise filtered by the
# rpmlintrc). Failures here are real spec defects worth stopping for.
rpmlint --rpmlintrc "$REPO/packaging/$NAME.rpmlintrc" "$SRPM"

if [ "${1:-}" = "--copr" ]; then
    echo "==> submitting to COPR project $COPR_PROJECT"
    if ! copr-cli build "$COPR_PROJECT" "$SRPM"; then
        echo "ERROR: copr build failed. If this was a 401, the API token has" >&2
        echo "expired (~180 days) — renew at https://copr.fedorainfracloud.org/api/" >&2
        exit 1
    fi
fi
