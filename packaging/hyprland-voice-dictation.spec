# RPM spec for hyprland-voice-dictation. Built in COPR from a local SRPM
# produced by packaging/build-srpm.sh (source tarball from the git tag +
# vendored cargo deps as Source1 — no rust-*-devel packages needed).
#
# ============================================================================
# IMPORTANT — NETWORK REQUIRED AT BUILD TIME (ort / ONNX Runtime):
# the `ort` crate's build script downloads prebuilt ONNX Runtime binaries
# during %build. The vendored cargo sources make dependency *resolution*
# fully offline, but the ort download is an HTTP fetch outside cargo's
# control. The COPR project MUST have networking enabled
# (`copr-cli modify --enable-net on <project>` or the "Enable internet
# access" checkbox) until ort is switched to its `load-dynamic` feature
# with a system-provided onnxruntime library.
# ============================================================================
#
# The test suite runs by default. Disable for a one-off build with
# --without check; COPR builds run the suite.
%bcond_without check

# Force cargo to honor the committed Cargo.lock in every phase by appending
# --locked to the shared option set the %%cargo_build/%%cargo_test/%%cargo_install
# macros pass to cargo. Without this, %%cargo_install ("cargo install --path .")
# re-resolves the dependency graph from scratch — unlike "cargo build", it
# ignores an existing lock file unless told to keep it. Under our vendored,
# source-replaced config that re-resolution fails for the git dependencies
# (schema-tui, ksni): cargo will not fetch a git source offline without a lock
# pinning its commit ("the source git+... requires a lock file to be present
# first before it can be used against vendored source code"). The committed
# Cargo.lock already pins those revisions, so --locked makes %%install succeed
# (the %%build phase already builds fine with the lock). The macro's own option
# parser rejects a leading "--" argument, so we inject the flag here rather than
# as `%%cargo_install --locked`.
%global __cargo_common_opts %{?_smp_mflags} -Z avoid-dev-deps --locked

Name:           hyprland-voice-dictation
Version:        0.5.0
Release:        1%{?dist}
Summary:        Offline voice dictation for Hyprland with Parakeet speech recognition
# Project code is MIT OR Apache-2.0; the binary links a large dependency
# tree — see LICENSE.dependencies generated at build time.
License:        MIT OR Apache-2.0
URL:            https://github.com/MasonRhodesDev/hyprland-voice-dictation
Source0:        %{url}/archive/v%{version}/%{name}-%{version}.tar.gz
Source1:        %{name}-%{version}-vendor.tar.xz

BuildRequires:  rust
BuildRequires:  cargo
BuildRequires:  cargo-rpm-macros >= 24
BuildRequires:  systemd-rpm-macros
BuildRequires:  pkg-config
BuildRequires:  clang-devel
# ort-sys links the system ONNX Runtime (no network in COPR for its downloader)
BuildRequires:  pkgconfig(libonnxruntime)
# reqwest (VAD model download) pulls native-tls -> openssl-sys, which needs the
# openssl headers to build.
BuildRequires:  openssl-devel
BuildRequires:  pipewire-devel
BuildRequires:  alsa-lib-devel
BuildRequires:  fontconfig-devel
BuildRequires:  freetype-devel
BuildRequires:  libxkbcommon-devel
BuildRequires:  wayland-devel
BuildRequires:  systemd-devel
Requires:       wtype
Requires:       pipewire
Recommends:     playerctl

%description
Offline voice dictation daemon for Hyprland (and other Wayland compositors)
using NVIDIA Parakeet TDT speech recognition via ONNX Runtime. Press a key
to start recording, press again to transcribe and type the result into the
focused window with wtype. Ships a systemd user service and a standalone
model download script. The ~1.6 GB Parakeet model is NOT part of this
package; download it after install with `voice-dictation download-model`.

%prep
# -a1 unpacks the vendor tarball (vendor/ + vendor-git-sources.toml at its
# root) into the source dir; vendor/ merges with the in-tree
# third_party/layer-shika-adapters path patch.
%autosetup -p1 -a1
%cargo_prep -v vendor
# %%cargo_prep only redirects crates.io to the vendored sources. This
# workspace also has git dependencies (schema-tui, ksni); append the git
# source replacements captured from `cargo vendor` by build-srpm.sh so the
# build resolves them offline too.
cfg=.cargo/config.toml
[ -f "$cfg" ] || cfg=.cargo/config
cat vendor-git-sources.toml >> "$cfg"

%build
%cargo_build
%{cargo_license_summary}
%{cargo_license} > LICENSE.dependencies

%install
%cargo_install
install -Dpm0644 dist/voice-dictation.service %{buildroot}%{_userunitdir}/voice-dictation.service
install -Dpm0755 scripts/download-parakeet-model.sh %{buildroot}%{_datadir}/%{name}/download-parakeet-model.sh

%if %{with check}
%check
%cargo_test
%endif

%post
%systemd_user_post voice-dictation.service

%preun
%systemd_user_preun voice-dictation.service

%postun
%systemd_user_postun_with_restart voice-dictation.service

%files
%license LICENSE-MIT LICENSE-APACHE LICENSE.dependencies
%doc README.md
%{_bindir}/voice-dictation
%{_userunitdir}/voice-dictation.service
%dir %{_datadir}/%{name}
%{_datadir}/%{name}/download-parakeet-model.sh

%changelog
* Tue Jul 21 2026 Mason Rhodes <mrhodesdev@gmail.com> - 0.5.0-1
- Event-emitting StreamingEngine contract + LocalEngineDriver (streaming
  partials and correct finalize); daemon loop rewired to consume engine events
- Opt-in OpenAI transcription engine: set model = "openai:whisper-1" with
  OPENAI_API_KEY (batch transcription, no streaming partials)
- Minimal (style2) overlay now renders live transcription text; UI split into
  reusable components

* Tue Jul 14 2026 Mason Rhodes <mrhodesdev@gmail.com> - 0.4.1-1
- BuildRequires pkgconfig(libonnxruntime): COPR builders have no network for
  ort-sys' prebuilt-binary fallback

* Sat Jul 04 2026 Mason Rhodes <mrhodesdev@gmail.com> - 0.4.0-1
- Correction/dictionary UX: dict/subst/corrections CLI, learned-correction
  store with hot-reload, wezterm-native correction backend, IME probe
- Reject shell-execution as corrections; hot-reload corrections.json on CLI edits
- Real AT-SPI2 e2e test; fix AT-SPI listener dying on first stream error

* Fri Jul 03 2026 Mason Rhodes <mrhodesdev@gmail.com> - 0.3.4-1
- Disable makepkg LTO (onig C objects vs ld.lld)

* Fri Jul 03 2026 Mason Rhodes <mrhodesdev@gmail.com> - 0.3.3-1
- pipewire-rs 0.10 migration (builds against pipewire 1.6)

* Fri Jul 03 2026 Mason Rhodes <mrhodesdev@gmail.com> - 0.3.2-1
- Fix first-run CI gates (see git log)

* Fri Jul 03 2026 Mason Rhodes <mrhodesdev@gmail.com> - 0.3.1-1
- Standardized packaging release: shared CI, arch-repo + COPR pipeline

* Thu Jul 02 2026 Mason Rhodes <mrhodesdev@gmail.com> - 0.3.0-1
- Standardized packaging: PKGBUILD + RPM spec share the dist/ payload,
  systemd user unit uses the packaged /usr/bin/voice-dictation path
- First COPR-buildable release (vendored cargo sources incl. git deps)
