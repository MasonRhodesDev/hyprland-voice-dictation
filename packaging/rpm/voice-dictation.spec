Name:           voice-dictation
Version:        0.1.0
Release:        1%{?dist}
Summary:        Offline voice dictation for Linux with Wayland overlay

License:        MIT OR Apache-2.0
URL:            https://github.com/yourusername/voice-dictation-rust
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  rust
BuildRequires:  cargo
BuildRequires:  gcc
BuildRequires:  alsa-lib-devel
BuildRequires:  fontconfig-devel

Requires:       wtype
Requires:       pipewire
Requires:       pipewire-pulseaudio
Requires:       python3

%description
Offline voice dictation system for Linux using Vosk speech recognition.
Features a two-model approach with live preview and Wayland overlay showing
audio spectrum and transcription.

%prep
%setup -q

%build
# Configure cargo for non-interactive builds
export CARGO_NET_OFFLINE=false
export CARGO_TERM_COLOR=never
export CARGO_HTTP_TIMEOUT=300
export RUST_BACKTRACE=1

# Build with parallel jobs
echo "Starting cargo build (this may take 5-10 minutes)..."
cargo build --release --jobs=%{_smp_build_ncpus}
echo "Cargo build completed successfully"

%install
# Install binary
mkdir -p %{buildroot}%{_bindir}
install -m 755 target/release/voice-dictation %{buildroot}%{_bindir}/

# Install scripts
mkdir -p %{buildroot}%{_datadir}/%{name}/scripts
install -m 755 scripts/dictation-control %{buildroot}%{_datadir}/%{name}/scripts/
install -m 755 scripts/send_confirm.py %{buildroot}%{_datadir}/%{name}/scripts/

# Install models (if they exist)
mkdir -p %{buildroot}%{_datadir}/%{name}/models
if [ -d "models/vosk-model-small-en-us-0.15" ]; then
    cp -r models/vosk-model-small-en-us-0.15 %{buildroot}%{_datadir}/%{name}/models/
fi
if [ -d "models/vosk-model-en-us-0.22" ]; then
    cp -r models/vosk-model-en-us-0.22 %{buildroot}%{_datadir}/%{name}/models/
fi
if [ -d "models/vosk-model-en-us-daanzu-20200905-lgraph" ]; then
    cp -r models/vosk-model-en-us-daanzu-20200905-lgraph %{buildroot}%{_datadir}/%{name}/models/
fi

# Install documentation
mkdir -p %{buildroot}%{_docdir}/%{name}
install -m 644 README.md %{buildroot}%{_docdir}/%{name}/

%post
# Create symlinks in user's ~/scripts (optional, user can do this)
echo "Voice Dictation installed!"
echo "To enable keybind, copy control script to ~/scripts/:"
echo "  mkdir -p ~/scripts"
echo "  cp %{_datadir}/%{name}/scripts/dictation-control ~/scripts/"
echo "  cp %{_datadir}/%{name}/scripts/send_confirm.py ~/scripts/"
echo ""
echo "Then add to Hyprland config:"
echo "  bind=\$Meh, V, exec, ~/scripts/dictation-control toggle"

%files
%license LICENSE-MIT LICENSE-APACHE
%doc %{_docdir}/%{name}/README.md
%{_bindir}/voice-dictation
%{_datadir}/%{name}/scripts/dictation-control
%{_datadir}/%{name}/scripts/send_confirm.py
%{_datadir}/%{name}/models/

%changelog
* Mon Oct 21 2024 Mason <you@email.com> - 0.1.0-1
- Initial RPM release
- Two-model approach with live preview
- Wayland overlay with spectrum and text
