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
cargo build --release

%install
# Install binaries
mkdir -p %{buildroot}%{_bindir}
install -m 755 target/release/dictation-engine %{buildroot}%{_bindir}/
install -m 755 target/release/dictation-gui %{buildroot}%{_bindir}/

# Install scripts
mkdir -p %{buildroot}%{_datadir}/%{name}/scripts
install -m 755 scripts/dictation-control %{buildroot}%{_datadir}/%{name}/scripts/
install -m 755 scripts/send_confirm.py %{buildroot}%{_datadir}/%{name}/scripts/

# Install models
mkdir -p %{buildroot}%{_datadir}/%{name}/models
cp -r models/vosk-model-small-en-us-0.15 %{buildroot}%{_datadir}/%{name}/models/
cp -r models/vosk-model-en-us-0.22 %{buildroot}%{_datadir}/%{name}/models/

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
%{_bindir}/dictation-engine
%{_bindir}/dictation-gui
%{_datadir}/%{name}/scripts/dictation-control
%{_datadir}/%{name}/scripts/send_confirm.py
%{_datadir}/%{name}/models/vosk-model-small-en-us-0.15/*
%{_datadir}/%{name}/models/vosk-model-en-us-0.22/*

%changelog
* Mon Oct 21 2024 Mason <you@email.com> - 0.1.0-1
- Initial RPM release
- Two-model approach with live preview
- Wayland overlay with spectrum and text
