#!/bin/bash
# Launcher script for voice dictation GUI with Wayland overlay layer configuration

# Force Wayland backend
export WINIT_UNIX_BACKEND=wayland

# Note: Iced 0.13 doesn't have native layer-shell support yet
# For now, we use AlwaysOnTop which works but doesn't draw over fullscreen apps
# 
# For true overlay functionality over fullscreen apps, we need one of:
# 1. Wait for Iced to add layer-shell support (planned for future releases)
# 2. Use a compositor-specific hack (swaymsg, hyprctl, etc.)
# 3. Patch winit to use layer-shell protocol

# Detect compositor
if [ "$XDG_CURRENT_DESKTOP" = "Hyprland" ]; then
    echo "Detected Hyprland - using windowrule for overlay"
    # Add Hyprland windowrule for true overlay
    hyprctl keyword windowrulev2 "float,title:^(Voice Dictation)$"
    hyprctl keyword windowrulev2 "pin,title:^(Voice Dictation)$"
    hyprctl keyword windowrulev2 "stayfocused,title:^(Voice Dictation)$"
    hyprctl keyword windowrulev2 "noborder,title:^(Voice Dictation)$"
    hyprctl keyword windowrulev2 "noblur,title:^(Voice Dictation)$"
    hyprctl keyword windowrulev2 "noshadow,title:^(Voice Dictation)$"
fi

# Run the GUI
exec "$(dirname "$0")/../target/release/dictation-gui"
