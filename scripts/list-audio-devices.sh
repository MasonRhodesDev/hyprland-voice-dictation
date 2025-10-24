#!/bin/bash
# List ALSA audio input devices with friendly descriptions
# Filters out duplicates and prefers commonly used device types

# Always include default first with description
echo "default (System Default)"

# Parse arecord -L output to get device names with descriptions
arecord -L 2>/dev/null | awk '
BEGIN { device = ""; desc = "" }
/^[^ ]/ {
    # Line without leading space = device name
    if (device != "" && device != "null" && device != "default") {
        # Skip certain device types to avoid duplicates
        # Skip: front:, surround:, iec958:, hdmi:, dmix, dsnoop
        if (device !~ /^(front|surround|iec958|hdmi|dmix|dsnoop):/) {
            # Output previous device with description
            if (desc != "") {
                # Clean up description - take first line only
                gsub(/^[ \t]+/, "", desc)
                printf "%s (%s)\n", device, desc
            } else {
                printf "%s\n", device
            }
        }
    }
    device = $0
    desc = ""
}
/^[ ]/ {
    # Line with leading space = description
    if (desc == "") {
        desc = $0
    }
}
END {
    # Output last device
    if (device != "" && device != "null" && device != "default") {
        if (device !~ /^(front|surround|iec958|hdmi|dmix|dsnoop):/) {
            if (desc != "") {
                gsub(/^[ \t]+/, "", desc)
                printf "%s (%s)\n", device, desc
            } else {
                printf "%s\n", device
            }
        }
    }
}
' | sort -u
