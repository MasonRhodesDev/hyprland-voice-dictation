#!/bin/bash

echo "default"

if command -v pactl &> /dev/null; then
    pactl list sources short 2>/dev/null | awk '{print $2}' | grep -v "monitor"
elif command -v arecord &> /dev/null; then
    arecord -L 2>/dev/null | grep -v "^    " | grep -v "^$"
fi
