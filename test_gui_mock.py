#!/usr/bin/env python3
import socket
import json
import struct
import time
import sys

def send_control_message(msg):
    """Send a control message to the GUI"""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect("/tmp/voice-dictation-control.sock")
        
        data = json.dumps(msg).encode()
        sock.sendall(struct.pack('I', len(data)))
        sock.sendall(data)
        print(f"✓ Sent: {msg}")
        sock.close()
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

print("=== GUI Mock Test ===")
print("Make sure GUI is running: ~/.local/bin/dictation-gui")
print()

time.sleep(2)

print("1. Sending transcription update (not final)...")
send_control_message({"TranscriptionUpdate": {"text": "Hello world", "is_final": False}})
time.sleep(2)

print("2. Sending transcription update (final)...")
send_control_message({"TranscriptionUpdate": {"text": "Hello world testing", "is_final": True}})
time.sleep(2)

print("3. Sending ProcessingStarted (should show spinner)...")
send_control_message({"ProcessingStarted": None})
time.sleep(3)

print("4. Sending Complete (should close)...")
send_control_message({"Complete": None})
time.sleep(2)

print("\n=== Test complete ===")
print("Did you see:")
print("  - Text appear in GUI?")
print("  - Spinner animation?")
print("  - GUI close?")
