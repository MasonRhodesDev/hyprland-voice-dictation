#!/usr/bin/env python3
"""Direct test - create server sockets and send messages"""
import socket
import json
import struct
import time
import os
import threading
from datetime import datetime

AUDIO_SOCK = "/tmp/voice-dictation.sock"
CONTROL_SOCK = "/tmp/voice-dictation-control.sock"

# Clean up old sockets
for sock in [AUDIO_SOCK, CONTROL_SOCK]:
    try:
        os.remove(sock)
    except:
        pass

def log(msg):
    """Print with timestamp"""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")

def audio_server():
    """Mock audio server"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(AUDIO_SOCK)
    sock.listen(1)
    log("Audio server listening...")
    
    conn, _ = sock.accept()
    log("✓ Audio client connected")
    
    # Send continuous silence (don't disconnect)
    while True:
        samples = [0.0] * 512
        data = struct.pack(f'{len(samples)}f', *samples)
        try:
            conn.sendall(data)
            time.sleep(0.1)
        except:
            break

def control_server():
    """Mock control server"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(CONTROL_SOCK)
    sock.listen(1)
    log("Control server listening...")
    
    conn, _ = sock.accept()
    log("✓ Control client connected")
    
    def send_msg(msg):
        data = json.dumps(msg).encode()
        conn.sendall(struct.pack('I', len(data)))
        conn.sendall(data)
        log(f"→ Sent: {msg}")
    
    time.sleep(2)
    log("\n=== Sending test sequence ===")
    
    send_msg({"TranscriptionUpdate": {"text": "Hello", "is_final": False}})
    time.sleep(2)
    
    send_msg({"TranscriptionUpdate": {"text": "Hello world", "is_final": False}})
    time.sleep(2)
    
    send_msg({"TranscriptionUpdate": {"text": "Hello world testing", "is_final": True}})
    time.sleep(1)
    
    send_msg("ProcessingStarted")
    log("Should see spinner now!")
    time.sleep(3)
    
    send_msg("Complete")
    log("Should close now!")
    time.sleep(2)
    
    log("\n=== Test complete ===")

# Start servers
audio_thread = threading.Thread(target=audio_server, daemon=True)
control_thread = threading.Thread(target=control_server, daemon=True)

audio_thread.start()
control_thread.start()

print("\nServers ready. Now start GUI in another terminal:")
print("  ~/.local/bin/dictation-gui")
print("\nWatch for:")
print("  1. 'Hello' text")
print("  2. 'Hello world' text")
print("  3. 'Hello world testing' text")
print("  4. Spinner animation")
print("  5. GUI closes")
print()

# Keep running
control_thread.join()
