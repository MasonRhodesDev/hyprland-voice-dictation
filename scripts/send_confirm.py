#!/usr/bin/env python3
import socket
import json
import struct
import sys

CONTROL_SOCKET = "/tmp/voice-dictation-control.sock"

def send_confirm():
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(CONTROL_SOCKET)
        
        msg = {"Confirm": None}
        data = json.dumps(msg).encode('utf-8')
        length = struct.pack('>I', len(data))
        
        sock.sendall(length + data)
        sock.close()
        return True
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if send_confirm():
        print("Confirm command sent")
        sys.exit(0)
    else:
        sys.exit(1)
