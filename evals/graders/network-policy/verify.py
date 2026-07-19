"""Self-check that the hidden-grader wrapper disables network sockets."""

import json
import socket
import sys


try:
    socket.socket()
except RuntimeError:
    print(json.dumps({"passed": True}))
    raise SystemExit(0)

print(json.dumps({"passed": False}))
raise SystemExit(1)
