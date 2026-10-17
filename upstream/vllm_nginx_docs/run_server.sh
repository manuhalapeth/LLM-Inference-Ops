#!/bin/sh
# Restart loop so a test can take the server down (touch /tmp/down; pkill) and bring it
# back (rm /tmp/down) without the container, its IP or its DNS name changing.
while true; do
  [ -f /tmp/down ] || python /app/server.py
  sleep 0.3
done
