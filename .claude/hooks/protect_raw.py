#!/usr/bin/env python3
"""Blocks any write into data/raw/. Exit code 2 stops the tool call."""
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool_input = payload.get("tool_input", {}) or {}
candidates = [
    str(tool_input.get("file_path", "")),
    str(tool_input.get("command", "")),
]

for text in candidates:
    normalized = text.replace("\\", "/")
    if "data/raw" in normalized:
        print("BLOCKED: data/raw is read-only source data.", file=sys.stderr)
        sys.exit(2)

sys.exit(0)