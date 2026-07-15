#!/bin/bash
# Phase 6 — One-time system dependency setup.
# Run this ONCE per Lightning Studio (or whenever you spin up a fresh one).
# These are OS-level libraries that pip cannot install — they used to be
# handled automatically by Modal's .apt_install() at image-build time.

set -e  # stop on first error, so you notice if something fails

echo "[setup] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y ffmpeg poppler-utils libsndfile1-dev libgl1

echo "[setup] Done. These persist on this Studio's disk — you won't need"
echo "        to run this again unless you create a brand new Studio."
