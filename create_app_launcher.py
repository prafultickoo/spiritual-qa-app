#!/usr/bin/env python3
"""
Script to create a macOS Application bundle for the Spiritual Q&A app.
This will create a double-clickable app in the project directory.
"""

import os
import sys
import stat
import shutil

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "Spiritual Q&A"
APP_DIR = os.path.join(PROJECT_DIR, f"{APP_NAME}.app")
CONTENTS_DIR = os.path.join(APP_DIR, "Contents")
MACOS_DIR = os.path.join(CONTENTS_DIR, "MacOS")
RESOURCES_DIR = os.path.join(CONTENTS_DIR, "Resources")

# Create necessary directories
os.makedirs(MACOS_DIR, exist_ok=True)
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Create Info.plist
INFO_PLIST = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.spiritual.qa</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>{APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.12</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""

# Write Info.plist
with open(os.path.join(CONTENTS_DIR, "Info.plist"), "w") as f:
    f.write(INFO_PLIST)

# Create launcher script
LAUNCHER_SCRIPT = """#!/bin/bash

# Get directory of this script and navigate to project root
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="$( dirname "$DIR" )"
CONTENTS_DIR="$( dirname "$APP_DIR" )"
PROJECT_DIR="$( dirname "$CONTENTS_DIR" )"
cd "$PROJECT_DIR"

# Colorful output
BLUE='\\033[0;34m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
RESET='\\033[0m'

# Display banner
echo -e "${YELLOW}"
echo "╔═══════════════════════════════════════════════╗"
echo "║                                               ║"
echo "║   Spiritual Q&A Application                   ║"
echo "║   Ancient Wisdom for Modern Questions         ║"
echo "║                                               ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "${BLUE}Starting servers... Please wait.${RESET}"

# Start backend server in background
echo "Starting backend server..."
cd "$PROJECT_DIR"
# For Python 3 compatibility, use 'python3' if available
if command -v python3 &>/dev/null; then
    python3 "$PROJECT_DIR/backend/main.py" &
else
    python "$PROJECT_DIR/backend/main.py" &
fi
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# Start frontend server in background
echo "Starting frontend server..."
cd "$PROJECT_DIR/frontend"
# For Python 3 compatibility, use 'python3' if available
if command -v python3 &>/dev/null; then
    python3 -m http.server 8080 &
else
    python -m http.server 8080 &
fi
FRONTEND_PID=$!

# Wait a moment for frontend to initialize
sleep 1

# Open browser
echo -e "${GREEN}Opening application in web browser...${RESET}"
open "http://localhost:8080"

echo -e "${GREEN}Application is running!${RESET}"
echo -e "${BLUE}Access the app at: http://localhost:8080${RESET}"
echo -e "${YELLOW}Close this terminal window when you're done using the app.${RESET}"

# Keep script running to keep servers alive
wait $BACKEND_PID $FRONTEND_PID
"""

# Write launcher script
launcher_path = os.path.join(MACOS_DIR, "launcher")
with open(launcher_path, "w") as f:
    f.write(LAUNCHER_SCRIPT)

# Make launcher executable
os.chmod(launcher_path, os.stat(launcher_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

print(f"✅ Created application bundle: {APP_DIR}")
print(f"✅ You can now double-click '{APP_NAME}.app' to start the Spiritual Q&A application")
