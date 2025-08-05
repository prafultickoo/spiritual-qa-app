#!/usr/bin/env python3
"""
Dynamically launches the Spiritual Q&A application.

This script will:
1. Find free ports for the backend and frontend servers.
2. Generate a frontend config file with the backend's dynamic URL.
3. Start both the backend (Uvicorn) and frontend (http.server) processes.
4. Open the application in a new browser tab.
5. Manage the shutdown of both server processes gracefully on Ctrl+C.
"""

import os
import sys
import time
import webbrowser
import subprocess
import signal
from utils.network_utils import find_free_port

# --- Configuration & Setup ---

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
API_DIR = os.path.join(PROJECT_DIR, "api")

# --- Helper Functions ---

def create_frontend_config(backend_url):
    """Creates a JavaScript config file for the frontend to know the backend URL."""
    config_content = f"window.APP_CONFIG = {{ API_BASE_URL: '{backend_url}' }};"
    config_path = os.path.join(FRONTEND_DIR, 'config.js')
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"{Colors.GREEN}✓ Frontend config created at {config_path}{Colors.ENDC}")
    except IOError as e:
        print(f"{Colors.RED}Error: Could not write to {config_path}. {e}{Colors.ENDC}")
        sys.exit(1)

def start_process(command, name, cwd):
    """Starts a subprocess and returns its handle."""
    print(f"{Colors.BLUE}Starting {name} server...{Colors.ENDC}")
    print(f"{Colors.YELLOW}  > Command: {' '.join(command)}{Colors.ENDC}")
    print(f"{Colors.YELLOW}  > Directory: {cwd}{Colors.ENDC}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd
    )
    return process

# --- Main Application Logic ---

def main():
    """Main function to orchestrate the application launch."""
    print(f"{Colors.YELLOW}--- Starting Spiritual Q&A Application ---{Colors.ENDC}")

    # 1. Find available ports
    try:
        backend_port = find_free_port()
        frontend_port = find_free_port()
        backend_url = f"http://localhost:{backend_port}"
        frontend_url = f"http://localhost:{frontend_port}"
        print(f"{Colors.GREEN}✓ Ports allocated (Backend: {backend_port}, Frontend: {frontend_port}){Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Error: Could not find free ports. {e}{Colors.ENDC}")
        sys.exit(1)

    # 2. Create frontend configuration
    create_frontend_config(backend_url)

    # 3. Start servers
    backend_command = [sys.executable, "-m", "uvicorn", "spiritual_api:app", "--host", "0.0.0.0", "--port", str(backend_port)]
    frontend_command = [sys.executable, "-m", "http.server", str(frontend_port)]

    backend_process = start_process(backend_command, "Backend", API_DIR)
    frontend_process = start_process(frontend_command, "Frontend", FRONTEND_DIR)

    # Give servers a moment to initialize
    time.sleep(3)

    # 4. Open browser
    print(f"{Colors.BLUE}Opening application at {frontend_url}{Colors.ENDC}")
    webbrowser.open(frontend_url)
    print(f"\n{Colors.GREEN}--- Application is running! ---{Colors.ENDC}")
    print(f"{Colors.YELLOW}Press Ctrl+C to shut down.{Colors.ENDC}")

    # 5. Graceful shutdown handling
    def shutdown(signum, frame):
        print(f"\n{Colors.YELLOW}Ctrl+C detected. Shutting down servers...{Colors.ENDC}")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print(f"{Colors.GREEN}Servers shut down gracefully.{Colors.ENDC}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep the main thread alive to listen for signals
    try:
        # This is more portable than signal.pause()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # This is a fallback for environments where the signal handler doesn't catch it
        shutdown(None, None)


if __name__ == "__main__":
    main()
