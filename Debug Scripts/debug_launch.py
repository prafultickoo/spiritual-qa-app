#!/usr/bin/env python3
"""
Debug launcher for the Spiritual Q&A application.

This script mirrors the dynamic logic of launch_app.py but provides more
verbose, real-time logging for both backend and frontend processes to help
diagnose startup issues.
"""

import os
import sys
import time
import webbrowser
import subprocess
import threading
import signal
from utils.network_utils import find_free_port

# --- Configuration & Setup ---

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
API_DIR = os.path.join(PROJECT_DIR, "api")

# --- Helper Functions ---

def create_frontend_config(backend_url):
    """Creates a JavaScript config file for the frontend."""
    config_content = f"window.APP_CONFIG = {{ API_BASE_URL: '{backend_url}' }};"
    config_path = os.path.join(FRONTEND_DIR, 'config.js')
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"{Colors.GREEN}✓ Frontend config created at {config_path}{Colors.ENDC}")
    except IOError as e:
        print(f"{Colors.RED}Error: Could not write to {config_path}. {e}{Colors.ENDC}")
        sys.exit(1)

def log_output(pipe, name, color):
    """Reads and logs output from a subprocess's pipe in real-time."""
    try:
        for line in iter(pipe.readline, ''):
            print(f"{color}[{name}] {line.strip()}{Colors.ENDC}")
    except ValueError:
        # Pipe closed
        pass
    finally:
        pipe.close()

def check_process(proc, name):
    """Checks if a process is still running after a short delay."""
    time.sleep(3)  # Give process time to start or fail
    if proc.poll() is not None:
        print(f"{Colors.RED}✗ [{name}] Process exited unexpectedly with code {proc.returncode}. Check logs.{Colors.ENDC}")
        return False
    print(f"{Colors.GREEN}✓ [{name}] Process started successfully (PID: {proc.pid}).{Colors.ENDC}")
    return True

# --- Main Application Logic ---

def main():
    """Main function to orchestrate the application launch with debugging."""
    print(f"{Colors.YELLOW}--- Starting Spiritual Q&A Application (DEBUG MODE) ---{Colors.ENDC}")

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

    processes = []

    def shutdown(signum, frame):
        print(f"\n{Colors.YELLOW}Ctrl+C detected. Shutting down all processes...{Colors.ENDC}")
        for p in processes:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        print(f"{Colors.GREEN}All processes shut down.{Colors.ENDC}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 3. Start Backend Server
    print(f"\n{Colors.BLUE}--- Starting Backend Server ---{Colors.ENDC}")
    backend_command = [sys.executable, "-m", "uvicorn", "spiritual_api:app", "--host", "0.0.0.0", "--port", str(backend_port)]
    backend_proc = subprocess.Popen(
        backend_command,
        cwd=API_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, # Ensures output is decoded as text
        bufsize=1, # Line-buffered
        universal_newlines=True
    )
    processes.append(backend_proc)

    threading.Thread(target=log_output, args=(backend_proc.stdout, 'Backend-out', Colors.CYAN), daemon=True).start()
    threading.Thread(target=log_output, args=(backend_proc.stderr, 'Backend-err', Colors.RED), daemon=True).start()

    if not check_process(backend_proc, "Backend"):
        shutdown(None, None)

    # 4. Start Frontend Server
    print(f"\n{Colors.BLUE}--- Starting Frontend Server ---{Colors.ENDC}")
    frontend_command = [sys.executable, "-m", "http.server", str(frontend_port)]
    frontend_proc = subprocess.Popen(
        frontend_command,
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    processes.append(frontend_proc)

    threading.Thread(target=log_output, args=(frontend_proc.stdout, 'Frontend-out', Colors.CYAN), daemon=True).start()
    threading.Thread(target=log_output, args=(frontend_proc.stderr, 'Frontend-err', Colors.RED), daemon=True).start()

    if not check_process(frontend_proc, "Frontend"):
        shutdown(None, None)

    # 5. Open Browser
    print(f"\n{Colors.BLUE}Opening application at {frontend_url}{Colors.ENDC}")
    webbrowser.open(frontend_url)

    print(f"\n{Colors.GREEN}--- Application is running in DEBUG mode! ---{Colors.ENDC}")
    print(f"{Colors.YELLOW}Press Ctrl+C to shut down.{Colors.ENDC}")

    # Keep main thread alive to monitor processes
    try:
        while True:
            # Check if any process has exited
            for p in processes:
                if p.poll() is not None:
                    print(f"{Colors.RED}A process has exited unexpectedly. Shutting down.{Colors.ENDC}")
                    shutdown(None, None)
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)

if __name__ == "__main__":
    main()
