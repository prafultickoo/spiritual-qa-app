#!/usr/bin/env python3
"""
Spiritual Q&A Application Launcher (Dynamic Ports)
-------------------------------------------------
Launches the backend API and frontend servers with dynamically allocated ports
and generates the frontend config to prevent port mismatches.

How to use:
1. Run this file (double-click or via terminal).
2. Your browser opens to the frontend URL.
3. Press Ctrl+C to stop both servers.
"""

import os
import sys
import time
import webbrowser
import subprocess
import threading
import signal
import platform

from utils.network_utils import find_free_port
from utils.app_launcher_utils import write_frontend_config

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
API_DIR = os.path.join(PROJECT_DIR, "api")

# Terminal colors for better visibility
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Disable colors if on Windows (unless using a modern terminal)
if platform.system() == "Windows" and "TERM" not in os.environ:
    for attr in dir(Colors):
        if not attr.startswith('__'):
            setattr(Colors, attr, '')

# Global state
running = True

def display_banner():
    """Display a welcome banner for the application."""
    banner = f"""{Colors.HEADER}
    ╔═══════════════════════════════════════════════╗
    ║                                               ║
    ║   {Colors.BOLD}Spiritual Q&A Application{Colors.ENDC}{Colors.HEADER}                  ║
    ║   {Colors.GREEN}Ancient Wisdom for Modern Questions{Colors.ENDC}{Colors.HEADER}        ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝{Colors.ENDC}
    """
    print(banner)
    print(f"{Colors.BLUE}Starting servers... Please wait.{Colors.ENDC}")

def start_backend(backend_port: int) -> subprocess.Popen:
    """Start the backend FastAPI server via Uvicorn in the api/ directory.

    Args:
        backend_port: Port to bind the backend server to.

    Returns:
        The subprocess handle for the backend server.
    """
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", "spiritual_api:app",
            "--host", "0.0.0.0", "--port", str(backend_port)
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=API_DIR,
        )
    except Exception as e:
        print(f"{Colors.FAIL}Error starting backend server: {e}{Colors.ENDC}")
        sys.exit(1)

def start_frontend(frontend_port: int) -> subprocess.Popen:
    """Start the frontend HTTP server using Python's built-in server.

    Args:
        frontend_port: Port to bind the frontend server to.

    Returns:
        The subprocess handle for the frontend server.
    """
    try:
        cmd = [sys.executable, "-m", "http.server", str(frontend_port)]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=FRONTEND_DIR,
        )
    except Exception as e:
        print(f"{Colors.FAIL}Error starting frontend server: {e}{Colors.ENDC}")
        sys.exit(1)

def log_output(process: subprocess.Popen, prefix: str) -> None:
    """Stream and print process output with a prefix."""
    global running
    try:
        for line in iter(process.stdout.readline, ''):
            if not running:
                break
            if line:
                print(f"{prefix}: {line.strip()}")
    except Exception as e:
        if running:
            print(f"{Colors.WARNING}Error reading {prefix} output: {e}{Colors.ENDC}")

def main():
    """Main function to start the application with dynamic ports and config."""
    display_banner()

    # Allocate ports
    backend_port = find_free_port()
    frontend_port = find_free_port()
    backend_url = f"http://localhost:{backend_port}"
    frontend_url = f"http://localhost:{frontend_port}"

    # Generate frontend config
    write_frontend_config(FRONTEND_DIR, backend_url)
    print(f"{Colors.GREEN}✓ Frontend config set to {backend_url}{Colors.ENDC}")

    # Start servers
    backend_process = start_backend(backend_port)
    print(f"{Colors.GREEN}✓ Backend server started on port {backend_port}{Colors.ENDC}")

    frontend_process = start_frontend(frontend_port)
    print(f"{Colors.GREEN}✓ Frontend server started on port {frontend_port}{Colors.ENDC}")

    # Log output (both platforms)
    backend_log = threading.Thread(target=log_output, args=(backend_process, "Backend"), daemon=True)
    backend_log.start()
    frontend_log = threading.Thread(target=log_output, args=(frontend_process, "Frontend"), daemon=True)
    frontend_log.start()

    # Open browser after short delay
    time.sleep(2)
    print(f"{Colors.GREEN}Opening application in web browser: {frontend_url}{Colors.ENDC}")
    webbrowser.open(frontend_url)

    print(f"{Colors.BLUE}Application is running!{Colors.ENDC}")
    print(f"{Colors.BLUE}Access the app at: {frontend_url}{Colors.ENDC}")
    print(f"{Colors.WARNING}Press Ctrl+C to stop the servers when done.{Colors.ENDC}")

    # Graceful shutdown with Ctrl+C
    def shutdown(signum=None, frame=None):
        global running
        running = False
        print(f"\n{Colors.WARNING}Shutting down servers... Please wait.{Colors.ENDC}")
        try:
            if backend_process:
                backend_process.terminate()
            if frontend_process:
                frontend_process.terminate()
            if backend_process:
                backend_process.wait(timeout=10)
            if frontend_process:
                frontend_process.wait(timeout=10)
        except Exception:
            pass
        finally:
            print(f"{Colors.GREEN}Servers shut down gracefully.{Colors.ENDC}")
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep alive
    try:
        while running:
            time.sleep(0.25)
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()
