#!/usr/bin/env python3
"""
Spiritual Q&A Application Launcher
---------------------------------
This script launches both the backend API server and the frontend web server,
allowing users to access the application via their web browser with a single click.

How to use:
1. Simply double-click this file or run it from your terminal
2. Your default web browser will automatically open to the application
3. When done, close the terminal window or press Ctrl+C to stop the servers
"""

import os
import sys
import time
import webbrowser
import subprocess
import threading
import signal
import platform

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")

# Configuration
BACKEND_PORT = 8000
FRONTEND_PORT = 8080
APP_URL = f"http://localhost:{FRONTEND_PORT}"

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

# Global flag to track if servers are running
running = True

def display_banner():
    """Display a welcome banner for the application"""
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

def start_backend():
    """Start the backend FastAPI server"""
    os.chdir(PROJECT_DIR)
    try:
        if platform.system() == "Windows":
            return subprocess.Popen(
                ["python", os.path.join(BACKEND_DIR, "main.py")],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            return subprocess.Popen(
                ["python", os.path.join(BACKEND_DIR, "main.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
    except Exception as e:
        print(f"{Colors.FAIL}Error starting backend server: {e}{Colors.ENDC}")
        sys.exit(1)

def start_frontend():
    """Start the frontend HTTP server using Python's built-in server"""
    os.chdir(FRONTEND_DIR)
    try:
        if platform.system() == "Windows":
            return subprocess.Popen(
                ["python", "-m", "http.server", str(FRONTEND_PORT)],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            return subprocess.Popen(
                ["python", "-m", "http.server", str(FRONTEND_PORT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
    except Exception as e:
        print(f"{Colors.FAIL}Error starting frontend server: {e}{Colors.ENDC}")
        sys.exit(1)

def log_output(process, prefix):
    """Read and display the output from a process"""
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

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down servers"""
    global running
    running = False
    print(f"\n{Colors.WARNING}Shutting down servers... Please wait.{Colors.ENDC}")
    sys.exit(0)

def open_browser():
    """Open the web browser to the application URL"""
    time.sleep(3)  # Wait for servers to start
    print(f"{Colors.GREEN}Opening application in web browser: {APP_URL}{Colors.ENDC}")
    webbrowser.open(APP_URL)

def main():
    """Main function to start the application"""
    display_banner()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start backend server
    backend_process = start_backend()
    print(f"{Colors.GREEN}✓ Backend server started on port {BACKEND_PORT}{Colors.ENDC}")
    
    # Start frontend server
    frontend_process = start_frontend()
    print(f"{Colors.GREEN}✓ Frontend server started on port {FRONTEND_PORT}{Colors.ENDC}")
    
    # Create log threads if not on Windows
    if platform.system() != "Windows":
        backend_log = threading.Thread(target=log_output, args=(backend_process, "Backend"))
        backend_log.daemon = True
        backend_log.start()
        
        frontend_log = threading.Thread(target=log_output, args=(frontend_process, "Frontend"))
        frontend_log.daemon = True
        frontend_log.start()
    
    # Open browser after short delay
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print(f"{Colors.BLUE}Application is running!{Colors.ENDC}")
    print(f"{Colors.BLUE}Access the app at: {APP_URL}{Colors.ENDC}")
    print(f"{Colors.WARNING}Press Ctrl+C to stop the servers when done.{Colors.ENDC}")
    
    try:
        # Keep the main thread alive to monitor Ctrl+C
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up processes
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()

if __name__ == "__main__":
    main()
