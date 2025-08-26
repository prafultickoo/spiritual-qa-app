"""
Utility helpers for launching the Spiritual Q&A app.

Each function does exactly one task and can be imported by launcher scripts.
"""
import os


def write_frontend_config(frontend_dir: str, backend_url: str) -> str:
    """Write config.js in the frontend directory with the backend API base URL.

    Args:
        frontend_dir: Absolute path to the frontend directory.
        backend_url: Base URL of the backend API (e.g., "http://localhost:8000").

    Returns:
        The absolute path to the written config.js file.
    """
    config_path = os.path.join(frontend_dir, "config.js")
    content = f"window.APP_CONFIG = {{ API_BASE_URL: '{backend_url}' }};"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return config_path
