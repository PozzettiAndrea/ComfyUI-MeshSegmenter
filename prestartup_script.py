"""ComfyUI-MeshSegmenter Prestartup Script."""

import logging
from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer

log = logging.getLogger("meshsegmenter")

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy text report viewer (JS widget + utils)
try:
    copy_viewer("text_report", SCRIPT_DIR / "web")
except Exception as e:
    log.warning(f"Failed to copy text_report viewer: {e}")


# Copy example 3D assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "**/*")
