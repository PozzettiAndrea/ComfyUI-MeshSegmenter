"""ComfyUI-MeshSegmenter Prestartup Script."""

import logging
from pathlib import Path
from comfy_env import setup_env, copy_files

log = logging.getLogger("meshsegmenter")

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy dynamic widgets JS
try:
    from comfy_dynamic_widgets import get_js_path
    import shutil
    src = Path(get_js_path())
    if src.exists():
        dst = SCRIPT_DIR / "web" / "js" / "dynamic_widgets.js"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
except ImportError:
    pass

# Copy example 3D assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "**/*")
