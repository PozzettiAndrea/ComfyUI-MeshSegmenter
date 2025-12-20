# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh Installation - Clone and setup the SAMesh library.
"""

import os
import subprocess
from pathlib import Path


def install_samesh():
    """Install the SAMesh library as a git submodule."""
    print("\n" + "="*60)
    print("ComfyUI-MeshSegmenter: SAMesh Installation")
    print("="*60 + "\n")

    # Get the directory containing this script's parent (the repo root)
    script_dir = Path(__file__).parent.parent.absolute()
    samesh_dir = script_dir / "samesh-main"

    if samesh_dir.exists() and (samesh_dir / "src" / "samesh").exists():
        print("[Install] SAMesh library already present.")

        # Check for SAM2 submodule
        sam2_dir = samesh_dir / "third_party" / "segment-anything-2"
        if not (sam2_dir / "sam2").exists():
            print("[Install] SAM2 submodule appears incomplete. Initializing...")
            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=str(samesh_dir),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                print("[Install] SAM2 submodule initialized.")
            except Exception as e:
                print(f"[Install] Warning: Could not initialize SAM2 submodule: {e}")
                print("[Install] You may need to run manually:")
                print(f"[Install]   cd {samesh_dir} && git submodule update --init --recursive")
        else:
            print("[Install] SAM2 submodule is present.")

        return True

    print("[Install] SAMesh library not found.")
    print("[Install] Please ensure the samesh-main directory is present with:")
    print("[Install]   - src/samesh/ - the main SAMesh source code")
    print("[Install]   - third_party/segment-anything-2/ - SAM2 submodule")
    print("")
    print("[Install] You can clone it from the SAMesh repository:")
    print("[Install]   git clone https://github.com/YOUR_SAMESH_REPO samesh-main")
    print("[Install]   cd samesh-main && git submodule update --init --recursive")

    return False
