# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Python Dependencies Installation - pip requirements for MeshSegmenter.
"""

import sys
import subprocess
from pathlib import Path


def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("\n" + "="*60)
    print("ComfyUI-MeshSegmenter: Python Dependencies Installation")
    print("="*60 + "\n")

    # Look for requirements.txt relative to this script's parent directory
    script_dir = Path(__file__).parent.parent.absolute()
    requirements_file = script_dir / "requirements.txt"

    if not requirements_file.exists():
        print(f"[Install] Warning: requirements.txt not found at {requirements_file}")
        print("[Install] Skipping Python dependencies installation.")
        return True

    print(f"[Install] Installing core Python dependencies...")
    print(f"[Install] This may take a few minutes...\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("\n[Install] All Python dependencies installed successfully!")
            return True
        else:
            print(f"\n[Install] Warning: Some packages failed to install")
            print("[Install] Attempting to install core dependencies individually...")

            core_packages = [
                "requests>=2.25.0", "tqdm>=4.60.0",
                "numpy>=1.21.0", "scipy>=1.7.0",
                "trimesh>=3.15.0",
                "Pillow>=9.0.0", "opencv-python>=4.5.0",
                "torch>=2.0.0", "torchvision>=0.15.0",
                "huggingface_hub>=0.20.0",
                "omegaconf>=2.3.0",
                "pyrender>=0.1.45",
                "networkx>=2.6.0", "python-igraph>=0.10.0"
            ]

            result_individual = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + core_packages,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result_individual.returncode == 0:
                print("\n[Install] Core Python dependencies installed successfully!")
                return True
            else:
                print(f"\n[Install] Error installing Python dependencies:")
                print(result_individual.stderr)
                print("\n[Install] You can try installing manually with:")
                print(f"[Install]   pip install -r {requirements_file}")
                return False

    except subprocess.TimeoutExpired:
        print("\n[Install] Error: Installation timed out after 10 minutes")
        print("[Install] You can try installing manually with:")
        print(f"[Install]   pip install -r {requirements_file}")
        return False
    except Exception as e:
        print(f"\n[Install] Error installing Python dependencies: {e}")
        print("[Install] You can try installing manually with:")
        print(f"[Install]   pip install -r {requirements_file}")
        return False
