#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
MeshSegmenter Installer

Automatically installs all dependencies for ComfyUI-MeshSegmenter.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_pip(*args):
    """Run pip with the given arguments."""
    return subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True)


def install_system_dependencies():
    """Install system-level dependencies (Linux only)."""
    print("\n[1/5] Checking system dependencies...")

    if sys.platform != "linux":
        print("  Skipping (not Linux)")
        return True

    # Check if main EGL library is available (not just NVIDIA drivers)
    egl_found = False
    try:
        result = subprocess.run(["/sbin/ldconfig", "-p"], capture_output=True, text=True)
        # Look specifically for libEGL.so (the main library, not nvidia-specific ones)
        for line in result.stdout.split('\n'):
            if 'libEGL.so' in line and 'nvidia' not in line.lower():
                egl_found = True
                break
    except:
        pass

    if egl_found:
        print("  EGL libraries found")
        return True

    print("  Installing EGL/OpenGL libraries for headless rendering...")
    try:
        # Clean apt cache first to avoid stale package errors
        subprocess.run(
            ["sudo", "apt-get", "clean"],
            capture_output=True,
            timeout=60
        )
        subprocess.run(
            ["sudo", "rm", "-rf", "/var/lib/apt/lists/*"],
            capture_output=True,
            timeout=60
        )
        result = subprocess.run(
            ["sudo", "apt-get", "update"],
            capture_output=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f"  Warning: apt-get update failed: {result.stderr}")

        # Install EGL packages - libegl1 provides libEGL.so
        result = subprocess.run(
            ["sudo", "apt-get", "install", "-y",
             "libegl1", "libegl-dev", "libgl1-mesa-glx", "libosmesa6-dev"],
            capture_output=True,
            timeout=180
        )
        if result.returncode != 0:
            print(f"  Warning: apt-get install failed: {result.stderr}")
            raise Exception(result.stderr)

        print("  System dependencies installed")
        return True
    except Exception as e:
        print(f"  Warning: Could not install system deps: {e}")
        print("  You may need to run: sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/* && sudo apt-get update && sudo apt-get install -y libegl1 libegl-dev libgl1-mesa-glx libosmesa6-dev")
        return False


def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("\n[2/5] Installing Python dependencies...")

    script_dir = Path(__file__).parent.absolute()
    requirements_file = script_dir / "requirements.txt"

    if not requirements_file.exists():
        print(f"  Error: requirements.txt not found at {requirements_file}")
        return False

    print(f"  Installing from {requirements_file}")
    result = run_pip("install", "-r", str(requirements_file))

    if result.returncode != 0:
        print(f"  Error installing dependencies:")
        print(result.stderr)
        return False

    print("  Python dependencies installed successfully")
    return True


def get_torch_cuda_info():
    """Get PyTorch version and CUDA version string."""
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.1.0"

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda  # e.g., "12.1"
            if cuda_version:
                cuda_version = cuda_version.replace('.', '')  # "121"
                cuda_str = f"cu{cuda_version[:3]}"  # "cu121"
            else:
                cuda_str = "cpu"
        else:
            cuda_str = "cpu"

        return torch_version, cuda_str
    except ImportError:
        return None, None


def install_torch_scatter():
    """Install torch_scatter with the correct CUDA-matched wheel."""
    print("\n[3/5] Installing torch_scatter...")

    # Check if already installed
    try:
        import torch_scatter
        print("  torch_scatter is already installed")
        return True
    except ImportError:
        pass

    torch_version, cuda_str = get_torch_cuda_info()
    if torch_version is None:
        print("  Error: PyTorch not found. Install PyTorch first.")
        return False

    # PyG wheel URL format
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_str}.html"

    print(f"  Detected PyTorch {torch_version} with {cuda_str}")
    print(f"  Installing from: {wheel_url}")

    result = run_pip("install", "torch_scatter", "-f", wheel_url)
    if result.returncode == 0:
        print("  torch_scatter installed successfully")
        return True

    # Try with major.minor.0 version
    torch_major_minor = '.'.join(torch_version.split('.')[:2]) + '.0'
    wheel_url_alt = f"https://data.pyg.org/whl/torch-{torch_major_minor}+{cuda_str}.html"
    print(f"  Retrying with: {wheel_url_alt}")

    result = run_pip("install", "torch_scatter", "-f", wheel_url_alt)
    if result.returncode == 0:
        print("  torch_scatter installed successfully")
        return True

    print(f"  Error: Could not install torch_scatter")
    print(f"  Try manually: pip install torch_scatter -f {wheel_url}")
    return False


def verify_samesh():
    """Verify SAMesh can be imported."""
    print("\n[4/5] Verifying SAMesh installation...")

    script_dir = Path(__file__).parent.absolute()
    samesh_src = script_dir / "samesh-main" / "src"

    if not samesh_src.exists():
        print(f"  Error: SAMesh source not found at {samesh_src}")
        print("  Please ensure samesh-main/src/samesh exists")
        return False

    # Set environment for headless rendering
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Add samesh to path and try import
    sys.path.insert(0, str(samesh_src))

    try:
        from samesh.models.sam_mesh import segment_mesh
        print("  SAMesh imports working correctly")
        return True
    except ImportError as e:
        print(f"  Warning: SAMesh import failed: {e}")
        print("  Some dependencies may be missing")
        return False


def verify_partfield():
    """Verify PartField can be imported."""
    print("\n[5/5] Verifying PartField installation...")

    script_dir = Path(__file__).parent.absolute()
    partfield_src = script_dir / "partfield-src"

    if not partfield_src.exists():
        print(f"  PartField source not found at {partfield_src}")
        print("  PartField nodes will not be available")
        return False

    sys.path.insert(0, str(partfield_src))

    try:
        from partfield.model_trainer_pvcnn_only_demo import Model
        print("  PartField imports working correctly")
        return True
    except ImportError as e:
        print(f"  Warning: PartField import failed: {e}")
        print("  PartField nodes may not work")
        return False


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("ComfyUI-MeshSegmenter: Installation")
    print("="*60)

    results = {
        'system_deps': install_system_dependencies(),
        'python_deps': install_python_dependencies(),
        'torch_scatter': install_torch_scatter(),
        'samesh': verify_samesh(),
        'partfield': verify_partfield(),
    }

    # Print summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    print(f"  System Dependencies: {'OK' if results['system_deps'] else 'WARN'}")
    print(f"  Python Dependencies: {'OK' if results['python_deps'] else 'FAIL'}")
    print(f"  torch_scatter:       {'OK' if results['torch_scatter'] else 'FAIL'}")
    print(f"  SAMesh Verification: {'OK' if results['samesh'] else 'WARN'}")
    print(f"  PartField Verify:    {'OK' if results['partfield'] else 'WARN'}")
    print("="*60 + "\n")

    if results['python_deps'] and results['torch_scatter']:
        print("Installation completed!")
        if not results['samesh']:
            print("\nNote: SAMesh verification failed but may still work.")
        if not results['partfield']:
            print("\nNote: PartField verification failed but may still work.")
        print("Try running ComfyUI and check for errors.")
        sys.exit(0)
    else:
        print("Installation failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
