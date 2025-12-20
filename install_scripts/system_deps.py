# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
System Dependencies Installation - OpenGL/EGL libraries for pyrender.
"""

import platform
import subprocess


def get_platform_info():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        plat = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return plat, arch


def install_system_dependencies():
    """Install required system dependencies (Linux only)."""
    plat, _ = get_platform_info()

    if plat != "linux":
        return True

    print("\n" + "="*60)
    print("ComfyUI-MeshSegmenter: System Dependencies")
    print("="*60 + "\n")

    print("[Install] Checking for required OpenGL/EGL libraries...")
    print("[Install] These are needed for pyrender headless rendering to work properly.")

    try:
        critical_packages = ["libgl1", "libopengl0", "libglu1-mesa", "libglx-mesa0", "libegl1"]
        optional_packages = ["libosmesa6"]

        all_packages = critical_packages + optional_packages
        print(f"[Install] Installing OpenGL/EGL libraries: {', '.join(all_packages)}")
        print("[Install] You may be prompted for your sudo password...")

        print("[Install] Updating apt cache...")
        update_result = subprocess.run(
            ['sudo', 'apt-get', 'update'],
            capture_output=True,
            text=True,
            timeout=120
        )

        if update_result.returncode != 0:
            print("[Install] Warning: Failed to update apt cache")
            print(f"[Install] You may need to run manually: sudo apt-get update")

        installed_packages = []
        failed_packages = []
        critical_failed = []

        print("[Install] Installing critical OpenGL/EGL libraries...")
        for package in critical_packages:
            result = subprocess.run(
                ['sudo', 'apt-get', 'install', '-y', package],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                installed_packages.append(package)
                print(f"[Install]   + {package}")
            else:
                failed_packages.append(package)
                critical_failed.append(package)
                print(f"[Install]   x {package} (failed)")

        print("[Install] Installing optional OpenGL libraries...")
        for package in optional_packages:
            result = subprocess.run(
                ['sudo', 'apt-get', 'install', '-y', package],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                installed_packages.append(package)
                print(f"[Install]   + {package}")
            else:
                failed_packages.append(package)
                print(f"[Install]   ~ {package} (optional, skipped)")

        if installed_packages:
            print(f"[Install] Installed: {', '.join(installed_packages)}")

        if failed_packages:
            print(f"[Install] Failed to install: {', '.join(failed_packages)}")

        if critical_failed:
            print(f"[Install] Warning: Some packages failed to install: {', '.join(critical_failed)}")
            print(f"[Install] pyrender headless rendering may not work correctly.")
            print(f"[Install] You may need to run manually:")
            print(f"[Install]   sudo apt-get install {' '.join(critical_failed)}")
            return False
        else:
            print("[Install] OpenGL/EGL libraries installed successfully!")
            return True

    except subprocess.TimeoutExpired:
        print("[Install] Warning: Installation timed out")
        print(f"[Install] You may need to run manually:")
        print(f"[Install]   sudo apt-get install libgl1 libopengl0 libglu1-mesa libglx-mesa0 libegl1")
        return False
    except FileNotFoundError:
        print("[Install] Warning: apt-get not found (not a Debian/Ubuntu system?)")
        print("[Install] Please install OpenGL/EGL libraries manually for your distribution")
        return True
    except KeyboardInterrupt:
        print("\n[Install] Installation cancelled by user")
        return False
    except Exception as e:
        print(f"[Install] Warning: Could not install system dependencies: {e}")
        print(f"[Install] pyrender headless rendering may not work without OpenGL/EGL libraries.")
        return False
