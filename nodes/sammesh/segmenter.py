# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh Segmenter Node - Segments mesh using SAM2.
"""

import os
import sys
import json
import subprocess
import trimesh

try:
    import folder_paths
    output_dir = folder_paths.get_output_directory()
    temp_dir = folder_paths.get_temp_directory()
except ImportError:
    output_dir = os.path.join(os.getcwd(), "output")
    temp_dir = os.path.join(os.getcwd(), "temp")

DEFAULT_OUTPUT_DIR = os.path.join(output_dir, "meshsegmenter")
DEFAULT_CACHE_DIR = os.path.join(temp_dir, "meshsegmenter_cache")

# Path to the worker script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up to package root
WORKER_SCRIPT_PATH = os.path.join(NODE_DIR, "_run_segmentation_worker.py")


class SamMeshSegmenter:
    """
    Segments a mesh using the SAMesh model (SAM2-based mesh segmentation).
    Runs segmentation in a subprocess and returns the segmented mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "mesh_path": ("STRING", {"forceInput": True}),
                "sam_checkpoint_path": ("STRING", {"forceInput": True}),
                "sam_model_config_path": ("STRING", {"forceInput": True}),
                "output_directory": ("STRING", {
                    "default": DEFAULT_OUTPUT_DIR,
                    "tooltip": "Directory to save output files."
                }),
                "cache_directory": ("STRING", {
                    "default": DEFAULT_CACHE_DIR,
                    "tooltip": "Directory for caching intermediate results."
                }),
                "keep_texture": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve original mesh texture during segmentation."
                }),
            },
            "optional": {
                "target_labels": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Desired number of segments. -1 for automatic."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible results."
                }),
            }
        }

    RETURN_TYPES = ("MESH", "STRING",)
    RETURN_NAMES = ("segmented_mesh", "face2label_path",)
    FUNCTION = "segment_mesh"
    CATEGORY = "meshsegmenter/sammesh"

    def segment_mesh(
        self,
        mesh: trimesh.Trimesh,
        mesh_path: str,
        sam_checkpoint_path: str,
        sam_model_config_path: str,
        output_directory: str,
        cache_directory: str,
        keep_texture: bool,
        target_labels: int = -1,
        seed: int = 0
    ):
        # Input validation
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Warning: Input 'mesh' is not a Trimesh object (got {type(mesh)})")

        mesh_path = os.path.abspath(mesh_path)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Original mesh file not found: {mesh_path}")

        if not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint_path}")
        if not os.path.exists(sam_model_config_path):
            raise FileNotFoundError(f"SAM config not found: {sam_model_config_path}")

        if not os.path.exists(WORKER_SCRIPT_PATH):
            raise FileNotFoundError(f"Worker script not found: {WORKER_SCRIPT_PATH}")

        print(f"SamMeshSegmenter: Starting segmentation for: {mesh_path}")

        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(cache_directory, exist_ok=True)

        # Cap seed to valid range for numpy
        capped_seed = seed % (2**32)

        # Build command
        cmd = [
            sys.executable,
            WORKER_SCRIPT_PATH,
            "--mesh_path", mesh_path,
            "--sam_checkpoint_path", sam_checkpoint_path,
            "--sam_model_config_path", sam_model_config_path,
            "--output_directory", output_directory,
            "--cache_directory", cache_directory,
            "--output_filename_prefix", "segmented",
            "--output_extension", "glb",
            "--target_labels", str(target_labels),
            "--seed", str(capped_seed),
        ]
        if keep_texture:
            cmd.append("--keep_texture")

        # Prepare environment
        env = os.environ.copy()
        try:
            env['COMFYUI_BASE_PATH'] = folder_paths.base_path
        except:
            env['COMFYUI_BASE_PATH'] = ''

        # Set working directory for worker
        samesh_cwd = os.path.join(NODE_DIR, "samesh-main", "src")
        if not os.path.exists(samesh_cwd):
            samesh_cwd = NODE_DIR

        print(f"SamMeshSegmenter: Executing worker: {' '.join(cmd)}")

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=samesh_cwd
            )
            print(f"SamMeshSegmenter: Worker stdout:\n{process.stdout}")
            if process.stderr:
                print(f"SamMeshSegmenter: Worker stderr:\n{process.stderr}")

            # Parse result from stdout
            last_line = process.stdout.strip().splitlines()[-1]
            result_data = json.loads(last_line)
            final_output_mesh_path = result_data.get("output_mesh_path")
            final_output_json_path = result_data.get("face2label_path")

            if not final_output_mesh_path or "Error:" in final_output_mesh_path:
                raise RuntimeError(f"Worker failed to produce mesh: {final_output_mesh_path}")
            if not final_output_json_path or "Error:" in final_output_json_path:
                raise RuntimeError(f"Worker failed to produce JSON: {final_output_json_path}")

            print(f"SamMeshSegmenter: Loading result mesh from {final_output_mesh_path}")

            if not os.path.exists(final_output_mesh_path):
                raise FileNotFoundError(f"Segmented mesh not found: {final_output_mesh_path}")

            segmented_mesh = trimesh.load(final_output_mesh_path, force='mesh')
            if not isinstance(segmented_mesh, trimesh.Trimesh):
                raise TypeError(f"Loaded result is not a Trimesh: {type(segmented_mesh)}")

            return (segmented_mesh, final_output_json_path,)

        except subprocess.CalledProcessError as e:
            print(f"\033[31mError: Worker script failed (code {e.returncode})\033[0m")
            print(f"\033[31mStderr:\n{e.stderr}\033[0m")
            print(f"\033[31mStdout:\n{e.stdout}\033[0m")
            raise RuntimeError("Segmentation worker failed. Check logs.") from e
        except json.JSONDecodeError as e:
            print(f"\033[31mError: Could not parse JSON result from worker\033[0m")
            raise RuntimeError("Failed to parse worker result.") from e
        except Exception as e:
            print(f"\033[31mError during segmentation: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise
