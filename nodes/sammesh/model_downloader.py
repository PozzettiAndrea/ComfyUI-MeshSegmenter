# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAM Model Loader Node - Downloads and loads SAM2 model.
"""

import os
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from .types import SAM_MODEL

try:
    import folder_paths
    sam_model_dir = os.path.join(folder_paths.models_dir, "sam")
except ImportError:
    sam_model_dir = os.path.join(os.path.expanduser("~"), ".cache", "sam_models")

os.makedirs(sam_model_dir, exist_ok=True)

# SAM2 Model Definitions
SAM_MODELS = {
    "SAM2 Hiera Large": {
        "checkpoint_filename": "sam2_hiera_large.pt",
        "config_filename": "sam2_hiera_l.yaml",
        "repo_id": "facebook/sam2-hiera-large",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_l.yaml"
    },
    "SAM2 Hiera Base+": {
        "checkpoint_filename": "sam2_hiera_base_plus.pt",
        "config_filename": "sam2_hiera_b+.yaml",
        "repo_id": "facebook/sam2-hiera-base-plus",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_b%2B.yaml"
    },
    "SAM2 Hiera Small": {
        "checkpoint_filename": "sam2_hiera_small.pt",
        "config_filename": "sam2_hiera_s.yaml",
        "repo_id": "facebook/sam2-hiera-small",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_s.yaml"
    },
    "SAM2 Hiera Tiny": {
        "checkpoint_filename": "sam2_hiera_tiny.pt",
        "config_filename": "sam2_hiera_t.yaml",
        "repo_id": "facebook/sam2-hiera-tiny",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_t.yaml"
    },
}

SAM_MODEL_NAMES = list(SAM_MODELS.keys())

# Cache for loaded models
_sam_model_cache = {}


class SamModelLoader:
    """
    Downloads and loads a SAM2 Hiera model.
    Returns the loaded model ready for use in Generate Masks node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (SAM_MODEL_NAMES, {
                    "default": SAM_MODEL_NAMES[0],
                    "tooltip": "Select the SAM2 Hiera model to download and load."
                }),
            }
        }

    RETURN_TYPES = (SAM_MODEL,)
    RETURN_NAMES = ("sam_model",)
    FUNCTION = "load_model"
    CATEGORY = "meshsegmenter/sammesh"

    def download_file(self, url, save_path, model_name):
        """Download a file from URL with progress bar."""
        try:
            print(f"SamModelDownloader ({model_name}): Downloading {os.path.basename(save_path)}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB

            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)

            print(f"SamModelDownloader ({model_name}): Download complete: {save_path}")
            return save_path
        except Exception as e:
            print(f"\033[31mError downloading {url} for {model_name}: {e}\033[0m")
            if os.path.exists(save_path):
                os.remove(save_path)
            raise

    def load_model(self, model_name: str):
        """Download (if needed) and load the specified SAM2 model."""
        if model_name not in SAM_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined.")

        # Check cache first
        if model_name in _sam_model_cache:
            print(f"SamModelLoader: Using cached model '{model_name}'")
            return (_sam_model_cache[model_name],)

        model_info = SAM_MODELS[model_name]
        checkpoint_filename = model_info["checkpoint_filename"]
        config_filename = model_info["config_filename"]
        repo_id = model_info["repo_id"]
        config_url = model_info["config_url"]

        checkpoint_path = os.path.join(sam_model_dir, checkpoint_filename)
        config_path = os.path.join(sam_model_dir, config_filename)

        # Download Checkpoint if missing
        if not os.path.exists(checkpoint_path):
            print(f"SamModelLoader ({model_name}): Checkpoint not found. Downloading...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=checkpoint_filename,
                    local_dir=sam_model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"SamModelLoader ({model_name}): Checkpoint downloaded to {checkpoint_path}")
            except Exception as e:
                print(f"\033[31mError downloading checkpoint for {model_name}: {e}\033[0m")
                raise
        else:
            print(f"SamModelLoader ({model_name}): Checkpoint found: {checkpoint_path}")

        # Download Config YAML if missing
        if not os.path.exists(config_path):
            print(f"SamModelLoader ({model_name}): Config not found. Downloading...")
            self.download_file(config_url, config_path, model_name)
        else:
            print(f"SamModelLoader ({model_name}): Config found: {config_path}")

        # Verify files exist
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Failed to locate checkpoint: {checkpoint_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Failed to locate config: {config_path}")

        # Load the model
        print(f"SamModelLoader ({model_name}): Loading model...")

        # Default engine config - can be adjusted in GenerateMasks node
        engine_config = {
            "points_per_side": 32,
            "crop_n_layers": 0,
            "pred_iou_thresh": 0.5,
            "stability_score_thresh": 0.7,
            "stability_score_offset": 1.0,
        }

        config = OmegaConf.create({
            "sam": {
                "checkpoint": checkpoint_path,
                "model_config": os.path.basename(config_path),
                "auto": True,
                "ground": False,
                "engine_config": engine_config,
            }
        })

        from ...samesh.models.sam import Sam2Model
        model = Sam2Model(config, device='cuda')

        # Cache the loaded model
        _sam_model_cache[model_name] = model
        print(f"SamModelLoader ({model_name}): Model loaded and cached!")

        return (model,)
