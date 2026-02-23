# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAM Model Loader Node - Downloads and loads SAM2/SAM3 models.
"""

import logging
import os

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from folder_paths import base_path as comfy_base_path

from .types import SAM_MODEL

log = logging.getLogger("meshsegmenter")

sam_model_dir = os.path.join(comfy_base_path, "models", "sam")
os.makedirs(sam_model_dir, exist_ok=True)

# SAM Model Definitions
SAM_MODELS = {
    # SAM2.1 models (upgraded from SAM2)
    # config_name matches HF_MODEL_ID_TO_FILENAMES in build_sam.py
    "SAM2.1 Hiera Large": {
        "type": "sam2",
        "checkpoint_filename": "sam2.1_hiera_large.pt",
        "config_name": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "repo_id": "facebook/sam2.1-hiera-large",
    },
    "SAM2.1 Hiera Base+": {
        "type": "sam2",
        "checkpoint_filename": "sam2.1_hiera_base_plus.pt",
        "config_name": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "repo_id": "facebook/sam2.1-hiera-base-plus",
    },
    "SAM2.1 Hiera Small": {
        "type": "sam2",
        "checkpoint_filename": "sam2.1_hiera_small.pt",
        "config_name": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "repo_id": "facebook/sam2.1-hiera-small",
    },
    "SAM2.1 Hiera Tiny": {
        "type": "sam2",
        "checkpoint_filename": "sam2.1_hiera_tiny.pt",
        "config_name": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "repo_id": "facebook/sam2.1-hiera-tiny",
    },
    # SAM3 model
    "SAM3": {
        "type": "sam3",
        "checkpoint_filename": "sam3.pt",
        "repo_id": "1038lab/sam3",
    },
}

SAM_MODEL_NAMES = list(SAM_MODELS.keys())

# Cache for loaded models
_sam_model_cache = {}


class SamModelLoader:
    """
    Downloads and loads a SAM2 or SAM3 model.
    Returns the loaded model ready for use in Generate Masks node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (SAM_MODEL_NAMES, {
                    "default": SAM_MODEL_NAMES[0],
                    "tooltip": "Select the SAM model to download and load. SAM2 variants or SAM3."
                }),
            },
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
            },
        }

    RETURN_TYPES = (SAM_MODEL,)
    RETURN_NAMES = ("sam_model",)
    FUNCTION = "load_model"
    CATEGORY = "meshsegmenter/sammesh"

    def _resolve_dtype(self, precision, device):
        import comfy.model_management
        if precision == "auto":
            if comfy.model_management.should_use_bf16(device):
                return torch.bfloat16
            elif comfy.model_management.should_use_fp16(device):
                return torch.float16
            else:
                return torch.float32
        elif precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        else:
            return torch.float32

    def load_model(self, model_name: str, precision: str = "auto"):
        """Download (if needed) and load the specified SAM model."""
        import comfy.model_management

        if model_name not in SAM_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined.")

        # Check cache first
        if model_name in _sam_model_cache:
            log.info("SamModelLoader: Using cached model '%s'", model_name)
            return (_sam_model_cache[model_name],)

        load_device = comfy.model_management.get_torch_device()
        dtype = self._resolve_dtype(precision, load_device)
        device_str = str(load_device)

        model_info = SAM_MODELS[model_name]
        model_type = model_info.get("type", "sam2")

        if model_type == "sam3":
            model = self._load_sam3(model_name, model_info, device_str)
        else:
            model = self._load_sam2(model_name, model_info, device_str)

        # Cache the loaded model
        _sam_model_cache[model_name] = model
        log.info("SamModelLoader (%s): Model loaded and cached on %s (%s)", model_name, device_str, dtype)

        return (model,)

    def _download_checkpoint(self, model_name, repo_id, checkpoint_filename):
        checkpoint_path = os.path.join(sam_model_dir, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            log.info("SamModelLoader (%s): Checkpoint not found. Downloading from %s...", model_name, repo_id)
            hf_hub_download(
                repo_id=repo_id,
                filename=checkpoint_filename,
                local_dir=sam_model_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            log.info("SamModelLoader (%s): Checkpoint downloaded to %s", model_name, checkpoint_path)
        else:
            log.info("SamModelLoader (%s): Checkpoint found: %s", model_name, checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Failed to locate checkpoint: {checkpoint_path}")

        return checkpoint_path

    def _load_sam2(self, model_name: str, model_info: dict, device: str):
        """Load a SAM2/SAM2.1 model."""
        checkpoint_filename = model_info["checkpoint_filename"]
        config_name = model_info["config_name"]
        repo_id = model_info["repo_id"]

        checkpoint_path = self._download_checkpoint(model_name, repo_id, checkpoint_filename)

        log.info("SamModelLoader (%s): Loading SAM2 model with config '%s'...", model_name, config_name)

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
                "model_config": config_name,
                "auto": True,
                "ground": False,
                "engine_config": engine_config,
            }
        })

        from ..samesh.models.sam import Sam2Model
        model = Sam2Model(config, device=device)

        return model

    def _load_sam3(self, model_name: str, model_info: dict, device: str):
        """Load a SAM3 model."""
        checkpoint_filename = model_info["checkpoint_filename"]
        repo_id = model_info["repo_id"]

        checkpoint_path = self._download_checkpoint(model_name, repo_id, checkpoint_filename)

        log.info("SamModelLoader (%s): Loading SAM3 model...", model_name)

        # Default engine config
        engine_config = {
            "points_per_side": 32,
            "pred_iou_thresh": 0.5,
            "stability_score_thresh": 0.7,
            "stability_score_offset": 1.0,
            "min_mask_region_area": 100,
            "box_nms_thresh": 0.7,
        }

        config = OmegaConf.create({
            "sam3": {
                "checkpoint": checkpoint_path,
                "engine_config": engine_config,
            }
        })

        from ..samesh.models.sam3 import Sam3Model
        model = Sam3Model(config, device=device)

        return model


NODE_CLASS_MAPPINGS = {
    "MeshSegSamModelLoader": SamModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshSegSamModelLoader": "SAM Model Loader (MeshSeg)",
}
