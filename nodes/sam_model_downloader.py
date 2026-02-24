# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAM Model Loader Node - Downloads SAM2/SAM3 checkpoints and returns
a serializable config dict for downstream nodes to instantiate.
"""

import logging
import os

from huggingface_hub import hf_hub_download
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


class SamModelLoader:
    """
    Downloads a SAM2 or SAM3 checkpoint and returns a serializable config dict.
    The actual model is instantiated by downstream nodes (e.g. GenerateMasks).
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

    def load_model(self, model_name: str, precision: str = "auto"):
        """Download (if needed) the checkpoint and return a serializable config dict."""
        if model_name not in SAM_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined.")

        model_info = SAM_MODELS[model_name]
        model_type = model_info.get("type", "sam2")
        repo_id = model_info["repo_id"]
        checkpoint_filename = model_info["checkpoint_filename"]

        checkpoint_path = self._download_checkpoint(model_name, repo_id, checkpoint_filename)

        sam_config = {
            "type": model_type,
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "precision": precision,
        }

        if model_type == "sam2":
            sam_config["config_name"] = model_info["config_name"]
            sam_config["engine_config"] = {
                "points_per_side": 32,
                "crop_n_layers": 0,
                "pred_iou_thresh": 0.5,
                "stability_score_thresh": 0.7,
                "stability_score_offset": 1.0,
            }
        else:  # sam3
            sam_config["engine_config"] = {
                "points_per_side": 32,
                "pred_iou_thresh": 0.5,
                "stability_score_thresh": 0.7,
                "stability_score_offset": 1.0,
                "min_mask_region_area": 100,
                "box_nms_thresh": 0.7,
            }

        log.info("SamModelLoader (%s): Config prepared, checkpoint at %s", model_name, checkpoint_path)
        return (sam_config,)


NODE_CLASS_MAPPINGS = {
    "MeshSegSamModelLoader": SamModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshSegSamModelLoader": "SAM Model Loader (MeshSeg)",
}
