# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Lift 2D to 3D Labels Node - Converts per-view 2D masks to consistent 3D face labels.
"""

from .types import MESH_RENDER_DATA, VIEW_MASKS, FACE_LABELS
from ...samesh.models.lifting import lift_masks_to_3d


class Lift2DTo3DLabels:
    """
    Lifts 2D per-view mask labels to consistent 3D mesh face labels.

    This node:
    1. Computes face-to-label mapping for each view
    2. Builds a match graph connecting labels across views
    3. Uses community detection to find consistent label groups
    4. Outputs a single face-to-label mapping for the entire mesh
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "render_data": (MESH_RENDER_DATA,),
                "combined_masks": (VIEW_MASKS,),
            },
            "optional": {
                "connections_threshold": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Minimum overlapping faces to form a connection between views."
                }),
                "face2label_threshold": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Minimum pixel count for a face to be assigned a label."
                }),
            }
        }

    RETURN_TYPES = (FACE_LABELS,)
    RETURN_NAMES = ("face_labels",)
    FUNCTION = "lift_labels"
    CATEGORY = "meshsegmenter/sammesh"

    def lift_labels(
        self,
        mesh,
        render_data: dict,
        combined_masks: dict,
        connections_threshold: int = 8,
        face2label_threshold: int = 4
    ):
        print("Lift2DTo3DLabels: Lifting 2D masks to 3D face labels...")

        # Extract data
        faces_list = render_data['faces']
        norms_list = render_data['norms']
        poses_list = render_data['poses']
        cmasks_list = combined_masks['cmasks']

        num_faces = len(mesh.faces)
        print(f"  Mesh has {num_faces} faces")
        print(f"  Processing {len(faces_list)} views")

        # Run lifting algorithm
        face2label = lift_masks_to_3d(
            faces_list=faces_list,
            cmasks_list=cmasks_list,
            norms_list=norms_list,
            poses_list=poses_list,
            face2label_threshold=face2label_threshold,
            connections_threshold=connections_threshold,
        )

        # Build FACE_LABELS output
        label_counts = {}
        for label in face2label.values():
            label_counts[label] = label_counts.get(label, 0) + 1

        face_labels_data = {
            'face2label': face2label,
            'num_faces': num_faces,
            'label_stats': label_counts,
        }

        labeled_count = len(face2label)
        unique_labels = len(set(face2label.values()))
        print(f"  Labeled {labeled_count}/{num_faces} faces ({100*labeled_count/num_faces:.1f}%)")
        print(f"  Found {unique_labels} unique labels")

        return (face_labels_data,)
