# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
2D-to-3D label lifting utilities extracted from sam_mesh.py.

These functions handle lifting 2D per-view mask labels to consistent
3D face labels across all views.
"""

import multiprocessing as mp
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import igraph

from ..data.common import NumpyTensor


def norms_mask(norms: NumpyTensor['h w 3'], cam2world: NumpyTensor['4 4'], threshold: float = 0.0) -> NumpyTensor['h w']:
    """
    Create mask of pixels whose normals face toward the camera.

    Args:
        norms: Normal vectors per pixel (H, W, 3)
        cam2world: Camera-to-world transformation matrix (4, 4)
        threshold: Minimum dot product threshold (0 = perpendicular, 1 = facing camera)

    Returns:
        Boolean mask (H, W) where True means normal faces camera
    """
    lookat = cam2world[:3, :3] @ np.array([0, 0, 1])
    return np.abs(np.dot(norms, lookat)) > threshold


def compute_face2label(
    labels: NumpyTensor['l'],
    faceid: NumpyTensor['h w'],
    mask: NumpyTensor['h w'],
    norms: NumpyTensor['h w 3'],
    pose: NumpyTensor['4 4'],
    label_sequence_count: int,
    threshold_counts: int = 16
) -> Dict[int, Counter]:
    """
    Compute face-to-label mapping for a single view.

    For each SAM label in the view, finds which mesh faces are visible
    within that label region and counts pixel overlap.

    Args:
        labels: Array of label IDs present in this view
        faceid: Face ID render (H, W) - which face at each pixel, -1 for background
        mask: Combined mask (H, W) - label ID at each pixel
        norms: Normal vectors (H, W, 3)
        pose: Camera pose matrix (4, 4)
        label_sequence_count: Starting label ID for this view (for global uniqueness)
        threshold_counts: Minimum pixel count for face-label assignment

    Returns:
        Dict mapping face_idx -> Counter of label assignments
    """
    normal_mask = norms_mask(norms, pose)

    face2label = defaultdict(Counter)
    for j, label in enumerate(labels):
        label_sequence = label_sequence_count + j
        faces_mask = (mask == label) & normal_mask
        faces, counts = np.unique(faceid[faces_mask], return_counts=True)
        faces = faces[counts > threshold_counts]
        faces = faces[faces != -1]  # remove background
        for face in faces:
            face2label[int(face)][label_sequence] += np.sum(faces_mask & (faceid == face))
    return face2label


def compute_connections(
    i: int,
    j: int,
    face2label1: Dict[int, Counter],
    face2label2: Dict[int, Counter],
    counter_threshold: int = 32
) -> Dict[int, Dict[int, int]]:
    """
    Compute label connections between two views.

    Two labels are connected if they both see the same mesh face.

    Args:
        i: Index of first view (for logging)
        j: Index of second view (for logging)
        face2label1: Face-to-label mapping for view i
        face2label2: Face-to-label mapping for view j
        counter_threshold: Minimum overlapping faces to form connection

    Returns:
        Dict mapping label1 -> {label2: count} of connections
    """
    connections = defaultdict(Counter)
    face2label1_common = {face: counter.most_common(1)[0][0] for face, counter in face2label1.items()}
    face2label2_common = {face: counter.most_common(1)[0][0] for face, counter in face2label2.items()}

    for face1, label1 in face2label1_common.items():
        for face2, label2 in face2label2_common.items():
            if face1 != face2:
                continue
            connections[label1][label2] += 1
            connections[label2][label1] += 1

    # remove connections where # overlapping faces is below threshold
    for label1, counter in connections.items():
        connections[label1] = {k: v for k, v in counter.items() if v > counter_threshold}
    return connections


def lift_masks_to_3d(
    faces_list: List[NumpyTensor['h w']],
    cmasks_list: List[NumpyTensor['h w']],
    norms_list: List[NumpyTensor['h w 3']],
    poses_list: NumpyTensor['n 4 4'],
    face2label_threshold: int = 16,
    connections_threshold: int = 32,
    connections_bin_resolution: int = 100,
    connections_bin_threshold_percentage: float = 0.125,
    counter_lens_threshold_min: int = 16,
    num_workers: int = None
) -> Dict[int, int]:
    """
    Lift 2D per-view masks to consistent 3D face labels.

    This is the main entry point for 2D->3D label lifting.

    Args:
        faces_list: List of face ID renders per view
        cmasks_list: List of combined SAM masks per view
        norms_list: List of normal renders per view
        poses_list: Camera poses (N, 4, 4)
        face2label_threshold: Min pixels for face-label assignment
        connections_threshold: Min overlapping faces for view connection
        connections_bin_resolution: Resolution for connection ratio histogram
        connections_bin_threshold_percentage: Percentile cutoff for connections
        counter_lens_threshold_min: Minimum threshold for label filtering
        num_workers: Number of parallel workers (default: CPU count)

    Returns:
        Dict mapping face_idx -> label
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    n_views = len(faces_list)

    # Step 1: Compute face2label for each view
    print(f'    Computing face2label for each view ({num_workers} CPU cores)...')
    label_sequence_count = 1  # background is 0
    args = []
    for faceid, cmask, norms, pose in zip(faces_list, cmasks_list, norms_list, poses_list):
        labels = np.unique(cmask)
        labels = labels[labels != 0]  # remove background
        args.append((labels, faceid, cmask, norms, pose, label_sequence_count, face2label_threshold))
        label_sequence_count += len(labels)

    with mp.Pool(num_workers) as pool:
        face2label_views = pool.starmap(compute_face2label, args)

    # Step 2: Build match graph
    print(f'    Building match graph ({num_workers} CPU cores)...')
    args = []
    for i, face2label1 in enumerate(face2label_views):
        for j, face2label2 in enumerate(face2label_views):
            if i < j:
                args.append((i, j, face2label1, face2label2, connections_threshold))

    with mp.Pool(num_workers) as pool:
        partial_connections = pool.starmap(compute_connections, args)

    # Aggregate connections
    connections_ratios = defaultdict(Counter)
    for c in partial_connections:
        for label1, counter in c.items():
            connections_ratios[label1].update(counter)

    # Normalize ratios
    for label1, counter in connections_ratios.items():
        total = sum(counter.values())
        connections_ratios[label1] = {k: v / total for k, v in counter.items()}

    # Filter noisy labels
    counter_lens = [len(counter) for counter in connections_ratios.values()]
    counter_lens = sorted(counter_lens)
    if len(counter_lens) == 0:
        print(f'    Warning: No connections found between views')
        counter_lens_threshold = counter_lens_threshold_min
    else:
        counter_lens_threshold = max(np.percentile(counter_lens, 95), counter_lens_threshold_min)

    print(f'    Counter lens threshold: {counter_lens_threshold}')
    removed = []
    for label, counter in connections_ratios.items():
        if len(counter) > counter_lens_threshold:
            removed.append(label)
    for label in removed:
        connections_ratios.pop(label)
        for counter in connections_ratios.values():
            if label in counter:
                counter.pop(label)

    # Compute connection ratio threshold using histogram
    bins = np.zeros(connections_bin_resolution + 1)
    for label1, counter in connections_ratios.items():
        for label2, ratio in counter.items():
            bins[int(ratio * connections_bin_resolution)] += 1

    cutoff = connections_bin_threshold_percentage * np.sum(bins)
    accum = 0
    accum_bin = 0
    while accum < cutoff:
        accum += bins[accum_bin]
        accum_bin += 1

    # Construct match graph edges
    connections = []
    connections_ratio_threshold = max(accum_bin / connections_bin_resolution, 0.075)
    print(f'    Connections ratio threshold: {connections_ratio_threshold}')

    for label1, counter in connections_ratios.items():
        for label2, ratio12 in counter.items():
            ratio21 = connections_ratios[label2][label1]
            # best buddy match above threshold
            if ratio12 > connections_ratio_threshold and ratio21 > connections_ratio_threshold:
                connections.append((label1, label2))

    print(f'    Found {len(connections)} connections')

    # Step 3: Community detection (Leiden algorithm)
    connection_graph = igraph.Graph(edges=connections, directed=False)
    connection_graph.simplify()
    communities = connection_graph.community_leiden(resolution_parameter=0)

    label2label_consistent = {}
    comm_count = 0
    for comm in communities:
        if len(comm) > 1:
            label2label_consistent.update({label: comm[0] for label in comm if label != comm[0]})
            comm_count += 1
    print(f'    Found {comm_count} communities (segment groups)')

    # Step 4: Merge consistent labels
    print('    Merging consistent labels...')
    face2label_combined = defaultdict(Counter)
    for face2label in face2label_views:
        face2label_combined.update(face2label)

    face2label_consistent = {}
    for face, labels in face2label_combined.items():
        hook = labels.most_common(1)[0][0]
        if hook in label2label_consistent:
            hook = label2label_consistent[hook]
        face2label_consistent[face] = hook

    return face2label_consistent
