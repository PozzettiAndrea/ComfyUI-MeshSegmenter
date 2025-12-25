# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Label smoothing utilities extracted from sam_mesh.py.

These functions handle smoothing, hole filling, component splitting,
and graph-cut optimization of face labels.
"""

from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import numpy as np
import trimesh


def build_mesh_graph(mesh: trimesh.Trimesh) -> Dict[int, Set[int]]:
    """
    Build face adjacency graph from mesh.

    Args:
        mesh: Input trimesh

    Returns:
        Dict mapping face_idx -> set of adjacent face indices
    """
    mesh_edges = trimesh.graph.face_adjacency(mesh=mesh)
    mesh_graph = defaultdict(set)
    for face1, face2 in mesh_edges:
        mesh_graph[face1].add(face2)
        mesh_graph[face2].add(face1)
    return mesh_graph


def label_components(
    face2label: Dict[int, int],
    mesh_graph: Dict[int, Set[int]],
    num_faces: int
) -> List[Set[int]]:
    """
    Find connected components where adjacent faces share the same label.

    Args:
        face2label: Dict mapping face_idx -> label
        mesh_graph: Face adjacency graph
        num_faces: Total number of faces in mesh

    Returns:
        List of sets, each set containing face indices of one component
    """
    components = []
    visited = set()

    def dfs(source: int):
        stack = [source]
        components.append({source})
        visited.add(source)

        while stack:
            node = stack.pop()
            for adj in mesh_graph[node]:
                if adj not in visited and adj in face2label and face2label[adj] == face2label[node]:
                    stack.append(adj)
                    components[-1].add(adj)
                    visited.add(adj)

    for face in range(num_faces):
        if face not in visited and face in face2label:
            dfs(face)

    return components


def smooth_labels(
    face2label: Dict[int, int],
    mesh: trimesh.Trimesh,
    mesh_graph: Dict[int, Set[int]] = None,
    threshold_percentage_size: float = 0.025,
    threshold_percentage_area: float = 0.025,
    smoothing_iterations: int = 64
) -> Dict[int, int]:
    """
    Smooth labels by removing small components and filling holes.

    Args:
        face2label: Dict mapping face_idx -> label
        mesh: Trimesh for area calculations
        mesh_graph: Face adjacency graph (computed if not provided)
        threshold_percentage_size: Remove components smaller than this % of largest
        threshold_percentage_area: Remove components with less than this % of largest area
        smoothing_iterations: Number of hole-filling iterations

    Returns:
        Smoothed face2label dict
    """
    if mesh_graph is None:
        mesh_graph = build_mesh_graph(mesh)

    num_faces = len(mesh.faces)
    face2label = dict(face2label)  # copy to avoid mutation

    # Find connected components
    components = label_components(face2label, mesh_graph, num_faces)

    if not components:
        print('    Warning: No components found to smooth')
        return face2label

    # Calculate component sizes and areas
    components = sorted(components, key=lambda x: len(x), reverse=True)
    components_area = [
        sum([float(mesh.area_faces[face]) for face in comp]) for comp in components
    ]
    max_size = max([len(comp) for comp in components])
    max_area = max(components_area)

    # Find components to remove (must be small in BOTH size and area)
    remove_comp_size = set()
    remove_comp_area = set()
    for i, comp in enumerate(components):
        if len(comp) < max_size * threshold_percentage_size:
            remove_comp_size.add(i)
        if components_area[i] < max_area * threshold_percentage_area:
            remove_comp_area.add(i)
    remove_comp = remove_comp_size.intersection(remove_comp_area)

    print(f'    Removing {len(remove_comp)} small components')
    for i in remove_comp:
        for face in components[i]:
            face2label.pop(face, None)

    # Fill holes by propagating labels from neighbors
    print(f'    Filling holes ({smoothing_iterations} iterations)...')
    for iteration in range(smoothing_iterations):
        count = 0
        changes = {}
        for face in range(num_faces):
            if face in face2label:
                continue
            labels_adj = Counter()
            for adj in mesh_graph[face]:
                if adj in face2label:
                    label = face2label[adj]
                    if label != 0:
                        labels_adj[label] += 1
            if len(labels_adj):
                count += 1
                changes[face] = labels_adj.most_common(1)[0][0]
        for face, label in changes.items():
            face2label[face] = label

        if count == 0:
            break  # converged

    return face2label


def split_disconnected_components(
    face2label: Dict[int, int],
    mesh_graph: Dict[int, Set[int]],
    num_faces: int
) -> Dict[int, int]:
    """
    Split disconnected components that share the same label into separate labels.

    If two regions have the same label but aren't connected, give them different labels.

    Args:
        face2label: Dict mapping face_idx -> label
        mesh_graph: Face adjacency graph
        num_faces: Total number of faces

    Returns:
        Updated face2label with split components
    """
    face2label = dict(face2label)  # copy

    components = label_components(face2label, mesh_graph, num_faces)

    labels_seen = set()
    labels_curr = max(face2label.values()) + 1 if face2label else 1
    labels_orig = labels_curr

    for comp in components:
        face = next(iter(comp))
        label = face2label[face]
        if label == 0 or label in labels_seen:
            # background or repeated label -> assign new label
            for f in comp:
                face2label[f] = labels_curr
            labels_curr += 1
        labels_seen.add(label)

    print(f'    Split into {labels_curr - labels_orig} additional components')
    return face2label


def fill_unlabeled_faces(
    face2label: Dict[int, int],
    num_faces: int,
    unlabeled_value: int = 0
) -> Dict[int, int]:
    """
    Fill any unlabeled faces with a default value.

    Args:
        face2label: Dict mapping face_idx -> label
        num_faces: Total number of faces
        unlabeled_value: Value to assign to unlabeled faces

    Returns:
        face2label with all faces assigned
    """
    face2label = dict(face2label)
    unlabeled_count = 0

    for face in range(num_faces):
        if face not in face2label:
            face2label[face] = unlabeled_value
            unlabeled_count += 1

    if unlabeled_count > 0:
        print(f'    Marked {unlabeled_count} unlabeled faces')

    return face2label


def compute_label_stats(face2label: Dict[int, int]) -> Dict[int, int]:
    """
    Compute statistics about label distribution.

    Args:
        face2label: Dict mapping face_idx -> label

    Returns:
        Dict mapping label -> face count
    """
    stats = Counter(face2label.values())
    return dict(stats)
