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
import networkx as nx
import igraph
from tqdm import tqdm


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
    smoothing_iterations: int = 64,
    area_faces: np.ndarray = None,
    num_faces: int = None,
) -> Dict[int, int]:
    """
    Smooth labels by removing small components and filling holes.

    Args:
        face2label: Dict mapping face_idx -> label
        mesh: Trimesh for area calculations (ignored if area_faces provided)
        mesh_graph: Face adjacency graph (computed if not provided)
        threshold_percentage_size: Remove components smaller than this % of largest
        threshold_percentage_area: Remove components with less than this % of largest area
        smoothing_iterations: Number of hole-filling iterations
        area_faces: Optional per-face area array (for quad mesh support)
        num_faces: Optional override for number of faces (for quad mesh support)

    Returns:
        Smoothed face2label dict
    """
    if mesh_graph is None:
        mesh_graph = build_mesh_graph(mesh)

    if num_faces is None:
        num_faces = len(mesh.faces)
    if area_faces is None:
        area_faces = mesh.area_faces

    face2label = dict(face2label)  # copy to avoid mutation

    # Find connected components
    components = label_components(face2label, mesh_graph, num_faces)

    if not components:
        print('    Warning: No components found to smooth')
        return face2label

    # Calculate component sizes and areas
    components = sorted(components, key=lambda x: len(x), reverse=True)
    components_area = [
        sum([float(area_faces[face]) for face in comp if face < len(area_faces)]) for comp in components
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


# ============================================================================
# Alpha-Expansion Graph Cut Repartitioning
# ============================================================================

EPSILON = 1e-20


def partition_cost(
    mesh: trimesh.Trimesh,
    partition: np.ndarray,
    cost_data: np.ndarray,
    cost_smoothness: np.ndarray
) -> float:
    """
    Compute total cost of a partition (data term + smoothness term).

    Args:
        mesh: Trimesh object
        partition: Array of label per face
        cost_data: (num_faces, num_labels) data cost matrix
        cost_smoothness: (num_edges,) smoothness cost per edge

    Returns:
        Total cost
    """
    cost = 0
    for f in range(len(partition)):
        cost += cost_data[f, partition[f]]
    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            cost += cost_smoothness[i]
    return cost


def construct_expansion_graph(
    label: int,
    mesh: trimesh.Trimesh,
    partition: np.ndarray,
    cost_data: np.ndarray,
    cost_smoothness: np.ndarray
) -> Tuple[nx.Graph, Dict]:
    """
    Construct the expansion graph for alpha-expansion move.

    Args:
        label: The label to expand
        mesh: Trimesh object
        partition: Current partition
        cost_data: Data cost matrix
        cost_smoothness: Smoothness cost per edge

    Returns:
        (graph, node2index) tuple
    """
    G = nx.Graph()
    A = 'alpha'
    B = 'alpha_complement'

    node2index = {}
    G.add_node(A)
    G.add_node(B)
    node2index[A] = 0
    node2index[B] = 1
    for i in range(len(mesh.faces)):
        G.add_node(i)
        node2index[i] = 2 + i

    aux_count = 0
    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            a = (f1, f2)
            if a in node2index:
                continue
            G.add_node(a)
            node2index[a] = len(mesh.faces) + 2 + aux_count
            aux_count += 1

    # Note: Use small epsilon instead of 0 to avoid igraph treating 0 as unweighted (default 1)
    CAPACITY_EPS = 1e-6
    for f in range(len(mesh.faces)):
        cap_A = max(cost_data[f, label], CAPACITY_EPS)
        cap_B = float('inf') if partition[f] == label else max(cost_data[f, partition[f]], CAPACITY_EPS)
        G.add_edge(A, f, capacity=cap_A)
        G.add_edge(B, f, capacity=cap_B)

    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        a = (f1, f2)
        if partition[f1] == partition[f2]:
            if partition[f1] != label:
                G.add_edge(f1, f2, capacity=cost_smoothness[i])
        else:
            G.add_edge(a, B, capacity=cost_smoothness[i])
            if partition[f1] != label:
                G.add_edge(f1, a, capacity=cost_smoothness[i])
            if partition[f2] != label:
                G.add_edge(a, f2, capacity=cost_smoothness[i])

    return G, node2index


def repartition(
    mesh: trimesh.Trimesh,
    partition: np.ndarray,
    cost_data: np.ndarray,
    cost_smoothness: np.ndarray,
    smoothing_iterations: int,
    _lambda: float = 1.0,
) -> np.ndarray:
    """
    Refine partition using alpha-expansion graph cuts.

    Uses min-cut/max-flow to iteratively expand each label,
    optimizing the energy function (data cost + smoothness cost).

    Args:
        mesh: Trimesh object
        partition: Initial partition (label per face)
        cost_data: (num_faces, num_labels) data cost
        cost_smoothness: (num_edges,) smoothness cost (based on dihedral angles)
        smoothing_iterations: Number of expansion iterations
        _lambda: Weight for smoothness term

    Returns:
        Refined partition
    """
    A = 'alpha'
    B = 'alpha_complement'
    labels = np.unique(partition)

    cost_smoothness = cost_smoothness * _lambda
    cost_min = partition_cost(mesh, partition, cost_data, cost_smoothness)

    for iteration in range(smoothing_iterations):
        for label in tqdm(labels, desc=f'    Repartition iter {iteration+1}', leave=False):
            G, node2index = construct_expansion_graph(label, mesh, partition, cost_data, cost_smoothness)
            index2node = {v: k for k, v in node2index.items()}

            # Use igraph for min-cut (faster than networkx)
            G_ig = igraph.Graph.from_networkx(G)
            outputs = G_ig.st_mincut(source=node2index[A], target=node2index[B], capacity='capacity')
            S = outputs.partition[0]
            T = outputs.partition[1]

            assert node2index[A] in S and node2index[B] in T
            S = np.array([index2node[v] for v in S if isinstance(index2node[v], int)]).astype(int)
            T = np.array([index2node[v] for v in T if isinstance(index2node[v], int)]).astype(int)

            # T consists of those assigned 'alpha' and S 'alpha_complement' (see paper)
            if len(T) > 0:
                partition[T] = label

            cost = partition_cost(mesh, partition, cost_data, cost_smoothness)
            if cost > cost_min + EPSILON:
                print(f'    Warning: Cost increased ({cost_min:.2f} -> {cost:.2f})')
            cost_min = min(cost, cost_min)

    return partition


def graph_cut_repartition(
    face2label: Dict[int, int],
    mesh: trimesh.Trimesh,
    repartition_lambda: float = 6.0,
    repartition_iterations: int = 1,
    target_labels: int = None,
    lambda_range: Tuple[float, float] = (1.0, 15.0),
    tolerance: int = 1,
    noise_threshold: int = 10,
) -> Dict[int, int]:
    """
    Refine face labels using alpha-expansion graph cuts.

    This smooths segmentation boundaries by optimizing an energy function
    that balances data fidelity (keeping original labels) with smoothness
    (preferring cuts at sharp dihedral angles).

    Args:
        face2label: Dict mapping face_idx -> label
        mesh: Trimesh object
        repartition_lambda: Weight for smoothness term (higher = smoother boundaries)
        repartition_iterations: Number of graph cut iterations
        target_labels: If set, auto-tune lambda to achieve this many segments
        lambda_range: (min, max) range for lambda search when using target_labels
        tolerance: Acceptable deviation from target_labels
        noise_threshold: Minimum faces for a segment to be counted in target mode

    Returns:
        Refined face2label dict
    """
    import multiprocessing as mp

    num_faces = len(mesh.faces)

    # Convert face2label to partition array
    partition = np.zeros(num_faces, dtype=np.int32)
    for face, label in face2label.items():
        partition[int(face)] = label

    max_label = int(partition.max())
    unique_labels = np.unique(partition)

    # Build cost matrices (matching original SAMesh exactly)
    # Data cost: 0 if face keeps its label, 1 otherwise (only for existing labels)
    cost_data = np.zeros((num_faces, max_label + 1), dtype=np.float32)
    for f in range(num_faces):
        for l in unique_labels:
            cost_data[f, l] = 0.0 if partition[f] == l else 1.0

    # Smoothness cost: based on dihedral angles
    # Sharp angles (small angle) = high cost to cut there
    # Smooth angles (large angle) = low cost to cut there
    angles = mesh.face_adjacency_angles  # radians, 0 = flat, pi = sharp
    cost_smoothness = -np.log(angles / np.pi + EPSILON)

    if target_labels is None:
        # Standard mode: use fixed lambda
        print(f'    Running graph cut repartition (lambda={repartition_lambda}, iters={repartition_iterations})...')
        partition = repartition(
            mesh, partition, cost_data, cost_smoothness,
            smoothing_iterations=repartition_iterations,
            _lambda=repartition_lambda
        )
    else:
        # Target labels mode: search for lambda that gives target segment count
        print(f'    Running graph cut with target_labels={target_labels} (tolerance={tolerance})...')

        def count_labels(part):
            """Count labels, ignoring tiny segments."""
            values, counts = np.unique(part, return_counts=True)
            return len(values[counts > noise_threshold])

        # Try multiple lambdas in parallel
        num_samples = min(mp.cpu_count(), 8)
        lambdas = np.linspace(lambda_range[0], lambda_range[1], num=num_samples)

        print(f'    Searching lambda in range [{lambda_range[0]}, {lambda_range[1]}] with {num_samples} samples...')

        # Run repartition for each lambda
        results = []
        for _lambda in lambdas:
            part_copy = partition.copy()
            refined = repartition(
                mesh, part_copy, cost_data, cost_smoothness,
                smoothing_iterations=repartition_iterations,
                _lambda=_lambda
            )
            n_labels = count_labels(refined)
            results.append((refined, n_labels, _lambda))
            print(f'      lambda={_lambda:.2f} -> {n_labels} labels')

        # Find best match
        best_partition = None
        best_diff = float('inf')
        best_lambda = repartition_lambda

        for refined, n_labels, _lambda in results:
            diff = abs(n_labels - target_labels)
            if diff < best_diff:
                best_diff = diff
                best_partition = refined
                best_lambda = _lambda

        partition = best_partition
        print(f'    Selected lambda={best_lambda:.2f} with {count_labels(partition)} labels (target={target_labels})')

        if best_diff > tolerance:
            print(f'    Warning: Could not achieve target within tolerance (diff={best_diff})')

    # Convert back to dict
    return {int(f): int(partition[f]) for f in range(num_faces)}
