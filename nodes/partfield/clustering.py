# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Face adjacency matrix construction for mesh clustering.

Extracted from partfield's run_part_clustering.py — only the 3 adjacency
builders + UnionFind helper used by PartFieldSegmenter.
"""

import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
import networkx as nx


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


def construct_face_adjacency_matrix_naive(face_list):
    """
    Construct face adjacency matrix from shared edges.
    If multiple components exist, connect them via naive chain of representatives.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        return csr_matrix((0, 0))

    edge_to_faces = defaultdict(list)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for e, faces_sharing_e in edge_to_faces.items():
        f_indices = list(set(faces_sharing_e))
        if len(f_indices) > 1:
            for i in range(len(f_indices)):
                for j in range(i + 1, len(f_indices)):
                    f_i = f_indices[i]
                    f_j = f_indices[j]
                    row.append(f_i)
                    col.append(f_j)
                    row.append(f_j)
                    col.append(f_i)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)),
        shape=(num_faces, num_faces)
    ).tocsr()

    n_components, labels = connected_components(face_adjacency, directed=False)

    if n_components > 1:
        component_representatives = []
        for comp_id in range(n_components):
            faces_in_comp = np.where(labels == comp_id)[0]
            if len(faces_in_comp) > 0:
                component_representatives.append(faces_in_comp[0])

        dummy_row = []
        dummy_col = []
        for i in range(len(component_representatives) - 1):
            f_i = component_representatives[i]
            f_j = component_representatives[i + 1]
            dummy_row.extend([f_i, f_j])
            dummy_col.extend([f_j, f_i])

        if dummy_row:
            dummy_data = np.ones(len(dummy_row), dtype=np.int8)
            dummy_mat = coo_matrix(
                (dummy_data, (dummy_row, dummy_col)),
                shape=(num_faces, num_faces)
            ).tocsr()
            face_adjacency = face_adjacency + dummy_mat

    return face_adjacency


def construct_face_adjacency_matrix_facemst(face_list, vertices, k=10, with_knn=True):
    """
    Construct face adjacency from shared edges, then connect components via
    face-centroid KNN + MST.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        return csr_matrix((0, 0))

    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1

    if n_components == 1:
        return face_adjacency

    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(face_centroids)
    distances, indices = knn.kneighbors(face_centroids)

    G = nx.Graph()
    G.add_nodes_from(range(num_faces))

    for i in range(num_faces):
        for j, dist in zip(indices[i], distances[i]):
            if i == j:
                continue
            G.add_edge(i, j, weight=dist)

    mst = nx.minimum_spanning_tree(G, weight='weight')
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )

    adjacency_lil = face_adjacency.tolil()

    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    face_adjacency = adjacency_lil.tocsr()

    if with_knn:
        dummy_row = []
        dummy_col = []
        for i in range(num_faces):
            for j in indices[i]:
                dummy_row.extend([i, j])
                dummy_col.extend([j, i])

        dummy_data = np.ones(len(dummy_row), dtype=np.int16)
        dummy_mat = coo_matrix(
            (dummy_data, (dummy_row, dummy_col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        face_adjacency = face_adjacency + dummy_mat

    return face_adjacency


def construct_face_adjacency_matrix_ccmst(face_list, vertices, k=10, with_knn=True):
    """
    Construct face adjacency from shared edges, then connect components via
    component-centroid KNN + MST (more efficient than face-level MST).
    """
    num_faces = len(face_list)
    if num_faces == 0:
        return csr_matrix((0, 0))

    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1

    if n_components == 1:
        return face_adjacency

    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    component_dict = {}
    for face_idx in range(num_faces):
        root = uf.find(face_idx)
        if root not in component_dict:
            component_dict[root] = set()
        component_dict[root].add(face_idx)

    connected_comps = list(component_dict.values())

    component_centroid_face_idx = []
    connected_component_centroids = []
    for component in connected_comps:
        curr_component_faces = list(component)
        curr_component_face_centroids = face_centroids[curr_component_faces]
        component_centroid = np.mean(curr_component_face_centroids, axis=0)

        face_idx = curr_component_faces[np.argmin(
            np.linalg.norm(curr_component_face_centroids - component_centroid, axis=-1)
        )]

        connected_component_centroids.append(component_centroid)
        component_centroid_face_idx.append(face_idx)

    component_centroid_face_idx = np.array(component_centroid_face_idx)
    connected_component_centroids = np.array(connected_component_centroids)

    knn_k = min(n_components, k)
    knn = NearestNeighbors(n_neighbors=knn_k, algorithm='auto')
    knn.fit(connected_component_centroids)
    distances, indices = knn.kneighbors(connected_component_centroids)

    G = nx.Graph()
    G.add_nodes_from(range(num_faces))

    for idx1 in range(n_components):
        i = component_centroid_face_idx[idx1]
        for idx2, dist in zip(indices[idx1], distances[idx1]):
            j = component_centroid_face_idx[idx2]
            if i == j:
                continue
            G.add_edge(i, j, weight=dist)

    mst = nx.minimum_spanning_tree(G, weight='weight')
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )

    adjacency_lil = face_adjacency.tolil()

    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    face_adjacency = adjacency_lil.tocsr()

    if with_knn:
        dummy_row = []
        dummy_col = []
        for idx1 in range(n_components):
            i = component_centroid_face_idx[idx1]
            for idx2 in indices[idx1]:
                j = component_centroid_face_idx[idx2]
                dummy_row.extend([i, j])
                dummy_col.extend([j, i])

        dummy_data = np.ones(len(dummy_row), dtype=np.int16)
        dummy_mat = coo_matrix(
            (dummy_data, (dummy_row, dummy_col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        face_adjacency = face_adjacency + dummy_mat

    return face_adjacency
