# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Segment a mesh into CADable patches and OUTPUT them (mesh + per-face region_id).

Pipeline:
  1. feature edges = dihedral angle >= crease_angle_deg (optionally broadened over
     `wide_rings` so chamfers / fine fillets read as one turn, not many tiny ones).
  2. flood-fill faces into regions, cutting at those feature edges.
  3. CLEAN UP: iteratively merge each region smaller than min_patch_faces (or
     min_patch_area_pct of the surface) into the neighbour it shares the most
     boundary with. This is what turns a speckled segmentation into a usable set
     of patches.

Outputs the mesh with face_attributes['region_id'] (1..R) so you can colour it in
Preview Mesh (fields), split it with Split By Field, or fit primitives per patch.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

# _edge_values / _face_regions inlined from GeometryPack's
# visualization/preview_mesh_boundaries.py (this pack has no such module).

log = logging.getLogger("meshsegmenter")


def _edge_values(mesh, field_name, reduction, wide_dihedral=0):
    """Return (adj_edges[M,2] vertex idx, values[M], used_field_name, available_fields)."""
    import numpy as np

    adj = getattr(mesh, "face_adjacency", None)
    adj_edges = getattr(mesh, "face_adjacency_edges", None)
    avail = sorted((getattr(mesh, "face_attributes", {}) or {}).keys())
    if adj is None or adj_edges is None or len(adj) == 0:
        return np.zeros((0, 2), int), np.zeros(0), field_name, avail

    fattr = getattr(mesh, "face_attributes", {}) or {}
    if field_name in (None, "", "face_normals"):
        field = np.asarray(mesh.face_normals, dtype=np.float64)
        used = "face_normals"
    elif field_name in fattr:
        field = np.asarray(fattr[field_name], dtype=np.float64)
        used = field_name
    else:
        field = np.asarray(mesh.face_normals, dtype=np.float64)
        used = f"face_normals (']{field_name}' not found)"

    # WIDE DIHEDRAL: optionally average each face's field over `wide_dihedral` rings of
    # edge-adjacent faces (area-weighted) BEFORE the per-edge reduction. With reduction
    # 'angle' on face_normals this turns the raw 2-face dihedral into the angle between
    # broadened, multi-face normals -- so a feature smeared over many faces (a chamfer /
    # fine fillet) shows up as one larger turn instead of many tiny per-edge angles.
    # (Trade-off: it also rounds genuinely-sharp single-edge creases; 2-4 is a good range.)
    if wide_dihedral and int(wide_dihedral) > 0:
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        a0, a1 = adj[:, 0], adj[:, 1]
        f2 = field.reshape(len(field), -1).astype(np.float64)
        for _ in range(int(wide_dihedral)):
            acc = f2 * areas[:, None]
            wsum = areas.copy()
            np.add.at(acc, a0, f2[a1] * areas[a1, None]); np.add.at(wsum, a0, areas[a1])
            np.add.at(acc, a1, f2[a0] * areas[a0, None]); np.add.at(wsum, a1, areas[a0])
            f2 = acc / np.maximum(wsum[:, None], 1e-12)
        field = f2.reshape(field.shape)
        used += f" +{int(wide_dihedral)}ring"

    A = field[adj[:, 0]]
    B = field[adj[:, 1]]
    red = reduction
    if red == "auto":
        red = "angle" if (A.ndim == 2 and A.shape[1] > 1) else "abs_diff"

    if red == "angle":
        A2, B2 = np.atleast_2d(A), np.atleast_2d(B)
        dot = np.sum(A2 * B2, axis=1)
        na = np.linalg.norm(A2, axis=1)
        nb = np.linalg.norm(B2, axis=1)
        cos = np.clip(dot / (na * nb + 1e-12), -1.0, 1.0)
        vals = np.degrees(np.arccos(cos))
    elif red == "l2":
        vals = np.linalg.norm(np.atleast_2d(A) - np.atleast_2d(B), axis=1)
    elif red in ("max", "mean", "min"):
        # reduce the two adjacent faces' SCALAR value (first component if vector) ->
        # threshold edges by the LEVEL of a face field (e.g. curvature_angle_deg),
        # not its jump. max = either side curved, mean = average, min = both curved.
        a = A.reshape(len(A), -1)[:, 0]
        b = B.reshape(len(B), -1)[:, 0]
        if red == "max":
            vals = np.maximum(a, b)
        elif red == "min":
            vals = np.minimum(a, b)
        else:
            vals = 0.5 * (a + b)
    else:  # abs_diff (first component if vector)
        a = A.reshape(len(A), -1)[:, 0]
        b = B.reshape(len(B), -1)[:, 0]
        vals = np.abs(a - b)

    return np.asarray(adj_edges, dtype=np.int64), np.asarray(vals, dtype=np.float64), used, avail


def _face_regions(n_faces, adj_pairs, wall_mask):
    """Flood-fill faces into connected regions ("CADable patches"), cutting at the
    wall (feature) edges. Two faces sharing an adjacency edge are in the same
    region iff that edge is NOT a wall. Faces walled off on all sides become a
    region of one. Returns (region_id[n_faces] 1-based, n_regions).
    """
    import numpy as np

    if n_faces == 0:
        return np.zeros(0, np.int64), 0
    adj_pairs = np.asarray(adj_pairs).reshape(-1, 2)
    wall = np.asarray(wall_mask, dtype=bool)
    keep = adj_pairs[~wall] if len(adj_pairs) else np.zeros((0, 2), np.int64)

    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        data = np.ones(len(keep), dtype=np.int8)
        g = sp.coo_matrix((data, (keep[:, 0], keep[:, 1])), shape=(n_faces, n_faces))
        n_reg, labels = connected_components(g, directed=False, connection="weak")
        return (labels + 1).astype(np.int64), int(n_reg)
    except Exception:
        parent = list(range(n_faces))

        def find(x):
            r = x
            while parent[r] != r:
                r = parent[r]
            while parent[x] != r:
                parent[x], x = r, parent[x]
            return r

        for a, b in keep:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        roots = np.array([find(i) for i in range(n_faces)])
        uniq, inv = np.unique(roots, return_inverse=True)
        return (inv + 1).astype(np.int64), int(len(uniq))



def _merge_small_regions(region_id, adj_pairs, wall_mask, face_areas, min_faces, min_area):
    """Merge regions below the size thresholds into their strongest neighbour
    (most shared boundary edges, ties broken by neighbour size). Iterates to a
    fixed point. Returns cleaned, contiguously-relabelled region_id (1..R)."""
    from collections import defaultdict

    labels = np.asarray(region_id, dtype=np.int64).copy()
    adj_pairs = np.asarray(adj_pairs).reshape(-1, 2)
    wall = np.asarray(wall_mask, dtype=bool)
    walls = adj_pairs[wall] if len(adj_pairs) else np.zeros((0, 2), np.int64)
    areas = np.asarray(face_areas, dtype=np.float64)

    def relabel(lbl):
        uniq, inv = np.unique(lbl, return_inverse=True)
        return (inv + 1).astype(np.int64)

    for _ in range(10000):  # converges fast (region count strictly drops per pass)
        labels = relabel(labels)
        n = int(labels.max()) if len(labels) else 0
        fcount = np.bincount(labels, minlength=n + 1)
        acount = np.bincount(labels, weights=areas, minlength=n + 1)

        small = [r for r in range(1, n + 1)
                 if fcount[r] > 0 and (fcount[r] < min_faces or acount[r] < min_area)]
        if not small:
            break

        # region-adjacency strength from the wall edges (boundary between regions)
        rl = labels[walls[:, 0]] if len(walls) else np.zeros(0, np.int64)
        rr = labels[walls[:, 1]] if len(walls) else np.zeros(0, np.int64)
        m = rl != rr
        nbr = defaultdict(lambda: defaultdict(int))
        for a, b in zip(rl[m].tolist(), rr[m].tolist()):
            nbr[a][b] += 1
            nbr[b][a] += 1

        # union-find merge of every small region into its strongest neighbour
        parent = list(range(n + 1))

        def find(x):
            r = x
            while parent[r] != r:
                r = parent[r]
            while parent[x] != r:
                parent[x], x = r, parent[x]
            return r

        merged = False
        for r in sorted(small, key=lambda r: fcount[r]):
            if not nbr[r]:
                continue  # isolated region with no neighbour: leave it
            target = max(nbr[r].items(), key=lambda kv: (kv[1], fcount[kv[0]]))[0]
            ra, rt = find(r), find(target)
            if ra != rt:
                parent[ra] = rt
                merged = True
        if not merged:
            break
        labels = np.array([find(int(l)) for l in labels], dtype=np.int64)

    return relabel(labels)


class SegmentPatchesNode(io.ComfyNode):
    """Flood-fill a mesh into CADable patches, merge speckle, output region_id."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MeshSegSegmentPatches",
            display_name="Segment Mesh into Patches",
            category="meshsegmenter/geometry",
            description=(
                "Segment a mesh into CADable surface patches and output them as a per-face "
                "region_id field (1..R). Feature edges = dihedral >= crease_angle_deg cut the "
                "mesh; faces flood-fill into patches; then small/speckle patches are merged into "
                "their strongest neighbour so you get a clean set instead of fragments.\n"
                "\n"
                "wide_rings broadens the normals first (chamfers / fine fillets become one patch "
                "border instead of many tiny ones). min_patch_faces / min_patch_area_pct control "
                "how aggressively speckle is absorbed.\n"
                "\n"
                "Color the output in Preview Mesh (fields -> region_id), split it per-patch with "
                "Split By Field, or feed it to primitive fitting."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("crease_angle_deg", default=30.0, min=0.0, max=180.0, step=1.0,
                    tooltip="Dihedral angle (deg) above which an edge cuts patches apart."),
                io.Int.Input("wide_rings", default=0, min=0, max=6, step=1,
                    tooltip="Broaden each face's normal over this many rings before measuring the "
                            "dihedral, so chamfers / fine fillets read as a single border. 0 = off, "
                            "2-4 catches wider blends (also rounds genuinely-sharp single edges)."),
                io.Int.Input("min_patch_faces", default=20, min=0, max=1000000, step=1,
                    tooltip="Merge any patch with fewer than this many faces into a neighbour. "
                            "Raise to absorb more speckle."),
                io.Float.Input("min_patch_area_pct", default=0.05, min=0.0, max=50.0, step=0.01,
                    tooltip="Also merge any patch whose area is below this percent of the total "
                            "surface area. Catches thin/sliver patches that have many faces."),
                io.Combo.Input("preclean", options=["true", "false"], default="true",
                    tooltip="Merge duplicate vertices + drop degenerate faces first (recommended; "
                            "needed for correct face adjacency)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_patches"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, crease_angle_deg=30.0, wide_rings=0,
                min_patch_faces=20, min_patch_area_pct=0.05, preclean="true"):
        mesh = trimesh.copy()
        if preclean == "true":
            try:
                mesh.merge_vertices()
            except Exception as e:
                log.debug("merge_vertices skipped: %s", e)
            try:
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.remove_unreferenced_vertices()
            except Exception as e:
                log.debug("degenerate cleanup skipped: %s", e)

        n_faces = int(len(mesh.faces))
        adj_pairs = np.asarray(getattr(mesh, "face_adjacency", np.zeros((0, 2), int)))
        _, vals, used, _ = _edge_values(mesh, "face_normals", "angle", wide_dihedral=int(wide_rings))
        passing = vals >= float(crease_angle_deg)

        region_id, n_raw = _face_regions(n_faces, adj_pairs, passing)

        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        total_area = float(areas.sum()) or 1.0
        min_area = (float(min_patch_area_pct) / 100.0) * total_area
        region_id = _merge_small_regions(region_id, adj_pairs, passing, areas,
                                         int(min_patch_faces), min_area)
        n_final = int(region_id.max()) if len(region_id) else 0

        mesh.face_attributes["region_id"] = region_id.astype(np.float32)
        mesh.metadata = (mesh.metadata.copy() if mesh.metadata else {})
        mesh.metadata["patches"] = {
            "crease_angle_deg": float(crease_angle_deg), "wide_rings": int(wide_rings),
            "n_patches": n_final, "n_raw": int(n_raw),
        }

        # patch-size summary (top few)
        counts = np.bincount(region_id.astype(np.int64))
        sizes = sorted(counts[1:].tolist(), reverse=True)
        biggest = ", ".join(str(s) for s in sizes[:8])
        info = (
            f"Segment Mesh into Patches\n"
            f"faces={n_faces:,}  feature edges={int(passing.sum()):,} ({used})\n"
            f"patches: {n_raw:,} raw -> {n_final:,} after merging speckle\n"
            f"  (min {int(min_patch_faces)} faces / {min_patch_area_pct:g}% area)\n"
            f"largest patches (faces): {biggest}"
        )
        log.info("[SegmentPatches] %s", info.replace("\n", " | "))
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"MeshSegSegmentPatches": SegmentPatchesNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MeshSegSegmentPatches": "Segment Mesh into Patches"}
