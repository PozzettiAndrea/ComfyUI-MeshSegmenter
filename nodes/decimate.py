"""
Mesh decimation node with multiple backend options.
"""
import trimesh
import numpy as np


class DecimateMesh:
    """
    Decimates a mesh to reduce triangle count while preserving shape.
    Multiple algorithm backends available.
    """

    BACKENDS = ["pymeshlab_quadric", "pyvista_quadric", "pyvista_pro"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "backend": (cls.BACKENDS, {
                    "default": "pymeshlab_quadric",
                    "tooltip": "Algorithm: pymeshlab_quadric (Garland-Heckbert), pyvista_quadric (VTK), pyvista_pro (Schroeder 1992)"
                }),
                "target_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Target ratio of faces to keep. 0.5 = keep 50% of faces."
                }),
            },
            "optional": {
                "target_faces": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000000,
                    "step": 1000,
                    "tooltip": "Exact target face count. -1 to use target_ratio instead."
                }),
                "preserve_boundary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve mesh boundary edges."
                }),
                "preserve_topology": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve mesh topology (no holes). May limit reduction."
                }),
                "quality_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "[pymeshlab] Quality threshold. Higher = stricter."
                }),
                "planar_simplification": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[pymeshlab] Aggressively simplify planar regions."
                }),
                "planar_weight": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.0001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "[pymeshlab] Planar weight. Lower = more aggressive on flat areas."
                }),
                "feature_angle": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "tooltip": "[pyvista_pro] Feature angle for edge detection."
                }),
                "volume_preservation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[pyvista_quadric] Preserve volume to reduce normal errors."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "stats")
    FUNCTION = "decimate"
    CATEGORY = "meshsegmenter/mesh"

    def decimate(
        self,
        mesh: trimesh.Trimesh,
        backend: str,
        target_ratio: float,
        target_faces: int = -1,
        preserve_boundary: bool = True,
        preserve_topology: bool = False,
        quality_threshold: float = 0.3,
        planar_simplification: bool = True,
        planar_weight: float = 0.001,
        feature_angle: float = 45.0,
        volume_preservation: bool = False,
    ):
        input_verts = len(mesh.vertices)
        input_faces = len(mesh.faces)

        # Calculate target
        if target_faces > 0:
            actual_target = target_faces
        else:
            actual_target = int(input_faces * target_ratio)

        print(f"[Decimate] Backend: {backend}")
        print(f"[Decimate] Input: {input_verts} vertices, {input_faces} faces")
        print(f"[Decimate] Target: {actual_target} faces")

        if backend == "pymeshlab_quadric":
            output_mesh = self._decimate_pymeshlab(
                mesh, actual_target, preserve_boundary, preserve_topology,
                quality_threshold, planar_simplification, planar_weight
            )
        elif backend == "pyvista_quadric":
            output_mesh = self._decimate_pyvista_quadric(
                mesh, actual_target, input_faces, volume_preservation
            )
        elif backend == "pyvista_pro":
            output_mesh = self._decimate_pyvista_pro(
                mesh, actual_target, input_faces, feature_angle, preserve_topology, preserve_boundary
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        output_verts = len(output_mesh.vertices)
        output_faces = len(output_mesh.faces)
        reduction = (1 - output_faces / input_faces) * 100

        stats = f"{backend}: {input_faces} -> {output_faces} faces ({reduction:.1f}% reduction)"
        print(f"[Decimate] {stats}")

        return (output_mesh, stats)

    def _decimate_pymeshlab(
        self, mesh, target_faces, preserve_boundary, preserve_topology,
        quality_threshold, planar_simplification, planar_weight
    ):
        """PyMeshLab Quadric Edge Collapse (Garland & Heckbert 1997)"""
        import pymeshlab

        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices.astype(np.float64),
            face_matrix=mesh.faces.astype(np.int32),
        )
        ms.add_mesh(m)

        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=preserve_boundary,
            preservetopology=preserve_topology,
            qualitythr=quality_threshold,
            planarquadric=planar_simplification,
            planarweight=planar_weight,
        )

        result = ms.current_mesh()
        return trimesh.Trimesh(
            vertices=result.vertex_matrix(),
            faces=result.face_matrix()
        )

    def _decimate_pyvista_quadric(self, mesh, target_faces, input_faces, volume_preservation):
        """PyVista/VTK Quadric Decimation"""
        import pyvista as pv

        # Convert to pyvista
        faces_pv = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int32),
            mesh.faces.astype(np.int32)
        ]).flatten()
        pv_mesh = pv.PolyData(mesh.vertices.astype(np.float64), faces_pv)

        # Calculate reduction ratio
        reduction = 1.0 - (target_faces / input_faces)
        reduction = max(0.0, min(0.99, reduction))

        result = pv_mesh.decimate(
            target_reduction=reduction,
            volume_preservation=volume_preservation,
        )

        # Convert back to trimesh
        faces_out = result.faces.reshape(-1, 4)[:, 1:4]
        return trimesh.Trimesh(
            vertices=np.array(result.points),
            faces=faces_out
        )

    def _decimate_pyvista_pro(
        self, mesh, target_faces, input_faces, feature_angle, preserve_topology, preserve_boundary
    ):
        """PyVista/VTK DecimatePro (Schroeder et al. 1992)"""
        import pyvista as pv

        # Convert to pyvista
        faces_pv = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int32),
            mesh.faces.astype(np.int32)
        ]).flatten()
        pv_mesh = pv.PolyData(mesh.vertices.astype(np.float64), faces_pv)

        # Calculate reduction ratio
        reduction = 1.0 - (target_faces / input_faces)
        reduction = max(0.0, min(0.99, reduction))

        result = pv_mesh.decimate_pro(
            reduction=reduction,
            feature_angle=feature_angle,
            preserve_topology=preserve_topology,
            boundary_vertex_deletion=not preserve_boundary,
        )

        # Convert back to trimesh
        faces_out = result.faces.reshape(-1, 4)[:, 1:4]
        return trimesh.Trimesh(
            vertices=np.array(result.points),
            faces=faces_out
        )


NODE_CLASS_MAPPINGS = {
    "DecimateMesh": DecimateMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DecimateMesh": "Decimate Mesh",
}
