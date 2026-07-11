# SPDX-License-Identifier: GPL-3.0-or-later
"""ComfyUI MeshSegmenter Nodes."""

# SAMesh nodes
from .sam_model_downloader import NODE_CLASS_MAPPINGS as _sam_model
from .sam_model_downloader import NODE_DISPLAY_NAME_MAPPINGS as _sam_model_d
from .loader import NODE_CLASS_MAPPINGS as _loader
from .loader import NODE_DISPLAY_NAME_MAPPINGS as _loader_d
from .multiview_renderer import NODE_CLASS_MAPPINGS as _mvr
from .multiview_renderer import NODE_DISPLAY_NAME_MAPPINGS as _mvr_d
from .generate_masks import NODE_CLASS_MAPPINGS as _genmasks
from .generate_masks import NODE_DISPLAY_NAME_MAPPINGS as _genmasks_d
from .combine_bmasks import NODE_CLASS_MAPPINGS as _cbm
from .combine_bmasks import NODE_DISPLAY_NAME_MAPPINGS as _cbm_d
from .combine_masks import NODE_CLASS_MAPPINGS as _cm
from .combine_masks import NODE_DISPLAY_NAME_MAPPINGS as _cm_d
from .lift_labels import NODE_CLASS_MAPPINGS as _lift
from .lift_labels import NODE_DISPLAY_NAME_MAPPINGS as _lift_d
from .smooth_labels import NODE_CLASS_MAPPINGS as _smooth
from .smooth_labels import NODE_DISPLAY_NAME_MAPPINGS as _smooth_d
from .graph_cut import NODE_CLASS_MAPPINGS as _gc
from .graph_cut import NODE_DISPLAY_NAME_MAPPINGS as _gc_d
from .apply_labels import NODE_CLASS_MAPPINGS as _al
from .apply_labels import NODE_DISPLAY_NAME_MAPPINGS as _al_d
from .renderer import NODE_CLASS_MAPPINGS as _rend
from .renderer import NODE_DISPLAY_NAME_MAPPINGS as _rend_d
from .exporter import NODE_CLASS_MAPPINGS as _exp
from .exporter import NODE_DISPLAY_NAME_MAPPINGS as _exp_d

# Quad mesh nodes
from .pyvista_loader import NODE_CLASS_MAPPINGS as _pvl
from .pyvista_loader import NODE_DISPLAY_NAME_MAPPINGS as _pvl_d
from .apply_labels_quad import NODE_CLASS_MAPPINGS as _alq
from .apply_labels_quad import NODE_DISPLAY_NAME_MAPPINGS as _alq_d
from .exporter_quad import NODE_CLASS_MAPPINGS as _expq
from .exporter_quad import NODE_DISPLAY_NAME_MAPPINGS as _expq_d

# PartField nodes
from .partfield_model_downloader import NODE_CLASS_MAPPINGS as _pf_model
from .partfield_model_downloader import NODE_DISPLAY_NAME_MAPPINGS as _pf_model_d
from .segmenter import NODE_CLASS_MAPPINGS as _seg
from .segmenter import NODE_DISPLAY_NAME_MAPPINGS as _seg_d
from .feature_extractor import NODE_CLASS_MAPPINGS as _fe
from .feature_extractor import NODE_DISPLAY_NAME_MAPPINGS as _fe_d
from .feature_visualizer import NODE_CLASS_MAPPINGS as _fv
from .feature_visualizer import NODE_DISPLAY_NAME_MAPPINGS as _fv_d
from .segment_by_features import NODE_CLASS_MAPPINGS as _sbf
from .segment_by_features import NODE_DISPLAY_NAME_MAPPINGS as _sbf_d

# Utility nodes
from .decimate import NODE_CLASS_MAPPINGS as _dec
from .decimate import NODE_DISPLAY_NAME_MAPPINGS as _dec_d

# Classical geometry segmentation / CAD-decomposition nodes (moved from GeometryPack)
from .detect_tangent_edges import NODE_CLASS_MAPPINGS as _dte
from .detect_tangent_edges import NODE_DISPLAY_NAME_MAPPINGS as _dte_d
from .region_growing import NODE_CLASS_MAPPINGS as _rg
from .region_growing import NODE_DISPLAY_NAME_MAPPINGS as _rg_d
from .segment_by_curvature import NODE_CLASS_MAPPINGS as _sbc
from .segment_by_curvature import NODE_DISPLAY_NAME_MAPPINGS as _sbc_d
from .find_fillets import NODE_CLASS_MAPPINGS as _ff
from .find_fillets import NODE_DISPLAY_NAME_MAPPINGS as _ff_d
from .fit_primitives import NODE_CLASS_MAPPINGS as _fp
from .fit_primitives import NODE_DISPLAY_NAME_MAPPINGS as _fp_d
from .segment_patches import NODE_CLASS_MAPPINGS as _sp
from .segment_patches import NODE_DISPLAY_NAME_MAPPINGS as _sp_d
from .vsa_segment import NODE_CLASS_MAPPINGS as _vs
from .vsa_segment import NODE_DISPLAY_NAME_MAPPINGS as _vs_d
from .compute_fillet_field import NODE_CLASS_MAPPINGS as _cff
from .compute_fillet_field import NODE_DISPLAY_NAME_MAPPINGS as _cff_d

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **_sam_model, **_loader, **_mvr, **_genmasks,
    **_cbm, **_cm, **_lift, **_smooth, **_gc,
    **_al, **_rend, **_exp,
    **_pvl, **_alq, **_expq,
    **_pf_model, **_seg, **_fe, **_fv, **_sbf,
    **_dec,
    **_dte, **_rg, **_sbc, **_ff, **_fp, **_sp, **_vs, **_cff,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_sam_model_d, **_loader_d, **_mvr_d, **_genmasks_d,
    **_cbm_d, **_cm_d, **_lift_d, **_smooth_d, **_gc_d,
    **_al_d, **_rend_d, **_exp_d,
    **_pvl_d, **_alq_d, **_expq_d,
    **_pf_model_d, **_seg_d, **_fe_d, **_fv_d, **_sbf_d,
    **_dec_d,
    **_dte_d, **_rg_d, **_sbc_d, **_ff_d, **_fp_d, **_sp_d, **_vs_d, **_cff_d,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
