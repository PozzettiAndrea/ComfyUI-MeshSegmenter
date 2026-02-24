import re
import sys
import os
import logging
from pathlib import Path
from contextlib import contextmanager, nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoModel

# Silence SAM2 verbose output before importing
logging.getLogger('sam2').setLevel(logging.ERROR)
logging.getLogger('sam2.sam2_image_predictor').setLevel(logging.ERROR)
logging.getLogger('sam2.automatic_mask_generator').setLevel(logging.ERROR)

USE_SAMHQ = False
if USE_SAMHQ:
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
else:
    from ...sam2.build_sam import build_sam2
    from ...sam2.sam2_image_predictor import SAM2ImagePredictor
    from ...sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Monkey-patch SAM2 to suppress verbose prints
    import io
    from contextlib import redirect_stdout, redirect_stderr

    _original_set_image = SAM2ImagePredictor.set_image
    def _quiet_set_image(self, image, *args, **kwargs):
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            return _original_set_image(self, image, *args, **kwargs)
    SAM2ImagePredictor.set_image = _quiet_set_image

    _original_generate = SAM2AutomaticMaskGenerator.generate
    def _quiet_generate(self, image, *args, **kwargs):
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            return _original_generate(self, image, *args, **kwargs)
    SAM2AutomaticMaskGenerator.generate = _quiet_generate


from ..data.common import NumpyTensor


@contextmanager
def suppress_output():
    """Suppress stdout/stderr prints at file descriptor level (catches C-level prints too)."""
    import logging
    # Suppress SAM2 loggers
    for name in ['sam2', 'sam2.sam2_image_predictor', 'sam2.automatic_mask_generator']:
        logging.getLogger(name).setLevel(logging.ERROR)

    # File descriptor level redirect (catches C prints and libraries that cache stdout)
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save copies of the file descriptors
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        # Redirect to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def combine_bmasks(masks: NumpyTensor['n h w'], sort=False) -> NumpyTensor['h w']:
    """
    """
    mask_combined = np.zeros_like(masks[0], dtype=int)
    if sort:
        masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    for i, mask in enumerate(masks):
        mask_combined[mask] = i + 1
    return mask_combined


def decompose_mask(mask: NumpyTensor['h w'], background=0) -> NumpyTensor['n h w']:
    """
    """
    labels = np.unique(mask)
    labels = labels[labels != background]
    return mask == labels[:, None, None]


def remove_artifacts(mask: NumpyTensor['h w'], mode: str, min_area=128) -> NumpyTensor['h w']:
    """
    Removes small islands/fill holes from a mask.
    """
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')

    def remove_helper(bmask):
        # opencv connected components operates on binary masks only
        bmask = (mode_holes ^ bmask).astype(np.uint8)
        nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
        sizes = stats[:, -1][1:]  # Row 0 corresponds to 0 pixels
        fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
        if not mode_holes:
            fill = [i for i in range(nregions) if i not in fill]
        return np.isin(regions, fill)

    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask): # also process background
        mask_combined[remove_helper(mask == label)] = label
    return mask_combined


def colormap_mask(
    mask : NumpyTensor['h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), foreground=None, blend=0.25
) -> Image.Image:
    """
    """
    palette = np.random.randint(0, 255, (np.max(mask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[mask.astype(int)] # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)


def colormap_bmask(bmask: NumpyTensor['h w']) -> Image.Image:
    """
    """
    return colormap_mask(bmask, background=np.array([0, 0, 0]), foreground=np.array([255, 255, 255]))


def colormap_bmasks(
    masks: NumpyTensor['n h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), blend=0.25
) -> Image.Image:
    """
    """
    mask = combine_bmasks(masks)
    return colormap_mask(mask, image, background=background, blend=blend)


def point_grid_from_mask(mask: NumpyTensor['h w'], n: int) -> NumpyTensor['n 2']:
    """
    Sample points within valid mask normalized to [0, 1] x [0, 1]
    """
    valid = np.argwhere(mask)
    if len(valid) == 0:
        raise ValueError('No valid points in mask')

    h, w = mask.shape
    n = min(n, len(valid))
    indices = np.random.choice(len(valid), n, replace=False)
    samples = valid[indices].astype(float)
    samples[:, 0] /= h - 1
    samples[:, 1] /= w - 1
    samples = samples[:, [1, 0]]
    samples = samples[np.lexsort((samples[:, 1], samples[:, 0]))]
    return samples


class SamModel(nn.Module):
    """
    """
    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        super().__init__()
        self.config = config
        self.device = device
        
        if config.sam.auto:
            self.setup_sam(mode='auto')
        else:
            if config.sam.ground:
                self.setup_grounding_dino()
            self.setup_sam(mode='pred')

    def setup_sam(self, mode='auto'):
        """
        """
        match = re.search(r'vit_(l|tiny|h)', self.config.sam.checkpoint)
        self.sam_model = sam_model_registry[match.group(0)](checkpoint=self.config.sam.checkpoint)
        self.sam_model = self.sam_model.to(self.device)
        self.sam_model.eval()
        self.engine = {
            'pred': SamPredictor,
            'auto': SamAutomaticMaskGenerator,
        }[mode](self.sam_model, **self.config.sam.get('engine_config', {}))

    def setup_grounding_dino(self):
        """
        """
        self.grounding_dino_processor, self.grounding_dino_model = \
            AutoProcessor.from_pretrained(self.config.grounding_dino.checkpoint), \
            AutoModel    .from_pretrained(self.config.grounding_dino.checkpoint).to(self.device)

    def process_image(self, image: Image, prompt: dict = None) -> NumpyTensor['n h w']:
        """
        For information on prompt format see:

        https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py#L104
        """
        image = np.array(image)

        # Suppress SAM2 verbose output
        with suppress_output():
            if self.config.sam.auto:
                annotations = self.engine.generate(image)
            else:
                self.engine.set_image(image)
                annotations = self.engine.predict(**prompt)[0]
                annotations = [{'segmentation': m, 'area': m.sum().item()} for m in annotations]

        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        masks = np.stack([anno['segmentation'] for anno in annotations])
        return masks

    def process_boxes(self, image: Image, texts: list[str]) -> tuple[
        list[NumpyTensor[4]],
        list[NumpyTensor[2]]
    ]:
        """
        """
        texts = '. '.join(texts)
        inputs = self.grounding_dino_processor(texts, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.grounding_dino_model(**inputs)

        boxes, logits = self.grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
        )
        return boxes, logits

    def forward(self, image: Image, texts: list[str]=None) -> NumpyTensor['n h w']:
        """
        """
        if self.config.sam.auto:
            masks = self.process_image(image)
        else:
            boxes, _ = self.process_boxes(image, texts)
            masks = []
            for box in boxes:
                masks.append(self.process_image(image, {'box': box}))
            masks = np.concatenate(masks)
        return masks


class Sam2Model(SamModel):
    """
    """
    def __init__(self, config: OmegaConf, device='cuda', dtype=None):
        self.dtype = dtype
        super().__init__(config, device)

    def setup_sam(self, mode='auto'):
        """
        """
        self.sam_model = build_sam2(self.config.sam.model_config, self.config.sam.checkpoint, device=self.device, apply_postprocessing=False)
        self.sam_model.eval()
        if self.dtype is not None:
            self.sam_model.to(dtype=self.dtype)
        self.engine = {
            'pred': SAM2ImagePredictor,
            'auto': SAM2AutomaticMaskGenerator,
        }[mode](self.sam_model, **self.config.sam.get('engine_config', {}))

    def _autocast_ctx(self):
        """Return nullcontext (model weights are already cast to target dtype)."""
        return nullcontext()

    def process_image(self, image: Image, prompt: dict = None) -> NumpyTensor['n h w']:
        """
        Override to suppress ALL SAM2 output including internal prints.
        """
        image = np.array(image)

        with suppress_output(), self._autocast_ctx():
            if self.config.sam.auto:
                annotations = self.engine.generate(image)
            else:
                self.engine.set_image(image)
                annotations = self.engine.predict(**prompt)[0]
                annotations = [{'segmentation': m, 'area': m.sum().item()} for m in annotations]

        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        masks = np.stack([anno['segmentation'] for anno in annotations])
        return masks

    def process_images_batch(
        self,
        images: list,
        point_grids: list,
        encode_batch_size: int = 6,
    ) -> list:
        """
        Batch-encode images then run per-image AMG decode.

        Uses set_image_batch() for efficient backbone encoding, then
        _predict(img_idx=i) to run the mask decoder per view.

        Args:
            images: list of np.ndarray (H,W,3) uint8
            point_grids: list of np.ndarray (N,2) normalized [0,1] per image
            encode_batch_size: max images per encoding batch (VRAM limit)

        Returns:
            list of np.ndarray (N_masks, H, W) bool per image
        """
        import time
        from torchvision.ops.boxes import batched_nms
        from ...sam2.utils.amg import (
            MaskData,
            batch_iterator,
            calculate_stability_score,
            mask_to_rle_pytorch,
            rle_to_mask,
            batched_mask_to_box,
            is_box_near_crop_edge,
            uncrop_masks,
        )

        engine = self.engine
        predictor = engine.predictor

        all_masks_out = []
        t_total = time.time()

        # Process in sub-batches for VRAM
        for batch_start in range(0, len(images), encode_batch_size):
            batch_end = min(batch_start + encode_batch_size, len(images))
            sub_images = images[batch_start:batch_end]
            sub_grids = point_grids[batch_start:batch_end]
            batch_label = f"[{batch_start+1}-{batch_end}/{len(images)}]"

            # Batch encode
            t_enc = time.time()
            with self._autocast_ctx():
                predictor.set_image_batch(sub_images)
            t_enc = time.time() - t_enc
            print(f"      {batch_label} Encoded {len(sub_images)} images in {t_enc:.2f}s")

            # Per-image AMG decode
            for local_idx, (image, pg) in enumerate(zip(sub_images, sub_grids)):
                view_idx = batch_start + local_idx
                h, w = image.shape[:2]
                im_size = (h, w)
                crop_box = [0, 0, w, h]

                if len(pg) == 0:
                    all_masks_out.append(np.zeros((1, h, w), dtype=bool))
                    print(f"      View {view_idx+1}/{len(images)}: skipped (no valid points)")
                    continue

                t_view = time.time()

                # Scale normalized points to pixel coords
                points_for_image = pg.copy()
                points_for_image[:, 0] *= w
                points_for_image[:, 1] *= h

                # Run point batches against cached features
                data = MaskData()
                with self._autocast_ctx():
                    for (points_batch,) in batch_iterator(engine.points_per_batch, points_for_image):
                        batch_data = self._process_point_batch(
                            predictor, engine, points_batch,
                            im_size, crop_box, local_idx,
                        )
                        data.cat(batch_data)
                        del batch_data

                # NMS
                n_masks = 0
                if "boxes" in data._stats and len(data["boxes"]) > 0:
                    keep = batched_nms(
                        data["boxes"].float(),
                        data["iou_preds"],
                        torch.zeros_like(data["boxes"][:, 0]),
                        iou_threshold=engine.box_nms_thresh,
                    )
                    data.filter(keep)

                data.to_numpy()

                # Extract binary masks from RLE
                if "rles" in data._stats and len(data["rles"]) > 0:
                    masks = [rle_to_mask(rle) for rle in data["rles"]]
                    masks = sorted(masks, key=lambda m: m.sum(), reverse=True)
                    all_masks_out.append(np.stack(masks))
                    n_masks = len(masks)
                else:
                    all_masks_out.append(np.zeros((1, h, w), dtype=bool))

                t_view = time.time() - t_view
                print(f"      View {view_idx+1}/{len(images)}: {n_masks} masks in {t_view:.2f}s")

            predictor.reset_predictor()

        t_total = time.time() - t_total
        print(f"      Total batch: {len(images)} views in {t_total:.2f}s ({t_total/len(images):.2f}s/view)")

        return all_masks_out

    @staticmethod
    def _process_point_batch(predictor, engine, points, im_size, crop_box, img_idx):
        """
        Replicate AMG._process_batch() logic using img_idx for batched features.
        """
        from ...sam2.utils.amg import (
            MaskData,
            calculate_stability_score,
            mask_to_rle_pytorch,
            batched_mask_to_box,
            is_box_near_crop_edge,
            uncrop_masks,
        )

        orig_h, orig_w = im_size

        points = torch.as_tensor(points, dtype=torch.float32, device=predictor.device)
        in_points = predictor._transforms.transform_coords(
            points, normalize=True, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )

        masks, iou_preds, low_res_masks = predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=engine.multimask_output,
            return_logits=True,
            img_idx=img_idx,
        )

        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat_interleave(masks.shape[1], dim=0),
            low_res_masks=low_res_masks.flatten(0, 1),
        )
        del masks

        # Filter by IoU
        if engine.pred_iou_thresh > 0.0:
            keep = data["iou_preds"] > engine.pred_iou_thresh
            data.filter(keep)

        # Filter by stability
        data["stability_score"] = calculate_stability_score(
            data["masks"], engine.mask_threshold, engine.stability_score_offset
        )
        if engine.stability_score_thresh > 0.0:
            keep = data["stability_score"] >= engine.stability_score_thresh
            data.filter(keep)

        # Threshold and boxes
        data["masks"] = data["masks"] > engine.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter edge boxes
        keep = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep):
            data.filter(keep)

        # RLE encode
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data


if __name__ == '__main__':
    import time

    device = 'cuda'
    image = Image.open('/home/ubuntu/meshseg/tests/examples/goldpot.png')

    config = OmegaConf.create({
        'sam': {
            'checkpoint': '/home/ubuntu/meshseg/checkpoints/sam_hq_vit_h.pth',
            'auto': True, 
            'ground': False,
            'engine_config': {'points_per_side': 32},
        },
        'grounding_dino': {
            'checkpoint': 'IDEA-Research/grounding-dino-tiny', # TODO find larger model
        },
    })

    sam = SamModel(config, device)
    start_time = time.time()
    masks = sam(image)
    print(f'Elapsed time: {time.time() - start_time:.2f} s')
    image = colormap_bmasks(masks, np.array(image))
    image.save('test_mask.png')

    '''
    config.sam.auto = False
    config.sam.ground = False
    sam = SamModel(config, device)
    masks = sam.process_image(image, prompt={
        'point_coords': np.array([[image.height // 2, image.width // 2]]),
        'point_labels': np.ones((1,)),
        'multimask_output': False
    })
    image = colormap_bmasks(masks, np.array(image))
    image.save('test_mask_grounded.png')
    '''

    # For RuntimeError: No available kernel. Aborting execution
    # https://github.com/facebookresearch/segment-anything-2/issues/48
    config2 = OmegaConf.create({
        'sam': {
            'model_config': 'sam2_hiera_l.yaml',
            'checkpoint'  : '/home/ubuntu/meshseg/checkpoints/sam2_hiera_large.pt',
            'auto': True, 
            'engine_config': {'points_per_side': 32},
        },
    })

    sam2 = Sam2Model(config2, device)
    start_time = time.time()
    masks = sam2(image)
    print(f'Elapsed time: {time.time() - start_time:.2f} s')
    image = colormap_bmasks(masks, np.array(image))
    image.save('test_mask2.png')