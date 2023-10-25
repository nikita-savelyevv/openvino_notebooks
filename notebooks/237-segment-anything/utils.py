import torch
import numpy as np
from copy import deepcopy
from torchvision.transforms.functional import resize, to_pil_image
import matplotlib.pyplot as plt
import cv2

from segment_anything.utils.amg import (
    MaskData,
    generate_crop_boxes,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    calculate_stability_score,
    rle_to_mask,
    batched_mask_to_box,
    mask_to_rle_pytorch,
    is_box_near_crop_edge,
    batch_iterator,
    remove_small_regions,
    build_all_layer_point_grids,
    box_xyxy_to_xywh,
    area_from_rle
)

from torchvision.ops.boxes import batched_nms, box_area
from typing import Tuple, List, Dict, Any


class SamExportableModel(torch.nn.Module):
    def __init__(
            self,
            model,
            return_single_mask: bool,
            use_stability_score: bool = False,
            return_extra_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1).to(torch.float32)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
                point_labels == -1
        ).to(torch.float32)

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i).to(torch.float32)

        return point_embedding

    def t_embed_masks(self, input_mask: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.model.prompt_encoder.mask_downscaling(input_mask)
        return mask_embedding

    def mask_postprocessing(self, masks: torch.Tensor) -> torch.Tensor:
        masks = torch.nn.functional.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks

    def select_masks(
            self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
            self,
            image_embeddings: torch.Tensor,
            point_coords: torch.Tensor,
            point_labels: torch.Tensor,
            mask_input: torch.Tensor = None,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        if mask_input is None:
            dense_embedding = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                point_coords.shape[0], -1, image_embeddings.shape[0], 64
            )
        else:
            dense_embedding = self._embed_masks(mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        upscaled_masks = self.mask_postprocessing(masks)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(
                upscaled_masks, self.model.mask_threshold, self.stability_score_offset
            )
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores


class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming numpy arrays.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


resizer = ResizeLongestSide(1024)


def preprocess_image(image: np.ndarray):
    resized_image = resizer.apply_image(image)
    resized_image = (resized_image.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
    resized_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)).astype(np.float32), 0)

    # Pad
    h, w = resized_image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = np.pad(resized_image, ((0, 0), (0, 0), (0, padh), (0, padw)))
    return x


def postprocess_masks(masks: np.ndarray, orig_size):
    size_before_pad = resizer.get_preprocess_shape(orig_size[0], orig_size[1], masks.shape[-1])
    masks = masks[..., :int(size_before_pad[0]), :int(size_before_pad[1])]
    masks = torch.nn.functional.interpolate(torch.from_numpy(masks), size=orig_size, mode="bilinear",
                                            align_corners=False).numpy()
    return masks


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def process_batch(
    ov_decoder,
    image_embedding: np.ndarray,
    points: np.ndarray,
    im_size: Tuple[int, ...],
    crop_box: List[int],
    orig_size: Tuple[int, ...],
    iou_thresh,
    mask_threshold,
    stability_score_offset,
    stability_score_thresh
) -> MaskData:
    orig_h, orig_w = orig_size

    # Run model on this batch
    transformed_points = resizer.apply_coords(points, im_size)
    in_points = transformed_points
    in_labels = np.ones(in_points.shape[0], dtype=int)

    inputs = {
        "image_embeddings": image_embedding,
        "point_coords": in_points[:, None, :],
        "point_labels": in_labels[:, None],
    }
    res = ov_decoder(inputs)
    masks = postprocess_masks(res[ov_decoder.output(0)], orig_size)
    masks = torch.from_numpy(masks)
    iou_preds = torch.from_numpy(res[ov_decoder.output(1)])

    # Serialize predictions and store in MaskData
    data = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
    )
    del masks

    # Filter by predicted IoU
    if iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > iou_thresh
        data.filter(keep_mask)

    # Calculate stability score
    data["stability_score"] = calculate_stability_score(
        data["masks"], mask_threshold, stability_score_offset
    )
    if stability_score_thresh > 0.0:
        keep_mask = data["stability_score"] >= stability_score_thresh
        data.filter(keep_mask)

    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])

    # Filter boxes that touch crop boundaries
    keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
    if not torch.all(keep_mask):
        data.filter(keep_mask)

    # Compress to RLE
    data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    del data["masks"]
    return data


def process_crop(
    ov_encoder,
    ov_decoder,
    image: np.ndarray,
    point_grids,
    crop_box: List[int],
    crop_layer_idx: int,
    orig_size: Tuple[int, ...],
    box_nms_thresh:float = 0.7,
    mask_threshold:float = 0.0,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    stability_score_offset: float = 1.0,
) -> MaskData:
    # Crop the image and calculate embeddings
    x0, y0, x1, y1 = crop_box
    cropped_im = image[y0:y1, x0:x1, :]
    cropped_im_size = cropped_im.shape[:2]
    preprocessed_cropped_im = preprocess_image(cropped_im)
    crop_embeddings = ov_encoder(preprocessed_cropped_im)[ov_encoder.output(0)]

    # Get points for this crop
    points_scale = np.array(cropped_im_size)[None, ::-1]
    points_for_image = point_grids[crop_layer_idx] * points_scale

    # Generate masks for this crop in batches
    data = MaskData()
    for (points,) in batch_iterator(points_per_batch, points_for_image):
        batch_data = process_batch(ov_decoder, crop_embeddings, points, cropped_im_size, crop_box, orig_size,
                                   pred_iou_thresh, mask_threshold, stability_score_offset, stability_score_thresh)
        data.cat(batch_data)
        del batch_data

    # Remove duplicates within this crop.
    keep_by_nms = batched_nms(
        data["boxes"].float(),
        data["iou_preds"],
        torch.zeros(len(data["boxes"])),  # categories
        iou_threshold=box_nms_thresh,
    )
    data.filter(keep_by_nms)

    # Return to the original image frame
    data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
    data["points"] = uncrop_points(data["points"], crop_box)
    data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

    return data


def generate_masks(ov_encoder, ov_decoder, image: np.ndarray, point_grids, crop_n_layers, crop_overlap_ratio,
                   crop_nms_thresh) -> MaskData:
    orig_size = image.shape[:2]
    crop_boxes, layer_idxs = generate_crop_boxes(
        orig_size, crop_n_layers, crop_overlap_ratio
    )

    # Iterate over image crops
    data = MaskData()
    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        crop_data = process_crop(ov_encoder, ov_decoder, image, point_grids, crop_box, layer_idx, orig_size)
        data.cat(crop_data)

    # Remove duplicate masks between crops
    if len(crop_boxes) > 1:
        # Prefer masks from smaller crops
        scores = 1 / box_area(data["crop_boxes"])
        scores = scores.to(data["boxes"].device)
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            scores,
            torch.zeros(len(data["boxes"])),  # categories
            iou_threshold=crop_nms_thresh,
        )
        data.filter(keep_by_nms)

    data.to_numpy()
    return data


def postprocess_small_regions(mask_data: MaskData, min_area: int, nms_thresh: float) -> MaskData:
    """
    Removes small disconnected regions and holes in masks, then reruns
    box NMS to remove any new duplicates.

    Edits mask_data in place.

    Requires open-cv as a dependency.
    """
    if len(mask_data["rles"]) == 0:
        return mask_data

    # Filter small disconnected regions and holes
    new_masks = []
    scores = []
    for rle in mask_data["rles"]:
        mask = rle_to_mask(rle)

        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        # Give score=0 to changed masks and score=1 to unchanged masks
        # so NMS will prefer ones that didn't need postprocessing
        scores.append(float(unchanged))

    # Recalculate boxes and remove any new duplicates
    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros(len(boxes)),  # categories
        iou_threshold=nms_thresh,
    )

    # Only recalculate RLEs for masks that have changed
    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_torch = masks[i_mask].unsqueeze(0)
            mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
            # update res directly
            mask_data["boxes"][i_mask] = boxes[i_mask]
    mask_data.filter(keep_by_nms)

    return mask_data


def automatic_mask_generation(
    ov_encoder, ov_decoder,
    image: np.ndarray, min_mask_region_area: int = 0, points_per_side: int = 32, crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 1, crop_overlap_ratio: float = 512 / 1500, box_nms_thresh: float = 0.7,
        crop_nms_thresh: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Generates masks for the given image.

    Arguments:
      image (np.ndarray): The image to generate masks for, in HWC uint8 format.

    Returns:
       list(dict(str, any)): A list over records for masks. Each record is
         a dict containing the following keys:
           segmentation (dict(str, any) or np.ndarray): The mask. If
             output_mode='binary_mask', is an array of shape HW. Otherwise,
             is a dictionary containing the RLE.
           bbox (list(float)): The box around the mask, in XYWH format.
           area (int): The area in pixels of the mask.
           predicted_iou (float): The model's own prediction of the mask's
             quality. This is filtered by the pred_iou_thresh parameter.
           point_coords (list(list(float))): The point coordinates input
             to the model to generate this mask.
           stability_score (float): A measure of the mask's quality. This
             is filtered on using the stability_score_thresh parameter.
           crop_box (list(float)): The crop of the image used to generate
             the mask, given in XYWH format.
    """
    point_grids = build_all_layer_point_grids(
        points_per_side,
        crop_n_layers,
        crop_n_points_downscale_factor,
    )
    mask_data = generate_masks(
        ov_encoder, ov_decoder, image, point_grids, crop_n_layers, crop_overlap_ratio, crop_nms_thresh)

    # Filter small disconnected regions and holes in masks
    if min_mask_region_area > 0:
        mask_data = postprocess_small_regions(
            mask_data,
            min_mask_region_area,
            max(box_nms_thresh, crop_nms_thresh),
        )

    mask_data["segmentations"] = [
        rle_to_mask(rle) for rle in mask_data["rles"]]

    # Write mask records
    curr_anns = []
    for idx in range(len(mask_data["segmentations"])):
        ann = {
            "segmentation": mask_data["segmentations"][idx],
            "area": area_from_rle(mask_data["rles"][idx]),
            "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
            "predicted_iou": mask_data["iou_preds"][idx].item(),
            "point_coords": [mask_data["points"][idx].tolist()],
            "stability_score": mask_data["stability_score"][idx].item(),
            "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
        }
        curr_anns.append(ann)

    return curr_anns


def draw_anns(image, anns):
    if len(anns) == 0:
        return
    segments_image = image.copy()
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for ann in sorted_anns:
        mask = ann["segmentation"]
        mask_color = np.random.randint(0, 255, size=(1, 1, 3)).astype(np.uint8)
        segments_image[mask] = mask_color
    return cv2.addWeighted(image.astype(np.float32), 0.7, segments_image.astype(np.float32), 0.3, 0.0)
