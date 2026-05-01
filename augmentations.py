from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


TargetType = Dict[str, Any]
SizeType = Union[int, Tuple[int, int], List[int]]


def _clone_target(target: Optional[TargetType]) -> TargetType:
    target = {} if target is None else target
    cloned: TargetType = {}
    for key, value in target.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def _get_image_size(image: Union[Image.Image, torch.Tensor]) -> Tuple[int, int]:
    if isinstance(image, torch.Tensor):
        return int(image.shape[-1]), int(image.shape[-2])
    return image.size


def _ensure_target(target: Optional[TargetType]) -> TargetType:
    target = _clone_target(target)
    boxes = target.get("boxes")
    labels = target.get("labels")

    if boxes is None:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
    else:
        boxes = boxes.clone().reshape(-1, 4).to(torch.float32)

    if labels is None:
        labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    else:
        labels = labels.clone().reshape(-1).to(torch.int64)

    target["boxes"] = boxes
    target["labels"] = labels
    return target


def _clamp_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(width))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(height))
    return boxes


def _filter_target(
    target: TargetType,
    width: int,
    height: int,
    min_size: float = 2.0,
    min_area: float = 4.0,
    visibility: Optional[torch.Tensor] = None,
    min_visibility: float = 0.25,
) -> TargetType:
    target = _ensure_target(target)
    boxes = _clamp_boxes(target["boxes"], width, height)

    if boxes.numel() == 0:
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0,), dtype=torch.int64)
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        target.setdefault("iscrowd", torch.zeros((0,), dtype=torch.int64))
        return target

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    keep = (widths >= min_size) & (heights >= min_size) & (areas >= min_area)

    if visibility is not None and visibility.numel() == keep.numel():
        keep = keep & (visibility >= min_visibility)

    old_count = boxes.shape[0]
    target["boxes"] = boxes[keep]

    for key, value in list(target.items()):
        if key == "boxes" or not torch.is_tensor(value):
            continue
        if value.ndim > 0 and value.shape[0] == old_count:
            target[key] = value[keep]

    new_boxes = target["boxes"]
    if new_boxes.numel() == 0:
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0,), dtype=torch.int64)
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        target.setdefault("iscrowd", torch.zeros((0,), dtype=torch.int64))
        return target

    new_widths = new_boxes[:, 2] - new_boxes[:, 0]
    new_heights = new_boxes[:, 3] - new_boxes[:, 1]
    target["area"] = new_widths * new_heights
    target.setdefault("iscrowd", torch.zeros((new_boxes.shape[0],), dtype=torch.int64))
    return target


def _boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4, 2), dtype=torch.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    return torch.stack(
        [
            torch.stack([x1, y1], dim=1),
            torch.stack([x2, y1], dim=1),
            torch.stack([x2, y2], dim=1),
            torch.stack([x1, y2], dim=1),
        ],
        dim=1,
    )


def _corners_to_boxes(corners: torch.Tensor) -> torch.Tensor:
    if corners.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    min_xy = corners.min(dim=1).values
    max_xy = corners.max(dim=1).values
    return torch.cat([min_xy, max_xy], dim=1)


def _apply_linear_to_boxes(
    boxes: torch.Tensor,
    matrix: torch.Tensor,
    center: Tuple[float, float],
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    cx, cy = center
    corners = _boxes_to_corners(boxes)
    corners[..., 0] -= cx
    corners[..., 1] -= cy
    transformed = corners @ matrix.T
    transformed[..., 0] += cx
    transformed[..., 1] += cy
    return _corners_to_boxes(transformed)


class Compose:
    def __init__(self, transforms: Sequence[Any]):
        self.transforms = list(transforms)

    def __call__(self, image: Union[Image.Image, torch.Tensor], target: Optional[TargetType] = None):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class DualCompose(Compose):
    pass


class RandomApply:
    def __init__(self, transforms: Sequence[Any], prob: float = 0.5):
        self.transforms = Compose(transforms)
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            return self.transforms(image, target)
        return image, target


class OneOf:
    def __init__(self, transforms: Sequence[Any], prob: float = 0.5, weights: Optional[Sequence[float]] = None):
        self.transforms = list(transforms)
        self.prob = prob
        self.weights = list(weights) if weights is not None else None

    def __call__(self, image, target=None):
        if not self.transforms or random.random() >= self.prob:
            return image, target
        transform = random.choices(self.transforms, weights=self.weights, k=1)[0]
        return transform(image, target)


class NoTransform:
    def __call__(self, image, target=None):
        return image, target


class Resize:
    def __init__(self, output_size: SizeType, interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        self.output_size = output_size
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        target = _ensure_target(target)
        old_w, old_h = _get_image_size(image)

        if isinstance(self.output_size, int):
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = int(self.output_size[0]), int(self.output_size[1])

        image = F.resize(image, [new_h, new_w], interpolation=self.interpolation)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            scale_x = new_w / float(old_w)
            scale_y = new_h / float(old_h)
            boxes = boxes.clone()
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            target["boxes"] = boxes
        return image, _filter_target(target, new_w, new_h)


class HorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target

        target = _ensure_target(target)
        width, height = _get_image_size(image)
        image = F.hflip(image)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            x1 = width - boxes[:, 2]
            x2 = width - boxes[:, 0]
            boxes = boxes.clone()
            boxes[:, 0] = x1
            boxes[:, 2] = x2
            target["boxes"] = boxes
        return image, _filter_target(target, width, height)


class VerticalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target

        target = _ensure_target(target)
        width, height = _get_image_size(image)
        image = F.vflip(image)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            y1 = height - boxes[:, 3]
            y2 = height - boxes[:, 1]
            boxes = boxes.clone()
            boxes[:, 1] = y1
            boxes[:, 3] = y2
            target["boxes"] = boxes
        return image, _filter_target(target, width, height)


class Rotate:
    def __init__(
        self,
        degree_range: Tuple[float, float] = (-10.0, 10.0),
        prob: float = 0.5,
        fill: Union[int, Tuple[int, int, int]] = 0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        self.degree_range = degree_range
        self.prob = prob
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target

        target = _ensure_target(target)
        width, height = _get_image_size(image)
        angle = random.uniform(self.degree_range[0], self.degree_range[1])
        image = F.rotate(image, angle=angle, interpolation=self.interpolation, expand=False, fill=self.fill)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            radians = math.radians(angle)
            matrix = torch.tensor(
                [
                    [math.cos(radians), -math.sin(radians)],
                    [math.sin(radians), math.cos(radians)],
                ],
                dtype=torch.float32,
            )
            center = (width / 2.0, height / 2.0)
            target["boxes"] = _apply_linear_to_boxes(boxes, matrix, center)
        return image, _filter_target(target, width, height)


class Scale:
    def __init__(
        self,
        ratio_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.5,
        fill: Union[int, Tuple[int, int, int]] = 0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        self.ratio_range = ratio_range
        self.prob = prob
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target

        target = _ensure_target(target)
        width, height = _get_image_size(image)
        ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
        image = F.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=ratio,
            shear=[0.0, 0.0],
            interpolation=self.interpolation,
            fill=self.fill,
        )

        boxes = target["boxes"]
        if boxes.numel() > 0:
            matrix = torch.tensor([[ratio, 0.0], [0.0, ratio]], dtype=torch.float32)
            center = (width / 2.0, height / 2.0)
            target["boxes"] = _apply_linear_to_boxes(boxes, matrix, center)
        return image, _filter_target(target, width, height)


class Translate:
    def __init__(
        self,
        x_frac: Tuple[float, float] = (-0.1, 0.1),
        y_frac: Tuple[float, float] = (-0.1, 0.1),
        prob: float = 0.5,
        fill: Union[int, Tuple[int, int, int]] = 0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        self.x_frac = x_frac
        self.y_frac = y_frac
        self.prob = prob
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target

        target = _ensure_target(target)
        width, height = _get_image_size(image)
        tx = int(round(random.uniform(self.x_frac[0], self.x_frac[1]) * width))
        ty = int(round(random.uniform(self.y_frac[0], self.y_frac[1]) * height))
        image = F.affine(
            image,
            angle=0.0,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=self.interpolation,
            fill=self.fill,
        )

        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, 0] += tx
            boxes[:, 2] += tx
            boxes[:, 1] += ty
            boxes[:, 3] += ty
            target["boxes"] = boxes
        return image, _filter_target(target, width, height)


class Shear:
    def __init__(
        self,
        x_degree_range: Tuple[float, float] = (-8.0, 8.0),
        y_degree_range: Tuple[float, float] = (0.0, 0.0),
        prob: float = 0.5,
        fill: Union[int, Tuple[int, int, int]] = 0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        self.x_degree_range = x_degree_range
        self.y_degree_range = y_degree_range
        self.prob = prob
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target

        target = _ensure_target(target)
        width, height = _get_image_size(image)
        sx = random.uniform(self.x_degree_range[0], self.x_degree_range[1])
        sy = random.uniform(self.y_degree_range[0], self.y_degree_range[1])
        image = F.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[sx, sy],
            interpolation=self.interpolation,
            fill=self.fill,
        )

        boxes = target["boxes"]
        if boxes.numel() > 0:
            shx = math.tan(math.radians(sx))
            shy = math.tan(math.radians(sy))
            matrix = torch.tensor([[1.0, shx], [shy, 1.0]], dtype=torch.float32)
            center = (width / 2.0, height / 2.0)
            target["boxes"] = _apply_linear_to_boxes(boxes, matrix, center)
        return image, _filter_target(target, width, height)


class RandomZoomOut:
    def __init__(
        self,
        side_range: Tuple[float, float] = (1.0, 1.4),
        prob: float = 0.3,
        fill: Union[int, Tuple[int, int, int]] = 0,
    ):
        self.side_range = side_range
        self.prob = prob
        self.fill = fill

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target

        target = _ensure_target(target)
        width, height = _get_image_size(image)
        ratio = random.uniform(self.side_range[0], self.side_range[1])
        new_w = int(round(width * ratio))
        new_h = int(round(height * ratio))

        left = random.randint(0, max(new_w - width, 0))
        top = random.randint(0, max(new_h - height, 0))
        right = new_w - width - left
        bottom = new_h - height - top

        image = F.pad(image, [left, top, right, bottom], fill=self.fill)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, 0] += left
            boxes[:, 2] += left
            boxes[:, 1] += top
            boxes[:, 3] += top
            target["boxes"] = boxes
        return image, _filter_target(target, new_w, new_h)


class RandomResizedCrop:
    def __init__(
        self,
        output_size: SizeType,
        scale: Tuple[float, float] = (0.7, 1.0),
        ratio: Tuple[float, float] = (0.8, 1.25),
        prob: float = 0.5,
        min_visibility: float = 0.25,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        self.output_size = output_size
        self.scale = scale
        self.ratio = ratio
        self.prob = prob
        self.min_visibility = min_visibility
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target

        target = _ensure_target(target)
        width, height = _get_image_size(image)
        image_area = width * height

        if isinstance(self.output_size, int):
            out_h = self.output_size
            out_w = self.output_size
        else:
            out_h, out_w = int(self.output_size[0]), int(self.output_size[1])

        boxes = target["boxes"]
        for _ in range(30):
            target_area = random.uniform(self.scale[0], self.scale[1]) * image_area
            aspect_ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))

            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < crop_w <= width and 0 < crop_h <= height:
                left = random.randint(0, width - crop_w)
                top = random.randint(0, height - crop_h)
                break
        else:
            crop_w = min(width, height)
            crop_h = crop_w
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2

        image = F.resized_crop(
            image,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[out_h, out_w],
            interpolation=self.interpolation,
        )

        if boxes.numel() > 0:
            original_boxes = boxes.clone()
            boxes = boxes.clone()
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes = _clamp_boxes(boxes, crop_w, crop_h)
            scale_x = out_w / float(crop_w)
            scale_y = out_h / float(crop_h)
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

            old_areas = (original_boxes[:, 2] - original_boxes[:, 0]) * (original_boxes[:, 3] - original_boxes[:, 1])
            new_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            visibility = new_areas / old_areas.clamp(min=1e-6)
            target["boxes"] = boxes
            return image, _filter_target(target, out_w, out_h, visibility=visibility, min_visibility=self.min_visibility)

        return image, _filter_target(target, out_w, out_h)


class ColorJitter:
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        prob: float = 0.5,
    ):
        self.prob = prob
        self.transform = tv_transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = self.transform(image)
        return image, target


class GaussianBlur:
    def __init__(self, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0), prob: float = 0.2):
        self.prob = prob
        self.transform = tv_transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = self.transform(image)
        return image, target


class RandomGrayscale:
    def __init__(self, prob: float = 0.1):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.rgb_to_grayscale(image, num_output_channels=3)
        return image, target


class RandomSharpness:
    def __init__(self, sharpness_factor_range: Tuple[float, float] = (0.5, 2.0), prob: float = 0.2):
        self.sharpness_factor_range = sharpness_factor_range
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            factor = random.uniform(self.sharpness_factor_range[0], self.sharpness_factor_range[1])
            image = F.adjust_sharpness(image, factor)
        return image, target


class ToTensor:
    def __call__(self, image, target=None):
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        return image, target


class GaussianNoise:
    def __init__(self, std_range: Tuple[float, float] = (0.01, 0.03), prob: float = 0.2):
        self.std_range = std_range
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(image) * std
        image = (image + noise).clamp(0.0, 1.0)
        return image, target


class RandomErasing:
    def __init__(
        self,
        scale: Tuple[float, float] = (0.02, 0.08),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[float, str] = 0.0,
        prob: float = 0.2,
    ):
        self.prob = prob
        self.transform = tv_transforms.RandomErasing(p=1.0, scale=scale, ratio=ratio, value=value)

    def __call__(self, image, target=None):
        if random.random() >= self.prob:
            return image, target
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        image = self.transform(image)
        return image, target


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def build_train_transforms(image_size):
    return [
        Resize((image_size, image_size)),
        HorizontalFlip(prob=0.5),
        # OneOf(
        #     [
        #         NoTransform(),
        #         Scale(ratio_range=(0.85, 1.15), prob=1.0),
        #         Translate(x_frac=(-0.08, 0.08), y_frac=(-0.08, 0.08), prob=1.0),
                 Rotate(degree_range=(-10, 10), prob=1.0),
        #         Shear(x_degree_range=(-8, 8), y_degree_range=(-4, 4), prob=1.0),
        #         RandomResizedCrop(
        #             output_size=(image_size, image_size),
        #             scale=(0.75, 1.0),
        #             ratio=(0.9, 1.1),
        #             prob=1.0,
        #             min_visibility=0.30,
        #         ),
        #         RandomZoomOut(side_range=(1.0, 1.3), prob=1.0),
        #     ],
        #     prob=0.80,
        # ),
        # OneOf(
        #     [
        #         NoTransform(),
                 ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03, prob=1.0),
                 GaussianBlur(kernel_size=3, sigma=(0.1, 1.5), prob=1.0),
        #         RandomGrayscale(prob=1.0),
        #         RandomSharpness(sharpness_factor_range=(0.7, 1.8), prob=1.0),
        #     ],
        #     prob=0.60,
        # ),
        # ToTensor(),
        # OneOf(
        #     [
        #         NoTransform(),
        #         GaussianNoise(std_range=(0.01, 0.03), prob=1.0),
        #         RandomErasing(scale=(0.02, 0.06), ratio=(0.5, 2.0), value="random", prob=1.0),
        #     ],
        #     prob=0.25,
        # ),
    ]


def build_val_transforms(image_size):
    return [
        Resize((image_size, image_size)),
        ToTensor(),
    ]
