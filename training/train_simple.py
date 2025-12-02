# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ultralytics",
#     "onnx",
#     "onnxruntime",
#     "onnxscript",
#     "pillow",
#     "timm",
#     "torch",
#     "torchvision",
#     "numpy",
#     "matplotlib",
#     "svgpathtools",
# ]
# ///
"""
Set Card Training Pipeline

1. Train YOLOv11n to detect "card" (1 class)
2. Train MobileNet classifier with synthetic + real data

Usage:
    uv run train_simple.py setup              # Create folders
    uv run train_simple.py train-detector     # Train card detector
    uv run train_simple.py extract-crops      # Extract card crops
    uv run train_simple.py preview-synthetic  # Preview synthetic cards
    uv run train_simple.py train-classifier   # Train attribute classifier
    uv run train_simple.py export             # Export to ONNX
    uv run train_simple.py test image.jpg     # Test full pipeline
"""

import colorsys
import io
import json
import random
import sys
from asyncio.unix_events import SelectorEventLoop
from calendar import c
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import timm
import torch
import torch.nn as nn
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from svgpathtools import parse_path
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ============================================
# Configuration
# ============================================
DATASET_DIR = Path("dataset")
# IMG_SIZE = 224
IMG_SIZE = 112

COLORS = ["red", "green", "purple"]
SHAPES = ["diamond", "oval", "squiggle"]
FILLS = ["solid", "striped", "empty"]
COUNTS = ["one", "two", "three"]

RGB_MAP = {
    "red": [
        (200, 40, 30),
        (225, 55, 45),
        (175, 30, 20),
    ],
    "green": [
        (70, 150, 60),
        (95, 170, 75),
        (120, 190, 90),
    ],
    "purple": [
        (90, 40, 140),
        (120, 60, 170),
        (150, 80, 210),
        (110, 30, 130),
        (160, 70, 180),
    ],
}

# ============================================
# SVG-based squiggle path
# ============================================
SQUIGGLE_SVG_PATH_D = (
    "m 24.96,87.5883 c -4.028842,-1.913654 -5.861533,-6.601316 -6.480577,-10.769283 "
    "-0.593066,-3.91203 0.553343,-7.881287 1.961701,-11.504986 2.313821,-5.863522 "
    "8.268072,-9.372167 14.240538,-10.47872 7.16545,-0.891984 14.430571,0.46549 "
    "21.224755,2.765187 6.079117,1.990663 12.181609,4.876379 18.756587,4.126312 "
    "5.485965,-0.245631 10.649989,-2.434239 15.155645,-5.469479 2.851746,-1.935101 "
    "7.107054,-2.512992 9.676276,0.198295 3.992755,4.312468 3.716285,11.033177 "
    "1.932765,16.254436 -1.488915,4.392532 -4.889848,7.871592 -8.681654,10.373835 "
    "-7.164696,4.132495 -15.856831,4.23187 -23.829382,3.133611 -6.780895,-0.985568 "
    "-12.871585,-5.14894 -19.897083,-4.769421 -6.228877,0.0588 -12.087092,2.651818 "
    "-17.322596,5.821301 C 29.603594,88.118752 27.063551,88.755733 24.96,87.5883 Z "
    "m 6.973989,-2.035486 c 4.067507,-2.336126 8.347171,-4.473568 13.059489,-5.059618 "
    "5.661985,-1.168865 11.424552,0.269643 16.671557,2.398296 5.360749,1.936323 "
    "11.072639,2.960623 16.770453,2.477022 7.938385,0.132371 16.310562,-3.520948 "
    "20.258063,-10.677232 2.661439,-4.513386 3.538649,-10.24748 1.317009,-15.124004 "
    "-1.094121,-3.650482 -5.622641,-4.616448 -8.544227,-2.481712 -4.734207,3.013662 "
    "-9.98506,5.447742 -15.653097,5.886188 -7.203027,0.870713 -14.05676,-1.981344 "
    "-20.690547,-4.349546 -6.831446,-2.077333 -14.175566,-3.574502 -21.284954,-2.174477 "
    "-5.170914,1.225212 -9.863311,4.651212 -11.950518,9.636169 -2.741168,5.904196 "
    "-3.214228,13.80733 1.580259,18.818385 1.629512,2.172364 4.632164,2.711412 "
    "7.005099,1.4652 0.511592,-0.224077 1.003102,-0.495182 1.461414,-0.814671 z"
)

_FULL_SQUIGGLE_PATH = parse_path(SQUIGGLE_SVG_PATH_D)
_SQUIGGLE_OUTER = list(_FULL_SQUIGGLE_PATH.continuous_subpaths())[0]
_SX_MIN, _SX_MAX, _SY_MIN, _SY_MAX = _SQUIGGLE_OUTER.bbox()
_SQUIGGLE_W = _SX_MAX - _SX_MIN
_SQUIGGLE_H = _SY_MAX - _SY_MIN
_SQUIGGLE_CX = (_SX_MIN + _SX_MAX) / 2.0
_SQUIGGLE_CY = (_SY_MIN + _SY_MAX) / 2.0


def _build_squiggle_path_in_rect(
    x1: float, y1: float, x2: float, y2: float, samples_per_segment: int = 50
) -> mpath.Path:
    """Build a matplotlib.path.Path for the squiggle, scaled to fit inside rect."""
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return mpath.Path([], [])

    scale = min(w / _SQUIGGLE_W, h / _SQUIGGLE_H) * 0.9
    cx_target = (x1 + x2) / 2.0
    cy_target = (y1 + y2) / 2.0

    vertices = []
    codes = []
    first = True

    for seg in _SQUIGGLE_OUTER:
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            z = seg.point(t)
            x = (z.real - _SQUIGGLE_CX) * scale + cx_target
            y = (z.imag - _SQUIGGLE_CY) * scale + cy_target
            if first:
                codes.append(mpath.Path.MOVETO)
                first = False
            else:
                codes.append(mpath.Path.LINETO)
            vertices.append((x, y))

    if vertices:
        vertices.append(vertices[0])
        codes.append(mpath.Path.CLOSEPOLY)

    return mpath.Path(vertices, codes)


# ============================================
# Setup
# ============================================
def setup():
    """Create directory structure."""
    dirs = [
        DATASET_DIR / "detector" / "images" / "train",
        DATASET_DIR / "detector" / "images" / "val",
        DATASET_DIR / "detector" / "labels" / "train",
        DATASET_DIR / "detector" / "labels" / "val",
        DATASET_DIR / "classifier" / "crops",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    yaml = f"""path: {(DATASET_DIR / "detector").absolute()}
train: images/train
val: images/val

names:
  0: card
"""
    (DATASET_DIR / "detector" / "data.yaml").write_text(yaml)
    print("Created dataset structure.")


# ============================================
# Synthetic Card Dataset
# ============================================
# ------------------------------------------------------------
# Strong homography warp helper
# ------------------------------------------------------------
def get_perspective_coeffs(src_pts, dst_pts):
    matrix = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src_pts, dst_pts):
        matrix.append([x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src])
        matrix.append([0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src])

    A = np.array(matrix)
    B = np.array(dst_pts).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


def strong_perspective_warp(img, bg_color):
    """Applies strong perspective like real Set card photos."""
    w, h = img.size

    def jitter(pt):
        return (
            pt[0] + random.uniform(-0.10 * w, 0.10 * w),
            pt[1] + random.uniform(-0.10 * h, 0.10 * h),
        )

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [jitter(p) for p in src]
    coeffs = get_perspective_coeffs(src, dst)

    return img.transform(
        (w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=bg_color
    )


def make_tv_transform(bg_color, img_size):
    # bg_color must be tuple of 3 ints
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.12, 0.12),
                scale=(0.85, 1.10),
                shear=(-10, 10),
                interpolation=transforms.InterpolationMode.BICUBIC,
                fill=bg_color,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.5,
                p=0.7,
                fill=bg_color,
            ),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BOX,
                # RandomResizedCrop *resamples*, so no fill needed
            ),
        ]
    )


def jitter_hue_saturation(color_name):
    rgb_list = RGB_MAP[color_name]
    # pick one canonical color
    rgb = random.choice(rgb_list)

    r, g, b = [c / 255.0 for c in rgb]  # normalize to 0-1
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if color_name == "purple":
        # Real dataset: purple spans blue→violet→magenta
        h += random.uniform(-0.3, 0.3)
        s *= random.uniform(0.8, 1.2)  # very washed-out to very saturated
        v *= random.uniform(0.8, 1.2)  # dark violet to bright lavender
    else:
        # Red, green are much more stable in your images
        h += random.uniform(-0.015, 0.015)
        s *= random.uniform(0.9, 1.1)
        v *= random.uniform(0.9, 1.1)
    # convert back
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)

    return (
        int(min(max(r2 * 255, 0), 255)),
        int(min(max(g2 * 255, 0), 255)),
        int(min(max(b2 * 255, 0), 255)),
    )


def apply_white_balance(img):
    arr = np.array(img).astype(np.float32)

    # Gains similar to camera WB drift
    r_gain = random.uniform(0.97, 1.03)
    g_gain = random.uniform(0.97, 1.03)
    b_gain = random.uniform(0.97, 1.03)

    arr[..., 0] *= r_gain
    arr[..., 1] *= g_gain
    arr[..., 2] *= b_gain

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_color_temperature(img):
    kelvin_shift = random.uniform(-300, 300)

    arr = np.array(img).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    if kelvin_shift > 0:  # warm light (more red/yellow)
        r *= 1 + (kelvin_shift / 300) * 0.03
        g *= 1 + (kelvin_shift / 300) * 0.03
    else:  # cold light (more blue)
        b *= 1 + (-kelvin_shift / 300) * 0.03

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def uneven_tint(img):
    arr = np.array(img).astype(np.float32)
    h, w, _ = arr.shape

    # gradient tint like real illumination
    tint_color = np.array(
        [random.randint(200, 240), random.randint(200, 240), random.randint(200, 240)],
        dtype=np.float32,
    )

    Y, X = np.ogrid[:h, :w]
    grad = (X / w) * random.uniform(0.02, 0.10)

    arr = arr * (1 - grad[..., None]) + tint_color * grad[..., None]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------
# Synthetic Card Dataset
# ------------------------------------------------------------
class SyntheticCardDataset(Dataset):
    """Photorealistic synthetic Set cards with strong perspective + torchvision aug."""

    def __init__(self, length=5000, img_size=IMG_SIZE):
        self.length = length
        self.img_size = img_size

    def __len__(self):
        return self.length

    # --------------------------------------------------------
    # Smooth warm background generator
    # --------------------------------------------------------
    def _background(self):
        w = h = self.img_size

        # Base warm off-white
        arr = np.random.normal(213, 8, (h, w, 3)).astype(np.float32)

        texture = np.random.normal(0, 6, (h, w, 1))

        arr += texture

        bg = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if random.random() < 0.5:
            bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))

        return bg

    def _avg_color(self, img):
        return tuple(
            np.array(img).reshape(-1, 3).mean(axis=0).astype(np.uint8).tolist()
        )

    # --------------------------------------------------------
    # Directional light
    # --------------------------------------------------------
    def _directional_light(self, img):
        arr = np.array(img).astype(np.float32)
        h, w, _ = arr.shape

        # Random light direction: angle 0–360 deg
        angle = random.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Create linear light gradient
        Y, X = np.mgrid[0:h, 0:w]
        norm = X * dx + Y * dy
        norm = (norm - norm.min()) / (norm.max() - norm.min())

        # Strength: subtle 5–15%
        strength = random.uniform(0.05, 0.15)
        lightmap = 1 + (norm - 0.5) * strength

        arr = arr * lightmap[..., None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    # --------------------------------------------------------
    # Camera imperfections
    # --------------------------------------------------------
    def _camera_artifacts(self, img):
        # mild Gaussian blur
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        # add sensor noise
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 10, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        # JPEG artifacts
        if random.random() < 0.8:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(60, 95))
            img = Image.open(buf)

        return img

    # ------------------------------------------------------------
    # Draw a single shape with matplotlib (same logic you had)
    # ------------------------------------------------------------
    def _draw_shape(self, ax, shape, fill_type, rgb, rect):
        x1, y1, x2, y2 = rect
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        color = tuple(c / 255 for c in rgb)
        face = color if fill_type == "solid" else (0, 0, 0, 0)
        linewidth = 0.9

        if shape == "oval":
            patch = patches.FancyBboxPatch(
                (x1, y1),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                edgecolor=color,
                facecolor=face,
                linewidth=linewidth,
            )
        elif shape == "diamond":
            verts = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
            patch = patches.Polygon(
                verts, closed=True, edgecolor=color, facecolor=face, linewidth=linewidth
            )
        else:  # squiggle
            path = _build_squiggle_path_in_rect(x1, y1, x2, y2)
            patch = patches.PathPatch(
                path, edgecolor=color, facecolor=face, linewidth=linewidth
            )

        ax.add_patch(patch)

        if fill_type == "striped":
            spacing = random.uniform(1.5, 3.0)
            for x in np.arange(x1 - 10, x2 + 10, spacing):
                ax.plot(
                    [x, x],
                    [y1 - 5, y2 + 5],
                    color=color,
                    linewidth=0.3,
                    clip_path=patch,
                )

    # ------------------------------------------------------------
    # Main generator
    # ------------------------------------------------------------
    def __getitem__(self, idx):
        # Sample attributes
        color_idx = random.randint(0, 2)
        shape_idx = random.randint(0, 2)
        fill_idx = random.randint(0, 2)
        count_idx = random.randint(0, 2)

        color_name = COLORS[color_idx]
        shape_name = SHAPES[shape_idx]
        fill_name = FILLS[fill_idx]
        count_val = count_idx + 1

        # --------------------------------------------------------
        # STEP 1 — Render clean card via matplotlib
        # --------------------------------------------------------
        fig = Figure(figsize=(self.img_size / 100, self.img_size / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.img_size)
        ax.set_ylim(self.img_size, 0)
        ax.axis("off")

        rgb = jitter_hue_saturation(color_name)

        w = h = self.img_size
        scale = random.uniform(0.85, 1.05)
        shape_w = (w / 2.6) * scale
        shape_h = (h / 5.2) * scale

        cx = w / 2 + random.randint(-10, 10)
        cy = h / 2 + random.randint(-10, 10)
        spacing = shape_h * 1.45

        if count_val == 1:
            centers = [(cx, cy)]
        elif count_val == 2:
            centers = [(cx, cy - spacing / 2), (cx, cy + spacing / 2)]
        else:
            centers = [(cx, cy - spacing), (cx, cy), (cx, cy + spacing)]

        for x, y in centers:
            x1 = x - shape_w / 2
            y1 = y - shape_h / 2
            x2 = x + shape_w / 2
            y2 = y + shape_h / 2
            self._draw_shape(ax, shape_name, fill_name, rgb, (x1, y1, x2, y2))

        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()), "RGBA").convert("RGB")

        # remove 1px transparency fringe
        img_arr = np.array(img)
        img_arr = np.where(img_arr < 3, 255, img_arr)  # lift transparent edge
        img = Image.fromarray(img_arr)

        blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
        img = Image.blend(img, blurred, alpha=0.08)

        # --------------------------------------------
        # Card border
        # --------------------------------------------
        outline = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(outline)
        draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0, 20), width=2)
        img = Image.alpha_composite(img.convert("RGBA"), outline).convert("RGB")

        # img = apply_white_balance(img)
        # img = apply_color_temperature(img)
        # img = uneven_tint(img)

        # --------------------------------------------
        # Background
        # --------------------------------------------
        bg = self._background()
        fill_color = self._avg_color(bg)

        bg.paste(img, (0, 0))

        # --------------------------------------------------------
        # STEP 2 — Strong homography
        # --------------------------------------------------------
        img = strong_perspective_warp(img, fill_color)

        if random.random() < 0.5:
            img = img.transform(
                img.size,
                Image.QUAD,
                data=[
                    0 + random.randint(-3, 3),
                    0,
                    w + random.randint(-3, 3),
                    0,
                    w,
                    h,
                    0,
                    h,
                ],
                resample=Image.BICUBIC,
            )

        # --------------------------------------------------------
        # STEP 4 — Camera-like imperfections
        # --------------------------------------------------------
        img = self._camera_artifacts(img)

        # --------------------------------------------------------
        # STEP 5 — Final torchvision transforms
        # --------------------------------------------------------
        tv = make_tv_transform(fill_color, self.img_size)
        img = tv(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        # --------------------------------------------------------
        # Targets
        # --------------------------------------------------------
        target = torch.zeros(4, 3)
        target[0, color_idx] = 1
        target[1, shape_idx] = 1
        target[2, fill_idx] = 1
        target[3, count_idx] = 1

        return img, target


# ============================================
# Real Card Dataset
# ============================================
class RealCardDataset(Dataset):
    """Dataset for real card crops."""

    def __init__(self, crops_dir: Path, labels_path: Path, transform=None):
        self.crops_dir = crops_dir
        self.transform = transform

        with open(labels_path) as f:
            self.labels = json.load(f)
        self.images = list(self.labels.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(self.crops_dir / img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[img_name]

        target = torch.zeros(4, 3)
        target[0, COLORS.index(label["color"])] = 1.0
        target[1, SHAPES.index(label["shape"])] = 1.0
        target[2, FILLS.index(label["fill"])] = 1.0
        target[3, COUNTS.index(label["count"])] = 1.0

        return img, target


# ============================================
# Model
# ============================================
class CardClassifier(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        self.backbone = timm.create_model(
            # "efficientformerv2_s0",
            # "mobileone_s1",
            # "mobilenetv4_conv_small",
            "lcnet_100",
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        # ------------------------------------------------
        # PROBE MODEL STRUCTURE
        # ------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            block_outputs = []
            x = self.backbone.conv_stem(dummy)

            # MobileNetV3 uses .bn1, LCNet also uses .bn1
            if hasattr(self.backbone, "bn1"):
                x = self.backbone.bn1(x)

            for blk in self.backbone.blocks:
                x = blk(x)
                block_outputs.append(x.shape[1:])  # (C,H,W)

        self.blocks_out = block_outputs

        # pick color stage at H <= 32
        self.stage_color = next(
            i for i, (_, h, _) in enumerate(self.blocks_out) if h <= 32
        )
        self.color_in_ch = self.blocks_out[self.stage_color][0]

        # last block index
        self.stage_final = len(self.blocks_out) - 1

        # ------------------------------------------------
        # Compute final feature dimension
        # ------------------------------------------------
        with torch.no_grad():
            _, final = self._extract_stages(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
            pooled = final.mean(dim=[2, 3])
            self.final_dim = pooled.shape[1]

        # ------------------------------------------------
        # COLOR HEAD (mid-level branch)
        # ------------------------------------------------
        self.color_conv = nn.Sequential(
            nn.Conv2d(self.color_in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head_color = nn.Linear(32, 3)

        # ------------------------------------------------
        # SHARED MLP for shape/fill/count
        # ------------------------------------------------
        self.hidden = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(self.final_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.head_shape = nn.Linear(256, 3)
        self.head_fill = nn.Linear(256, 3)
        self.head_count = nn.Linear(256, 3)

        # freeze backbone if required
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # ------------------------------------------------
    # UNIVERSAL FEATURE EXTRACTOR
    # ------------------------------------------------
    def _extract_stages(self, x):
        x = self.backbone.conv_stem(x)

        if hasattr(self.backbone, "bn1"):
            x = self.backbone.bn1(x)

        early = None
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i == self.stage_color:
                early = x

        # Now x is the last block output → pass through conv_head
        if hasattr(self.backbone, "conv_head"):
            x = self.backbone.conv_head(x)

        # MobileNetV3 has act2; LCNet does not
        if hasattr(self.backbone, "act2") and self.backbone.act2 is not None:
            x = self.backbone.act2(x)

        final = x
        return early, final

    # ------------------------------------------------
    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------
    def forward(self, x):
        early, final = self._extract_stages(x)

        color_feats = self.color_conv(early).flatten(1)
        pooled = final.mean(dim=[2, 3])
        shared = self.hidden(pooled)

        return (
            self.head_color(color_feats),
            self.head_shape(shared),
            self.head_fill(shared),
            self.head_count(shared),
        )

    #     # weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    #     # self.backbone = mobilenet_v3_small(weights=weights)
    #     # feat_dim = 0
    #     feat_dim = self.backbone.num_features
    #     self.pool = nn.AdaptiveAvgPool2d(1)

    #     # -----------------------------
    #     # Shared bottleneck MLP
    #     # -----------------------------
    #     hidden_in = feat_dim
    #     hidden_out = 256
    #     self.hidden = nn.Sequential(
    #         nn.Dropout(0.2),
    #         nn.Linear(hidden_in, hidden_out),
    #         nn.ReLU(inplace=True),
    #         # nn.Dropout(0.25),
    #     )
    #     # ---------------------------------------------------
    #     # Separate color head
    #     # ---------------------------------------------------
    #     self.head_color = nn.Sequential(
    #         nn.Linear(feat_dim, hidden_out),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(hidden_out, 3),
    #     )
    #     # Other heads: shape, fill, count
    #     self.head_shape = nn.Linear(hidden_out, 3)
    #     self.head_fill = nn.Linear(hidden_out, 3)
    #     self.head_count = nn.Linear(hidden_out, 3)

    #     if freeze_backbone:
    #         for p in self.backbone.parameters():
    #             p.requires_grad = False

    # def unfreeze_backbone(self):
    #     for p in self.backbone.parameters():
    #         p.requires_grad = True

    # def forward(self, x):
    #     feats = self.backbone.forward_features(x)
    #     pooled = self.pool(feats).flatten(1)
    #     shared = self.hidden(pooled)
    #     return (
    #         self.head_color(pooled),
    #         self.head_shape(shared),
    #         self.head_fill(shared),
    #         self.head_count(shared),
    #     )


# ============================================
# Training
# ============================================
def train_classifier(
    epochs_pretrain: int = 5, epochs_finetune: int = 15, batch_size: int = 16
):
    """Train classifier with synthetic pretrain + real finetune."""

    labels_path = DATASET_DIR / "classifier" / "labels.json"
    crops_dir = DATASET_DIR / "classifier" / "crops"

    if not labels_path.exists():
        print(f"Labels not found: {labels_path}")
        return

    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transforms - use BOX (INTER_AREA equivalent) for better stripe preservation
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BOX
            ),
            transforms.RandomRotation(15),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BOX
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load real dataset
    with open(labels_path) as f:
        all_labels = json.load(f)

    all_images = list(all_labels.keys())
    train_size = int(0.8 * len(all_images))

    train_labels = {k: all_labels[k] for k in all_images[:train_size]}
    val_labels = {k: all_labels[k] for k in all_images[train_size:]}

    train_labels_path = DATASET_DIR / "classifier" / "labels_train.json"
    val_labels_path = DATASET_DIR / "classifier" / "labels_val.json"
    train_labels_path.write_text(json.dumps(train_labels))
    val_labels_path.write_text(json.dumps(val_labels))

    real_train_ds = RealCardDataset(crops_dir, train_labels_path, train_transform)
    real_val_ds = RealCardDataset(crops_dir, val_labels_path, val_transform)

    val_loader = DataLoader(real_val_ds, batch_size=batch_size)

    # Synthetic dataset
    synth_pretrain = SyntheticCardDataset(length=8000, img_size=IMG_SIZE)
    synth_loader = DataLoader(synth_pretrain, batch_size=batch_size, shuffle=True)

    model = CardClassifier(freeze_backbone=True).to(device)

    # --------------------------------------
    # Learning rate scheduling (warmup + cosine)
    # --------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=7e-4,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,  # peak LR
        steps_per_epoch=len(synth_loader),
        epochs=epochs_pretrain + epochs_finetune,
        pct_start=0.10,  # warmup %
        div_factor=10,  # initial LR = max_lr/10
        final_div_factor=20,  # cooldown
        anneal_strategy="cos",
    )
    criterion_color = nn.CrossEntropyLoss(label_smoothing=0.15)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def calc_loss(outs, targs):
        loss = 2.5 * criterion_color(outs[0], targs[:, 0].argmax(dim=1))
        for i in range(1, 4):
            loss += criterion(outs[i], targs[:, i].argmax(dim=1))
        return loss

    def validate():
        model.eval()
        correct = [0, 0, 0, 0]
        total = 0
        with torch.no_grad():
            for imgs, targs in val_loader:
                imgs, targs = imgs.to(device), targs.to(device)
                outs = model(imgs)
                for i in range(4):
                    pred = outs[i].argmax(dim=1)
                    true = targs[:, i].argmax(dim=1)
                    correct[i] += (pred == true).sum().item()
                total += imgs.size(0)
        return [c / total * 100 for c in correct]

    Path("runs").mkdir(exist_ok=True)
    best_acc = 0

    # Phase 1: Pretrain on synthetic data
    print(f"\n=== Phase 1: Synthetic Pretraining ({epochs_pretrain} epochs) ===")
    for epoch in range(epochs_pretrain):
        model.train()
        total_loss = 0
        for imgs, targs in synth_loader:
            imgs, targs = imgs.to(device), targs.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = calc_loss(outs, targs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        accs = validate()
        avg_acc = sum(accs) / 4
        print(
            f"Epoch {epoch + 1}/{epochs_pretrain} - Loss: {total_loss / len(synth_loader):.3f} - "
            f"Val: C={accs[0]:.1f}% S={accs[1]:.1f}% F={accs[2]:.1f}% N={accs[3]:.1f}% (avg={avg_acc:.1f}%)"
        )

    # Phase 2: Finetune on mixed real + synthetic
    print(f"\n=== Phase 2: Mixed Finetuning ({epochs_finetune} epochs) ===")
    mixed_ds = ConcatDataset(
        [
            real_train_ds,
            SyntheticCardDataset(length=600, img_size=IMG_SIZE),
        ]
    )
    mixed_loader = DataLoader(mixed_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs_finetune):
        # Unfreeze backbone after 5 epochs
        if epoch == 4:
            print(">>> Unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": 8e-5},
                    {"params": model.head_color.parameters(), "lr": 3e-4},
                    {"params": model.head_shape.parameters(), "lr": 2e-4},
                    {"params": model.head_fill.parameters(), "lr": 2e-4},
                    {"params": model.head_count.parameters(), "lr": 2e-4},
                ],
                weight_decay=0.01,
            )

        model.train()
        total_loss = 0
        for imgs, targs in mixed_loader:
            imgs, targs = imgs.to(device), targs.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = calc_loss(outs, targs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        accs = validate()
        avg_acc = sum(accs) / 4
        print(
            f"Epoch {epoch + 1}/{epochs_finetune} - Loss: {total_loss / len(mixed_loader):.3f} - "
            f"Val: C={accs[0]:.1f}% S={accs[1]:.1f}% F={accs[2]:.1f}% N={accs[3]:.1f}% (avg={avg_acc:.1f}%)"
        )

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), "runs/card_classifier_best.pt")

    torch.save(model.state_dict(), "runs/card_classifier.pt")
    print(f"\nTraining complete. Best accuracy: {best_acc:.1f}%")


# ============================================
# YOLO Detector Training
# ============================================
def train_detector(epochs: int = 50):
    """Train YOLOv11n to detect cards."""
    from ultralytics import YOLO

    data_yaml = DATASET_DIR / "detector" / "data.yaml"
    train_labels = list((DATASET_DIR / "detector" / "labels" / "train").glob("*.txt"))

    if not train_labels:
        print("No labels found!")
        return

    print(f"Training detector on {len(train_labels)} images...")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=4,
        imgsz=(480, 640),
        device="0" if torch.cuda.is_available() else "cpu",
        project="runs",
        name="card_detector",
        patience=15,
    )
    print("Detector trained: runs/card_detector/weights/best.pt")


def extract_crops(model_path: str = "runs/card_detector/weights/best.pt"):
    """Extract card crops using detector."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    crops_dir = DATASET_DIR / "classifier" / "crops"

    images = list((DATASET_DIR / "detector" / "images" / "train").glob("*"))

    crop_id = 0
    for img_path in images:
        results = model.predict(str(img_path), conf=0.5, verbose=False)
        img = Image.open(img_path)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = img.crop((x1, y1, x2, y2))
            crop.save(crops_dir / f"card_{crop_id:04d}.jpg")
            crop_id += 1

    print(f"Extracted {crop_id} card crops to {crops_dir}/")


# ============================================
# Preview Synthetic
# ============================================
def preview_synthetic():
    """Generate preview grid of synthetic cards."""
    print("Generating synthetic preview...")
    ds = SyntheticCardDataset(length=16, img_size=IMG_SIZE)
    grid_img = Image.new("RGB", (IMG_SIZE * 4, IMG_SIZE * 4))

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    for i in range(16):
        tensor, target = ds[i]
        img_t = inv_normalize(tensor)
        img_pil = transforms.ToPILImage()(img_t.clamp(0, 1))
        x, y = (i % 4) * IMG_SIZE, (i // 4) * IMG_SIZE
        grid_img.paste(img_pil, (x, y))

    grid_img.save("synthetic_preview.jpg")
    print("Saved 'synthetic_preview.jpg'")


# ============================================
# Test Pipeline
# ============================================
def test(image_path: str):
    """Test full detection + classification pipeline."""
    from PIL import ImageDraw
    from ultralytics import YOLO

    detector = YOLO("runs/card_detector/weights/best.pt")

    device = "cpu"
    model = CardClassifier(freeze_backbone=False)
    model.load_state_dict(
        torch.load(
            "runs/card_classifier_mobileone_s1.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BOX
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    results = detector.predict(image_path, conf=0.5)

    print(f"\nDetected {len(results[0].boxes)} cards:")

    draw = ImageDraw.Draw(img)
    color_map = {"red": "#FF4444", "green": "#44FF44", "purple": "#AA44FF"}

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img.crop((x1, y1, x2, y2))

        inp = transform(crop).unsqueeze(0)
        with torch.no_grad():
            color_out, shape_out, fill_out, count_out = model(inp)

        color = COLORS[color_out.argmax().item()]
        shape = SHAPES[shape_out.argmax().item()]
        fill = FILLS[fill_out.argmax().item()]
        count = count_out.argmax().item() + 1

        box_color = color_map.get(color, "#FFFFFF")
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        label = f"{i + 1}"
        bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(
            [x1, y1 - bbox[3] + bbox[1] - 6, x1 + bbox[2] - bbox[0] + 6, y1],
            fill=box_color,
        )
        draw.text((x1 + 3, y1 - bbox[3] + bbox[1] - 3), label, fill="black")

        print(f"  Card {i + 1}: {count} {color} {shape} {fill}")

    output_path = Path(image_path).stem + "_annotated.jpg"
    img.save(output_path)
    print(f"\nAnnotated image saved: {output_path}")


def test_onnx(image_path: str):
    """Test full detection + classification pipeline using ONNX models."""
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from PIL import Image, ImageDraw

    # ----------------------------
    # Load ONNX models
    # ----------------------------
    yolo_sess = ort.InferenceSession(
        "../docs/segmentationv3.onnx", providers=["CPUExecutionProvider"]
    )
    clf_sess = ort.InferenceSession(
        "../docs/classificationv2.onnx", providers=["CPUExecutionProvider"]
    )
    print("Cls Inputs: ", clf_sess.get_inputs()[0].shape)

    # ----------------------------
    # Utility: preprocess for YOLO
    # Must match your 640x640 or 640x480 export size
    # ----------------------------
    YOLO_W, YOLO_H = 640, 480  # adjust if your detector export shape is different

    def yolo11_decode_cxcywh(raw, conf_thresh=0.5):
        raw = np.asarray(raw, dtype=np.float32).squeeze()  # (5,6300)

        cx = raw[0]
        cy = raw[1]
        w = raw[2]
        h = raw[3]
        conf = raw[4]

        keep = conf > conf_thresh

        cx = cx[keep]
        cy = cy[keep]
        w = w[keep]
        h = h[keep]
        conf = conf[keep]

        # convert center to corner
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # filter out invalid shapes
        valid = (x2 > x1) & (y2 > y1)
        x1, y1, x2, y2, conf = x1[valid], y1[valid], x2[valid], y2[valid], conf[valid]

        boxes = np.stack([x1, y1, x2, y2, conf, np.zeros_like(conf)], axis=1)
        return boxes

    def nms(boxes, iou_thresh=0.45):
        if len(boxes) == 0:
            return boxes

        b = boxes.copy()
        x1, y1, x2, y2, score, cls = b.T

        idxs = np.argsort(-score)
        keep = []

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_j = (x2[idxs[1:]] - x1[idxs[1:]]) * (y2[idxs[1:]] - y1[idxs[1:]])
            iou = inter / (area_i + area_j - inter + 1e-9)

            idxs = idxs[1:][iou < iou_thresh]

        return b[keep]

    def preprocess_yolo(img):
        """Resize + normalize to shape (1,3,H,W)."""
        img_resized = img.resize((YOLO_W, YOLO_H))
        arr = np.array(img_resized).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        arr = np.expand_dims(arr, axis=0)
        return arr

    # ----------------------------
    # Utility: preprocess crops for classifier
    # ----------------------------
    # IMG_SIZE = 224

    def preprocess_classifier(crop):
        crop = crop.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(crop).astype(np.float32) / 255.0

        # Normalize using ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        arr = arr.transpose(2, 0, 1)  # CHW
        arr = np.expand_dims(arr, axis=0)
        return arr

    # ----------------------------
    # Inference: YOLO detector
    # ----------------------------
    img = Image.open(image_path).convert("RGB")
    yolo_inp = preprocess_yolo(img)

    # YOLO ONNX input name depends on model
    yolo_input_name = yolo_sess.get_inputs()[0].name
    yolo_out = yolo_sess.run(None, {yolo_input_name: yolo_inp})

    decoded = yolo11_decode_cxcywh(yolo_out, conf_thresh=0.6)
    print("After threshold:", decoded.shape)
    boxes = nms(decoded, iou_thresh=0.3)
    print("After NMS:", boxes.shape)

    draw = ImageDraw.Draw(img)
    color_map = {"red": "#FF4444", "green": "#44FF44", "purple": "#AA44FF"}

    # Replace with your mapping lists

    for i, (x1, y1, x2, y2, conf, cls) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        crop = img.crop((x1, y1, x2, y2))

        # ----------------------------
        # Classifier inference
        # ----------------------------
        inp = preprocess_classifier(crop)
        inp_name = clf_sess.get_inputs()[0].name

        # Run classifier ONNX
        out_color, out_shape, out_fill, out_count = clf_sess.run(None, {inp_name: inp})

        # Argmax each head
        color_idx = int(np.argmax(out_color))
        shape_idx = int(np.argmax(out_shape))
        fill_idx = int(np.argmax(out_fill))
        count_idx = int(np.argmax(out_count)) + 1  # counts 1..3

        color = COLORS[color_idx]
        shape = SHAPES[shape_idx]
        fill = FILLS[fill_idx]

        box_color = color_map.get(color, "#FFFFFF")
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        label = f"{i + 1}"
        bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(
            [x1, y1 - (bbox[3] - bbox[1]) - 6, x1 + (bbox[2] - bbox[0]) + 6, y1],
            fill=box_color,
        )
        draw.text((x1 + 3, y1 - (bbox[3] - bbox[1]) - 3), label, fill="black")

        print(f"  Card {i + 1}: {count_idx} {color} {shape} {fill}")

    output_path = Path(image_path).stem + "_onnx_annotated.jpg"
    img.save(output_path)
    print(f"\nAnnotated image saved: {output_path}")


# ============================================
# Export
# ============================================
def export():
    """Export models to ONNX."""
    from ultralytics import YOLO

    detector_path = "runs/card_detector/weights/best.pt"
    if Path(detector_path).exists():
        model = YOLO(detector_path)
        model.export(
            format="onnx",
            imgsz=(480, 640),
            opset=18,
            nms=False,  # Web backend doesn't support the instructions :(
            simplify=True,
        )
        print(f"Detector exported")

    classifier_path = "runs/card_classifier.pt"
    if Path(classifier_path).exists():
        model = CardClassifier(freeze_backbone=False)
        model.load_state_dict(
            torch.load(classifier_path, map_location="cpu", weights_only=True)
        )
        model.eval()

        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        torch.onnx.export(
            model,
            dummy,
            "runs/card_classifier.onnx",
            input_names=["image"],
            output_names=["color", "shape", "fill", "count"],
            opset_version=18,
            export_params=True,
            # do_constant_folding=True,
            # keep_initializers_as_inputs=False,
        )
        import onnx
        from onnx import numpy_helper

        model = onnx.load("runs/card_classifier.onnx", load_external_data=True)

        # Force all initializers to embed raw data
        for init in model.graph.initializer:
            if init.data_location == onnx.TensorProto.EXTERNAL:
                # Load the external data into raw_data
                array = numpy_helper.to_array(init)
                init.ClearField("external_data")
                init.data_location = onnx.TensorProto.DEFAULT
                init.raw_data = array.tobytes()

        onnx.save(model, "runs/card_classifier_single.onnx")

        print(f"Classifier exported: runs/card_classifier.onnx")


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "setup":
        setup()
    elif cmd == "train-detector":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        train_detector(epochs)
    elif cmd == "extract-crops":
        extract_crops()
    elif cmd == "preview-synthetic":
        preview_synthetic()
    elif cmd == "train-classifier":
        epochs_pt = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        epochs_ft = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        train_classifier(epochs_pt, epochs_ft)
    elif cmd == "export":
        export()
    elif cmd == "test":
        if len(sys.argv) < 3:
            print("Usage: uv run train_simple.py test <image.jpg>")
        else:
            # test(sys.argv[2])
            test_onnx(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
