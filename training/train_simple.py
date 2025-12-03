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
import shutil
import sys
from functools import reduce
from operator import mul
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import timm
import torch
import torch.nn as nn
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFilter
from svgpathtools import parse_path
from timm.utils import ModelEmaV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

# from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ============================================
# Configuration
# ============================================
DATASET_DIR = Path("dataset")
# IMG_SIZE = 224
# IMG_SIZE = 112
# IMG_SIZE = 160
IMG_SIZE = 128

COLORS = ["red", "green", "purple"]
SHAPES = ["diamond", "oval", "squiggle"]
FILLS = ["solid", "striped", "empty"]
COUNTS = ["one", "two", "three"]


RGB_MAP = {
    "red": [
        (235, 30, 35),
        (200, 40, 40),
        (160, 20, 20),  # Dark Red
        (255, 100, 100),  # Faded Red
    ],
    "green": [
        (50, 205, 50),
        (30, 160, 30),
        (20, 100, 20),  # Dark Green
        (100, 200, 100),  # Faded Green
    ],
    "purple": [
        (60, 25, 140),
        (50, 20, 100),
        (50, 30, 190),  # <--- High-Blue Purple (To distinguish from Green)
        (90, 80, 180),  # Faded Purple
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
                distortion_scale=0.3,
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
        # TIGHT hue control.
        # +/- 0.03 keeps it blue-violet. +/- 0.30 was turning it pink/green.
        h += random.uniform(-0.03, 0.03)

        # Allow brightness (V) to vary, but keep saturation (S) high
        s *= random.uniform(0.9, 1.1)
        v *= random.uniform(0.7, 1.3)  # Allow dark cards
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

    def _add_sensor_noise(self, img):
        """Simulate high ISO grain (color noise)."""
        if random.random() > 0.7:
            return img

        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape

        # Grain strength
        sigma = random.uniform(5, 20)
        noise = np.random.normal(0, sigma, (h, w, c))

        arr = arr + noise
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

    def _add_glare(self, img):
        if random.random() > 0.4:
            return img
        w, h = img.size
        glare = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(glare)
        for _ in range(random.randint(1, 2)):
            x = random.randint(0, w)
            y = random.randint(0, h)
            r = random.randint(w // 6, w // 3)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=random.randint(50, 180))
        glare = glare.filter(ImageFilter.GaussianBlur(radius=random.uniform(10, 20)))
        img_arr = np.array(img).astype(np.float32)
        glare_arr = np.array(glare).astype(np.float32) / 255.0
        for c in range(3):
            img_arr[..., c] += glare_arr * 255.0 * 0.6
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr)

    # ------------------------------------------------------------
    # Draw a single shape
    # ------------------------------------------------------------
    def _draw_shape(self, ax, shape, fill_type, rgb, rect):
        x1, y1, x2, y2 = rect
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        base_color = [c / 255.0 for c in rgb]

        if fill_type == "solid":
            # Solids: slight transparency to mimic ink absorption
            alpha = random.uniform(0.90, 1.0)
            face_color = (*base_color, alpha)
            edge_color = (*base_color, 1.0)
            linewidth = 0.0
        elif fill_type == "empty":
            face_color = (0, 0, 0, 0)
            edge_color = (*base_color, 1.0)
            linewidth = random.uniform(1, 2)  # Thicker lines to be visible at 160px
        else:  # Striped
            face_color = (0, 0, 0, 0)
            edge_color = (*base_color, 1.0)
            linewidth = random.uniform(1.0, 2.0)

        # Create Patch
        if shape == "oval":
            patch = patches.FancyBboxPatch(
                (x1, y1),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                edgecolor=edge_color,
                facecolor=face_color,
                linewidth=linewidth,
            )
        elif shape == "diamond":
            verts = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
            patch = patches.Polygon(
                verts,
                closed=True,
                edgecolor=edge_color,
                facecolor=face_color,
                linewidth=linewidth,
            )
        else:
            path = _build_squiggle_path_in_rect(x1, y1, x2, y2)
            patch = patches.PathPatch(
                path, edgecolor=edge_color, facecolor=face_color, linewidth=linewidth
            )

        ax.add_patch(patch)

        if fill_type == "striped":
            # Wider spacing to prevent "Solid" confusion
            spacing = random.uniform(2.5, 5.0)

            angle_deg = random.uniform(-25, 25)
            angle_rad = np.radians(angle_deg)
            diameter = np.sqrt(w**2 + h**2) * 1.5
            num_lines = int(diameter / spacing)
            offsets = np.arange(-num_lines // 2, num_lines // 2) * spacing
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            for offset in offsets:
                lx1 = offset * cos_a - (-diameter / 2) * sin_a
                ly1 = offset * sin_a + (-diameter / 2) * cos_a
                lx2 = offset * cos_a - (diameter / 2) * sin_a
                ly2 = offset * sin_a + (diameter / 2) * cos_a
                # Use solid color for stripes, no alpha
                ax.plot(
                    [cx + lx1, cx + lx2],
                    [cy + ly1, cy + ly2],
                    color=edge_color,
                    linewidth=linewidth / 2,
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
        # 1. Independent Scaling (Fixes Shape "Fatness")
        # Allow shapes to be slightly fatter/thinner independently
        scale_w = random.uniform(0.80, 1.10)
        scale_h = random.uniform(0.80, 1.10)

        shape_w = (w / 2.6) * scale_w
        shape_h = (h / 5.2) * scale_h

        cx = w / 2 + random.randint(-10, 10)
        cy = h / 2 + random.randint(-10, 10)

        # 2. Random Spacing (Fixes Count 2 vs 3)
        # 1.4 is tight, 1.7 is loose.
        spacing_mult = random.uniform(1.35, 1.65)
        spacing = shape_h * spacing_mult

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

        # img = apply_white_balance(img)
        if random.random() < 0.6:
            img = apply_color_temperature(img)
        if random.random() < 0.3:
            img = uneven_tint(img)

        img = self._add_glare(img)
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
        # STEP 3 — Sensor noise
        # --------------------------------------------------------
        img = self._add_sensor_noise(img)

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

        # img = transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])(img)
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
            "mobilenetv4_conv_small",
            # "lcnet_100",
            pretrained=True,
            num_classes=0,
            # global_pool="",
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            self.backbone.eval()
            real_out = self.backbone(dummy)
            feats_dim = real_out.shape[1]

        # ------------------------------------------------
        # SHARED MLP for shape/fill/count
        # ------------------------------------------------
        shared_dim = 256
        self.hidden = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feats_dim, shared_dim),
            nn.Hardswish(),  # Faster/better than ReLU in MobileNets
            nn.Dropout(0.2),
        )

        color_hidden = 128
        self.head_color = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feats_dim, color_hidden),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(color_hidden, 3),
        )
        self.head_shape = nn.Linear(shared_dim, 3)
        self.head_fill = nn.Linear(shared_dim, 3)
        self.head_count = nn.Linear(shared_dim, 3)

        # freeze backbone if required
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        color = self.head_color(x)
        y = self.hidden(x)

        return (
            color,
            self.head_shape(y),
            self.head_fill(y),
            self.head_count(y),
        )


# ============================================
# Training
# ============================================
def train_classifier(
    epochs_pretrain: int = 5, epochs_finetune: int = 15, batch_size: int = 64
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
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.3, saturation=0.8, hue=0.005
            ),
            transforms.RandomAdjustSharpness(
                sharpness_factor=1.5, p=0.3
            ),  # Make some crisp
            transforms.RandomAdjustSharpness(
                sharpness_factor=0.5, p=0.3
            ),  # Make some blurry
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(15),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load real dataset
    with open(labels_path) as f:
        all_labels = json.load(f)

    all_images = list(all_labels.keys())
    random.seed(42)
    random.shuffle(all_images)
    train_size = int(0.7 * len(all_images))

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
    # EMA Setup
    model_ema = ModelEmaV2(model, decay=0.995)

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
        max_lr=5e-4,  # peak LR
        steps_per_epoch=len(synth_loader),
        epochs=epochs_pretrain + epochs_finetune,
        pct_start=0.10,  # warmup %
        div_factor=10,  # initial LR = max_lr/10
        final_div_factor=20,  # cooldown
        anneal_strategy="cos",
    )
    criterion_color = nn.CrossEntropyLoss(label_smoothing=0.15)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def calc_loss(outs, targs, finetuning=False):
        m = 2.0 if finetuning else 1.0
        loss = m * criterion_color(outs[0], targs[:, 0].argmax(dim=1))
        for i in range(1, 4):
            loss += criterion(outs[i], targs[:, i].argmax(dim=1))
        return loss

    def validate():
        test_model = model_ema.module
        test_model.eval()
        correct_attrs = [0, 0, 0, 0]
        perfect_cards = 0
        total = 0
        with torch.no_grad():
            for imgs, targs in val_loader:
                imgs, targs = imgs.to(device), targs.to(device)
                outs = test_model(imgs)

                # Stack predictions: Tuple of 4x(B, 3) -> Tensor (B, 4)
                preds = torch.stack([out.argmax(dim=1) for out in outs], dim=1)

                # Stack targets: Tensor (B, 4, 3) -> Tensor (B, 4)
                # We simply argmax the one-hot encoding at the last dimension
                trues = targs.argmax(dim=2)

                # Check individual attributes
                for i in range(4):
                    correct_attrs[i] += (preds[:, i] == trues[:, i]).sum().item()

                # Check perfect matches (rows where all columns match)
                perfect_match = (preds == trues).all(dim=1)
                perfect_cards += perfect_match.sum().item()

                total += imgs.size(0)
        return [c / total * 100 for c in correct_attrs], perfect_cards / total * 100

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
            model_ema.update(model)
            scheduler.step()
            total_loss += loss.item()
        accs, joint = validate()
        print(
            f"Epoch {epoch + 1} - Loss: {total_loss / len(synth_loader):.3f} - Val: C={accs[0]:.1f}% (Joint Perfect={joint:.1f}%)"
        )

    # Phase 2: Finetune on mixed real + synthetic
    print(f"\n=== Phase 2: Mixed Finetuning ({epochs_finetune} epochs) ===")
    mixed_ds = ConcatDataset(
        [real_train_ds] * 4 + [SyntheticCardDataset(length=1000, img_size=IMG_SIZE)]
    )
    mixed_loader = DataLoader(mixed_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs_finetune):
        # Unfreeze backbone after 5 epochs
        if epoch == 4:
            print(">>> Unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": 1e-4},
                    {"params": model.head_color.parameters(), "lr": 5e-4},
                    {"params": model.head_shape.parameters(), "lr": 3e-4},
                    {"params": model.head_fill.parameters(), "lr": 3e-4},
                    {"params": model.head_count.parameters(), "lr": 3e-4},
                ],
                weight_decay=0.01,
            )

        model.train()
        total_loss = 0
        for imgs, targs in mixed_loader:
            imgs, targs = imgs.to(device), targs.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = calc_loss(outs, targs, finetuning=True)
            loss.backward()
            optimizer.step()
            model_ema.update(model)
            scheduler.step()
            total_loss += loss.item()

        accs, joint = validate()
        print(
            f"Epoch {epoch + 1} - Loss: {total_loss / len(mixed_loader):.3f} - Val: C={accs[0]:.1f}% S={accs[1]:.1f}% F={accs[2]:.1f}% N={accs[3]:.1f}% (Joint Perfect={joint:.1f}%)"
        )

        if joint > best_acc:
            best_acc = joint
            torch.save(model_ema.module.state_dict(), "runs/card_classifier_best.pt")

    torch.save(model_ema.module.state_dict(), "runs/card_classifier.pt")
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


def analyze_failures(model_path="runs/card_classifier.pt"):
    from torchvision.utils import save_image

    device = "mps" if torch.mps.is_available() else "cpu"
    # Re-instantiate model structure
    model = CardClassifier(freeze_backbone=False)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval().to(device)

    # Load Val Data
    labels_path = DATASET_DIR / "classifier" / "labels_val.json"
    val_ds = RealCardDataset(
        DATASET_DIR / "classifier" / "crops",
        labels_path,
        transform=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Ensure this matches training
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )

    failures = []
    print(f"Analyzing {len(val_ds)} validation images...")

    with torch.no_grad():
        for i in range(len(val_ds)):
            img, targ = val_ds[i]
            img = img.to(device).unsqueeze(0)

            c, s, f, n = model(img)

            pred_c = c.argmax().item()
            pred_f = f.argmax().item()

            true_c = targ[0].argmax().item()
            true_f = targ[2].argmax().item()

            # Check for Color or Fill errors
            if pred_c != true_c or pred_f != true_f:
                # Un-normalize for saving
                inv_img = (
                    img.cpu().squeeze(0)
                    * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                    + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                )
                failures.append(inv_img)
                print(
                    f"Fail {i}: True(C={COLORS[true_c]}, F={FILLS[true_f]}) vs Pred(C={COLORS[pred_c]}, F={FILLS[pred_f]})"
                )

    if failures:
        # Save top 64 failures to a grid
        stack = torch.stack(failures[:64])
        save_image(stack, "runs/failures.jpg", nrow=8)
        print(f"Saved {len(failures)} failures to runs/failures.jpg")
    else:
        print("No failures found!")


# ============================================
# Data Cleaning Tools
# ============================================
def find_label_errors(model_path="runs/card_classifier.pt"):
    """
    1. Runs inference on ALL real crops.
    2. If Prediction != Label, saves image to 'suspicious_labels/'
    3. User reviews the folder, deleting images where the MODEL was wrong.
    """
    import shutil

    # Setup paths
    labels_path = DATASET_DIR / "classifier" / "labels.json"
    crops_dir = DATASET_DIR / "classifier" / "crops"
    out_dir = Path("suspicious_labels")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # Load Model
    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"Loading model from {model_path} on {device}...")
    model = CardClassifier(freeze_backbone=False)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval().to(device)

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load all labels
    with open(labels_path) as f:
        all_labels = json.load(f)

    print(f"Checking {len(all_labels)} images for mismatches...")

    mismatch_count = 0

    for filename, label in all_labels.items():
        img_path = crops_dir / filename
        if not img_path.exists():
            continue

        # Load and Predict
        pil_img = Image.open(img_path).convert("RGB")
        img_t = val_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            c, s, f, n = model(img_t)

        pred_color = COLORS[c.argmax().item()]
        pred_shape = SHAPES[s.argmax().item()]
        pred_fill = FILLS[f.argmax().item()]
        pred_count = COUNTS[n.argmax().item()]

        # Compare
        is_mismatch = False
        mismatch_str = []

        if pred_color != label["color"]:
            is_mismatch = True
            mismatch_str.append(f"C_{label['color']}->{pred_color}")
        if pred_shape != label["shape"]:
            is_mismatch = True
            mismatch_str.append(f"S_{label['shape']}->{pred_shape}")
        if pred_fill != label["fill"]:
            is_mismatch = True
            mismatch_str.append(f"F_{label['fill']}->{pred_fill}")
        if pred_count != label["count"]:
            is_mismatch = True
            mismatch_str.append(f"N_{label['count']}->{pred_count}")

        if is_mismatch:
            mismatch_count += 1
            # Save file with details in name for easy review
            # Format: originalname___Diffs.jpg
            diff_text = "__".join(mismatch_str)
            save_name = f"{Path(filename).stem}___{diff_text}.jpg"
            pil_img.save(out_dir / save_name)

    print(f"\nFound {mismatch_count} mismatches.")
    print(f"1. Open the '{out_dir}' folder.")
    print(
        f"2. View the images. The filename tells you the change: e.g., 'C_red->purple'."
    )
    print(f"   (This means Label was Red, Model thinks Purple)")
    print(
        f"3. If the MODEL is WRONG (the label was actually correct), DELETE the file."
    )
    print(f"4. If the MODEL is RIGHT (the label was wrong), KEEP the file.")
    print(f"5. Run 'uv run train_simple.py fix-labels' to apply the changes.")


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
            # REQUIRED for batches to work
            dynamic_axes={
                "image": {0: "batch_size"},
                "color": {0: "batch_size"},
                "shape": {0: "batch_size"},
                "fill": {0: "batch_size"},
                "count": {0: "batch_size"},
            },
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

        onnx.save(model, "runs/card_classifier_mn4s_v3.onnx")

        print(f"Classifier exported: runs/card_classifier.onnx")


def fix_labels():
    """
    Reads the remaining files in 'suspicious_labels/' and updates labels.json.
    """
    suspicious_dir = Path("suspicious_labels")
    labels_path = DATASET_DIR / "classifier" / "labels.json"

    if not suspicious_dir.exists():
        print("No suspicious_labels folder found.")
        return

    with open(labels_path) as f:
        data = json.load(f)

    # Parse filenames in folder
    files = list(suspicious_dir.glob("*.jpg"))
    if not files:
        print("No files in suspicious_labels/. Nothing to fix.")
        return

    fixed_count = 0
    for file_path in files:
        # Extract original filename and changes
        # Name format: card_0123___C_red->purple__F_solid->striped.jpg
        name_parts = file_path.stem.split("___")
        original_stem = name_parts[0]
        original_filename = original_stem + ".jpg"  # Assuming .jpg extension

        changes = name_parts[1].split("__")

        if original_filename not in data:
            print(f"Warning: {original_filename} not in labels.json")
            continue

        # Apply changes
        for change in changes:
            # Format: Type_Old->New
            # e.g. C_red->purple
            attr_code = change[0]
            val_part = change.split("_")[1]
            new_val = val_part.split("->")[1]

            if attr_code == "C":
                data[original_filename]["color"] = new_val
            elif attr_code == "S":
                data[original_filename]["shape"] = new_val
            elif attr_code == "F":
                data[original_filename]["fill"] = new_val
            elif attr_code == "N":
                data[original_filename]["count"] = new_val

        fixed_count += 1

    # Save Backup
    shutil.copy(labels_path, str(labels_path) + ".bak")

    # Save New
    with open(labels_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {fixed_count} labels in labels.json!")
    print(f"Original file backed up to labels.json.bak")

    # Clean up
    shutil.rmtree(suspicious_dir)


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
    elif cmd == "analyze-failures":
        analyze_failures()
    elif cmd == "find-errors":
        find_label_errors()
    elif cmd == "fix-labels":
        fix_labels()
    elif cmd == "test":
        if len(sys.argv) < 3:
            print("Usage: uv run train_simple.py test <image.jpg>")
        else:
            # test(sys.argv[2])
            test_onnx(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
