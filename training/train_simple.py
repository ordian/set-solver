# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ultralytics",
#     "onnx",
#     "onnxruntime",
#     "onnxscript",
#     "pillow",
#     "torch",
#     "torchvision",
#     "numpy",
#     "matplotlib",
#     "svgpathtools",
#     "opencv-python",
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

import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import torch
import torch.nn as nn
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageFilter
from svgpathtools import parse_path
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ============================================
# Configuration
# ============================================
DATASET_DIR = Path("dataset")
IMG_SIZE = 224  # Higher resolution for better stripe detection

COLORS = ["red", "green", "purple"]
SHAPES = ["diamond", "oval", "squiggle"]
FILLS = ["solid", "striped", "empty"]
COUNTS = ["one", "two", "three"]

RGB_MAP = {"red": (235, 50, 35), "green": (50, 160, 75), "purple": (100, 40, 160)}

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
class SyntheticCardDataset(Dataset):
    """Generates synthetic Set cards with realistic variations."""

    def __init__(self, length=5000, img_size=224, dirty=True):
        self.length = length
        self.img_size = img_size
        self.dirty = dirty
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Random attributes
        color_idx = random.randint(0, 2)
        shape_idx = random.randint(0, 2)
        fill_idx = random.randint(0, 2)
        count_idx = random.randint(0, 2)

        color_name = COLORS[color_idx]
        shape_name = SHAPES[shape_idx]
        fill_name = FILLS[fill_idx]
        count_val = count_idx + 1

        # Random light background
        bg_color = (
            random.randint(230, 255),
            random.randint(230, 255),
            random.randint(230, 255),
        )
        bg_norm = tuple(c / 255.0 for c in bg_color)

        # Create figure
        dpi = 100
        fig = Figure(figsize=(self.img_size / dpi, self.img_size / dpi), dpi=dpi)
        fig.patch.set_facecolor(bg_norm)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.img_size)
        ax.set_ylim(self.img_size, 0)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        ax.set_facecolor(bg_norm)

        # Layout calculation
        w, h = float(self.img_size), float(self.img_size)
        scale_jitter = random.uniform(0.85, 1.05)

        shape_w = (w / 2.5) * scale_jitter
        shape_h = (h / 5.0) * scale_jitter

        center_x = w / 2.0 + random.randint(-5, 5)
        center_y = h / 2.0 + random.randint(-5, 5)

        spacing = shape_h * 1.4
        if count_val == 1:
            centers = [(center_x, center_y)]
        elif count_val == 2:
            centers = [
                (center_x, center_y - spacing / 2.0),
                (center_x, center_y + spacing / 2.0),
            ]
        else:
            centers = [
                (center_x, center_y - spacing),
                (center_x, center_y),
                (center_x, center_y + spacing),
            ]

        # Draw shapes
        rgb = RGB_MAP[color_name]
        for cx, cy in centers:
            x1 = cx - shape_w / 2.0
            y1 = cy - shape_h / 2.0
            x2 = cx + shape_w / 2.0
            y2 = cy + shape_h / 2.0
            self._add_shape_patch(
                ax, shape_name, fill_name, rgb, bg_color, (x1, y1, x2, y2)
            )

        # Render to PIL
        canvas = FigureCanvas(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        img = Image.fromarray(rgba, mode="RGBA").convert("RGB")

        # "Dirty" post-processing
        if self.dirty:
            rot = random.uniform(-5, 5)
            img = img.rotate(rot, resample=Image.BICUBIC, fillcolor=bg_color)

            if random.random() > 0.8:
                img = img.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.8))
                )

            arr = np.array(img, dtype=np.int16)
            noise = np.random.normal(0, 4, arr.shape).astype(np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")

        # Target tensor
        target = torch.zeros(4, 3)
        target[0, color_idx] = 1.0
        target[1, shape_idx] = 1.0
        target[2, fill_idx] = 1.0
        target[3, count_idx] = 1.0

        return self.transform(img), target

    @staticmethod
    def _add_shape_patch(ax, shape, fill_type, color_rgb, bg_rgb, rect):
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        color = tuple(c / 255.0 for c in color_rgb)
        bg_color = tuple(c / 255.0 for c in bg_rgb)
        linewidth = 1.4

        if fill_type == "solid":
            face = color
        elif fill_type == "striped":
            face = bg_color
        else:
            face = (0, 0, 0, 0)

        # Create patch based on shape
        if shape == "oval":
            rounding_size = h / 2.0
            patch = patches.FancyBboxPatch(
                (x1, y1),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={rounding_size}",
                edgecolor=color,
                facecolor=face,
                linewidth=linewidth,
            )
        elif shape == "diamond":
            verts = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
            patch = patches.Polygon(
                verts,
                closed=True,
                edgecolor=color,
                facecolor=face,
                linewidth=linewidth,
            )
        elif shape == "squiggle":
            path = _build_squiggle_path_in_rect(x1, y1, x2, y2)
            patch = patches.PathPatch(
                path,
                edgecolor=color,
                facecolor=face,
                linewidth=linewidth,
            )
        else:
            return

        ax.add_patch(patch)

        # Draw stripes for "striped" fill
        if fill_type == "striped":
            stripe_spacing = 4.0
            stripe_width = 1.0
            x_start = x1 - stripe_spacing * 2
            x_end = x2 + stripe_spacing * 2
            y_top = y1 - 5
            y_bottom = y2 + 5

            for x in np.arange(x_start, x_end, stripe_spacing):
                ax.plot(
                    [x, x],
                    [y_top, y_bottom],
                    color=color,
                    linewidth=stripe_width,
                    clip_path=patch,
                )


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
    """Simplified MobileNetV3 classifier - direct 576->3 heads."""

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.2)

        # Direct heads from backbone features
        self.color_head = nn.Linear(576, 3)
        self.shape_head = nn.Linear(576, 3)
        self.fill_head = nn.Linear(576, 3)
        self.count_head = nn.Linear(576, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)

        return (
            torch.softmax(self.color_head(x), dim=1),
            torch.softmax(self.shape_head(x), dim=1),
            torch.softmax(self.fill_head(x), dim=1),
            torch.softmax(self.count_head(x), dim=1),
        )

    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


# ============================================
# Training
# ============================================
def train_classifier(
    epochs_pretrain: int = 5, epochs_finetune: int = 15, batch_size: int = 32
):
    """Train classifier with synthetic pretrain + real finetune."""

    labels_path = DATASET_DIR / "classifier" / "labels.json"
    crops_dir = DATASET_DIR / "classifier" / "crops"

    if not labels_path.exists():
        print(f"Labels not found: {labels_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transforms - use BOX (INTER_AREA equivalent) for better stripe preservation
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BOX
            ),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
    synth_pretrain = SyntheticCardDataset(length=10000, img_size=IMG_SIZE, dirty=True)
    synth_loader = DataLoader(synth_pretrain, batch_size=batch_size, shuffle=True)

    model = CardClassifier(freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()

    def calc_loss(outs, targs):
        loss = 0
        for i in range(4):
            loss += criterion(torch.log(outs[i] + 1e-9), targs[:, i].argmax(dim=1))
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
            total_loss += loss.item()

        accs = validate()
        avg_acc = sum(accs) / 4
        print(
            f"Epoch {epoch + 1}/{epochs_pretrain} - Loss: {total_loss / len(synth_loader):.3f} - "
            f"Val: C={accs[0]:.0f}% S={accs[1]:.0f}% F={accs[2]:.0f}% N={accs[3]:.0f}% (avg={avg_acc:.0f}%)"
        )

    # Phase 2: Finetune on mixed real + synthetic
    print(f"\n=== Phase 2: Mixed Finetuning ({epochs_finetune} epochs) ===")
    mixed_ds = ConcatDataset(
        [
            real_train_ds,
            SyntheticCardDataset(length=3000, img_size=IMG_SIZE, dirty=True),
        ]
    )
    mixed_loader = DataLoader(mixed_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs_finetune):
        # Unfreeze backbone after 5 epochs
        if epoch == 5:
            print(">>> Unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        total_loss = 0
        for imgs, targs in mixed_loader:
            imgs, targs = imgs.to(device), targs.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = calc_loss(outs, targs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        accs = validate()
        avg_acc = sum(accs) / 4
        print(
            f"Epoch {epoch + 1}/{epochs_finetune} - Loss: {total_loss / len(mixed_loader):.3f} - "
            f"Val: C={accs[0]:.0f}% S={accs[1]:.0f}% F={accs[2]:.0f}% N={accs[3]:.0f}% (avg={avg_acc:.0f}%)"
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
        imgsz=640,
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
    ds = SyntheticCardDataset(length=16, img_size=IMG_SIZE, dirty=True)
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
        torch.load("runs/card_classifier.pt", map_location=device, weights_only=True)
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


# ============================================
# Export
# ============================================
def export():
    """Export models to ONNX."""
    from ultralytics import YOLO

    detector_path = "runs/card_detector/weights/best.pt"
    if Path(detector_path).exists():
        model = YOLO(detector_path)
        model.export(format="onnx", imgsz=640, simplify=True, opset=18)
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
        )
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
            test(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
