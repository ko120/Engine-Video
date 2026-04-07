"""
Reliable Pseudo-Label Generation with YOLO + SAM3.

Pipeline:
  1. Extract frames from video.
  2. Run YOLO on key frames (every N frames); apply confidence filter,
     per-class NMS, and size/aspect-ratio checks.
  3. For each key frame, initialise SAM3 tracks from accepted YOLO boxes
     and propagate ±W frames.
  4. Validate each candidate track: length, mean YOLO confidence at key
     frames, temporal IoU, max consecutive gap, matched-keyframe count,
     crop-classifier agreement (optional), and geometry/motion sanity.
  5. Merge overlapping duplicate tracks (keep highest-scoring one).
  6. Export accepted pseudo-labels in YOLO format for re-training and
     JSON format with original raw class IDs.

Key rule:
  YOLO decides WHAT the object is.
  SAM3 decides WHERE it continues over time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision.ops import nms as torchvision_nms
from ultralytics import YOLO

try:
    # Public API shown in Ultralytics SAM3 docs
    from ultralytics.models.sam import SAM3VideoPredictor
    SAM3_AVAILABLE = True
except Exception:
    SAM3_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

TRACK_CLASSES = [0, 1, 2, 3, 36]  # COCO/raw IDs: person, bicycle, car, motorcycle, skateboard
CLASS_NAMES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    36: "skateboard",
}

# For YOLO training export only: contiguous IDs required by training datasets.
TRAIN_ID_MAP = {
    0:  0,
    1:  1,
    2:  2,
    3:  3,
    36: 4,
}
TRAIN_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "skateboard",
}

# Per-class (min_w/h, max_w/h) aspect-ratio limits
CLASS_ASPECT_LIMITS: dict[int, tuple[float, float]] = {
    0:  (0.15, 1.50),  # person
    1:  (0.50, 3.00),  # bicycle
    2:  (0.50, 4.00),  # car
    3:  (0.40, 3.00),  # motorcycle
    36: (0.30, 4.00),  # skateboard
}

MIN_BOX_AREA_FRAC = 5e-4
MAX_BOX_AREA_FRAC = 0.90


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Detection:
    frame_idx: int
    box: np.ndarray  # [x1, y1, x2, y2] float32
    cls: int
    conf: float


@dataclass
class TrackFrame:
    frame_idx: int
    box: np.ndarray  # [x1, y1, x2, y2]
    conf: float = 0.0  # matched YOLO conf at this frame (0 if no match)


@dataclass
class CandidateTrack:
    obj_id: int
    cls: int
    init_conf: float
    key_frame: int
    frames: list[TrackFrame] = field(default_factory=list)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def clip_box(box: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = np.clip(x1, 0, img_w)
    x2 = np.clip(x2, 0, img_w)
    y1 = np.clip(y1, 0, img_h)
    y2 = np.clip(y2, 0, img_h)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    """Binary mask → tight box [x1, y1, x2, y2], or None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2 + 1, y2 + 1], dtype=np.float32)


def box_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)


# ── Detection filtering ───────────────────────────────────────────────────────

def _per_class_nms(dets: list[Detection], iou_thresh: float) -> list[Detection]:
    if not dets:
        return []
    result = []
    for cls in sorted({d.cls for d in dets}):
        cd = [d for d in dets if d.cls == cls]
        boxes = torch.tensor(np.stack([d.box for d in cd]), dtype=torch.float32)
        scores = torch.tensor([d.conf for d in cd], dtype=torch.float32)
        keep = torchvision_nms(boxes, scores, iou_thresh).tolist()
        result.extend(cd[i] for i in keep)
    return result


def filter_detections(
    dets: list[Detection],
    img_h: int,
    img_w: int,
    tau_det: float,
    tau_nms: float,
) -> list[Detection]:
    """Apply confidence threshold, per-class NMS, size, and aspect checks."""
    img_area = img_h * img_w

    dets = [d for d in dets if d.conf >= tau_det]
    dets = _per_class_nms(dets, tau_nms)

    out = []
    for d in dets:
        box = clip_box(d.box, img_h, img_w)
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        area_frac = (bw * bh) / img_area
        if not (MIN_BOX_AREA_FRAC <= area_frac <= MAX_BOX_AREA_FRAC):
            continue
        ar_min, ar_max = CLASS_ASPECT_LIMITS.get(d.cls, (0.1, 10.0))
        ar = bw / bh
        if not (ar_min <= ar <= ar_max):
            continue
        d.box = box
        out.append(d)
    return out


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, frame_dir: str) -> tuple[int, int, float, int]:
    """Write every frame of video_path to frame_dir as 000000.jpg, ...
    Returns (height, width, fps, n_frames)."""
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idx = 0
    with tqdm(total=n_total, desc="extract frames", unit="f") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(
                os.path.join(frame_dir, f"{idx:06d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            idx += 1
            pbar.update(1)

    cap.release()
    return img_h, img_w, fps, idx


# ── Track statistics and validation ──────────────────────────────────────────

def _track_stats(track: CandidateTrack) -> dict:
    frames = track.frames
    if not frames:
        return {
            "track_len": 0,
            "mean_conf": 0.0,
            "temporal_iou": 0.0,
            "max_gap": 999,
            "matched_kf": 0,
        }

    track_len = len(frames)
    matched = [f.conf for f in frames if f.conf > 0]
    mean_conf = float(np.mean(matched)) if matched else track.init_conf

    tiou_vals = [
        box_iou(frames[i - 1].box, frames[i].box)
        for i in range(1, len(frames))
    ]
    temporal_iou = float(np.mean(tiou_vals)) if tiou_vals else 1.0

    idxs = sorted(f.frame_idx for f in frames)
    max_gap = max((idxs[i] - idxs[i - 1] - 1 for i in range(1, len(idxs))), default=0)

    return {
        "track_len": track_len,
        "mean_conf": mean_conf,
        "temporal_iou": temporal_iou,
        "max_gap": max_gap,
        "matched_kf": len(matched),
    }


def _motion_ok(
    frames: list[TrackFrame],
    img_diag: float,
    max_jump_frac: float = 0.35,
) -> bool:
    for i in range(1, len(frames)):
        c1x, c1y = box_center(frames[i - 1].box)
        c2x, c2y = box_center(frames[i].box)
        if np.hypot(c2x - c1x, c2y - c1y) > max_jump_frac * img_diag:
            return False
    return True


def _area_ok(frames: list[TrackFrame], img_area: float) -> bool:
    for f in frames:
        b = f.box
        frac = (b[2] - b[0]) * (b[3] - b[1]) / img_area
        if not (MIN_BOX_AREA_FRAC <= frac <= MAX_BOX_AREA_FRAC):
            return False
    return True


def _aspect_ok(track: CandidateTrack) -> bool:
    ar_min, ar_max = CLASS_ASPECT_LIMITS.get(track.cls, (0.1, 10.0))
    for f in track.frames:
        bw = f.box[2] - f.box[0]
        bh = f.box[3] - f.box[1]
        if bw <= 0 or bh <= 0:
            return False
        ar = bw / bh
        if not (ar_min <= ar <= ar_max):
            return False
    return True


def _validate_track(
    track: CandidateTrack,
    img_h: int,
    img_w: int,
    tau_len: int,
    tau_conf: float,
    tau_tiou: float,
    max_gap: int,
    min_matched_keyframes: int,
    crop_classifier: Optional[Callable],
    confusing_classes: set[int],
    frame_dir: str,
) -> bool:
    stats = _track_stats(track)

    if stats["track_len"] < tau_len:
        return False
    if stats["mean_conf"] < tau_conf:
        return False
    if stats["temporal_iou"] < tau_tiou:
        return False
    if stats["max_gap"] > max_gap:
        return False
    if stats["matched_kf"] < min_matched_keyframes:
        return False

    if not _area_ok(track.frames, img_h * img_w):
        return False
    if not _motion_ok(track.frames, np.hypot(img_h, img_w)):
        return False
    if not _aspect_ok(track):
        return False

    # Optional crop-classifier for confusing classes
    if crop_classifier is not None and track.cls in confusing_classes:
        mid = track.frames[len(track.frames) // 2]
        img_bgr = cv2.imread(os.path.join(frame_dir, f"{mid.frame_idx:06d}.jpg"))
        if img_bgr is not None:
            x1, y1, x2, y2 = mid.box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_bgr.shape[1], x2)
            y2 = min(img_bgr.shape[0], y2)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size > 0 and crop_classifier(crop) != track.cls:
                return False

    return True


# ── Duplicate merging ─────────────────────────────────────────────────────────

def _track_score(track: CandidateTrack) -> float:
    s = _track_stats(track)
    return s["track_len"] * max(s["temporal_iou"], 1e-6) * max(s["mean_conf"], 1e-6)


def merge_duplicate_tracks(
    tracks: list[CandidateTrack],
    iou_thresh: float = 0.50,
    min_common_frames: int = 2,
) -> list[CandidateTrack]:
    """Keep highest-scoring track when same-class tracks overlap enough."""
    tracks = sorted(tracks, key=_track_score, reverse=True)
    kept: list[CandidateTrack] = []

    for track in tracks:
        t_boxes = {f.frame_idx: f.box for f in track.frames}
        duplicate = False

        for ktrack in kept:
            if ktrack.cls != track.cls:
                continue

            k_boxes = {f.frame_idx: f.box for f in ktrack.frames}
            common = sorted(set(t_boxes) & set(k_boxes))
            if len(common) < min_common_frames:
                continue

            mean_iou = float(np.mean([box_iou(t_boxes[fi], k_boxes[fi]) for fi in common]))
            if mean_iou >= iou_thresh:
                duplicate = True
                break

        if not duplicate:
            kept.append(track)

    return kept


# ── Export helpers ────────────────────────────────────────────────────────────

def _split_frame_ids(
    frame_ids: list[int],
    val_ratio: float = 0.2,
    min_val_frames: int = 1,
) -> tuple[set[int], set[int]]:
    """
    Temporal split to reduce leakage.
    Uses the last chunk as val rather than random frame-level split.
    """
    frame_ids = sorted(frame_ids)
    if not frame_ids:
        return set(), set()

    n_val = max(min_val_frames, int(round(len(frame_ids) * val_ratio)))
    n_val = min(n_val, len(frame_ids))
    val_ids = set(frame_ids[-n_val:])
    train_ids = set(frame_ids[:-n_val]) if n_val < len(frame_ids) else set(frame_ids)
    if not train_ids:
        train_ids = set(frame_ids)
        val_ids = set()
    return train_ids, val_ids


def export_pseudo_labels(
    tracks: list[CandidateTrack],
    frame_dir: str,
    out_dir: str,
    img_h: int,
    img_w: int,
    json_path: Optional[str] = None,
    export_yolo_txt: bool = True,
    remap_for_training: bool = True,
    val_ratio: float = 0.2,
) -> str:
    """
    Export:
      - YOLO-format dataset (.txt labels + image copies) for training
      - JSON predictions with raw/original class IDs

    Returns:
        Path to dataset.yaml
    """
    out_path = Path(out_dir)
    labels_train_dir = out_path / "labels" / "train"
    labels_val_dir = out_path / "labels" / "val"
    images_train_dir = out_path / "images" / "train"
    images_val_dir = out_path / "images" / "val"

    if export_yolo_txt:
        for d in [labels_train_dir, labels_val_dir, images_train_dir, images_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # frame_data: frame_idx -> list[(box_xyxy, score, raw_cls, obj_id)]
    frame_data: dict[int, list[tuple[np.ndarray, float, int, int]]] = {}
    for track in tracks:
        for tf in track.frames:
            frame_data.setdefault(tf.frame_idx, []).append(
                (tf.box, tf.conf if tf.conf > 0 else track.init_conf, track.cls, track.obj_id)
            )

    frame_ids = sorted(frame_data.keys())
    train_ids, val_ids = _split_frame_ids(frame_ids, val_ratio=val_ratio)

    # YOLO .txt labels + image copies
    if export_yolo_txt:
        for fidx in frame_ids:
            entries = frame_data[fidx]
            lbls: list[str] = []
            for box, _, raw_cls, _ in entries:
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                train_cls = TRAIN_ID_MAP[raw_cls] if remap_for_training else raw_cls
                lbls.append(f"{train_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            src = os.path.join(frame_dir, f"{fidx:06d}.jpg")
            if fidx in val_ids:
                lbl_path = labels_val_dir / f"{fidx:06d}.txt"
                img_path = images_val_dir / f"{fidx:06d}.jpg"
            else:
                lbl_path = labels_train_dir / f"{fidx:06d}.txt"
                img_path = images_train_dir / f"{fidx:06d}.jpg"

            if os.path.exists(src):
                shutil.copy2(src, img_path)
            with open(lbl_path, "w") as fh:
                fh.write("\n".join(lbls))

    # dataset.yaml
    yaml_path = out_path / "dataset.yaml"
    if remap_for_training:
        names = [TRAIN_CLASS_NAMES[i] for i in range(len(TRAIN_CLASS_NAMES))]
        nc = len(TRAIN_CLASS_NAMES)
    else:
        # Only valid if your training consumer truly supports sparse IDs.
        sparse_ids = sorted(CLASS_NAMES.keys())
        names = [CLASS_NAMES[i] for i in sparse_ids]
        nc = len(names)

    with open(yaml_path, "w") as fh:
        fh.write(f"path: {out_path.resolve()}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write(f"nc: {nc}\n")
        fh.write(f"names: {names}\n")

    # JSON with original/raw IDs preserved
    all_predictions = []
    for fidx in frame_ids:
        entries = frame_data[fidx]
        boxes_xyxy = [e[0].tolist() for e in entries]
        scores = [float(e[1]) for e in entries]
        classes = [int(e[2]) for e in entries]  # raw/original IDs kept
        class_names = [CLASS_NAMES.get(int(e[2]), f"class_{e[2]}") for e in entries]
        ids = [int(e[3]) for e in entries]
        all_predictions.append(
            {
                "frame_idx": fidx,
                "boxes_xyxy": boxes_xyxy,
                "scores": scores,
                "classes": classes,
                "class_names": class_names,
                "ids": ids,
                "n_masks": len(entries),
            }
        )

    json_out = Path(json_path) if json_path else out_path / "predictions.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(all_predictions, fh)

    n_inst = sum(len(v) for v in frame_data.values())
    print(f"[export] {len(frame_ids)} labelled frames, {n_inst} instances → {out_path}")
    print(f"[export] train frames={len(train_ids)} val frames={len(val_ids)}")
    print(f"[export] JSON (raw IDs preserved) → {json_out}")
    return str(yaml_path)


# ── SAM3 helpers ──────────────────────────────────────────────────────────────

def _best_iou_match(
    candidate_box: np.ndarray,
    active_tracks: dict[int, CandidateTrack],
    frame_idx: int,
    min_iou: float = 0.2,
) -> Optional[int]:
    """
    Associate current SAM result box to a track using last available box IoU.
    Used as a fallback when returned IDs/order are unreliable.
    """
    best_tid = None
    best_iou = min_iou

    for tid, track in active_tracks.items():
        if not track.frames:
            continue
        prev = track.frames[-1]
        # Prefer temporal continuity
        if abs(frame_idx - prev.frame_idx) > 2:
            continue
        iou = box_iou(candidate_box, prev.box)
        if iou > best_iou:
            best_iou = iou
            best_tid = tid

    return best_tid


def _extract_result_boxes(
    r,
    img_h: int,
    img_w: int,
    prefer_mask_box: bool = True,
) -> list[tuple[Optional[int], np.ndarray]]:
    """
    Returns list of (result_track_id_or_none, box_xyxy).
    If masks exist and prefer_mask_box=True, derive tighter boxes from masks.
    """
    r_cpu = r.cpu()
    if r_cpu.boxes is None or len(r_cpu.boxes) == 0:
        return []

    ids = None
    if getattr(r_cpu.boxes, "id", None) is not None:
        try:
            ids = r_cpu.boxes.id.int().tolist()
        except Exception:
            ids = None

    boxes_xyxy = r_cpu.boxes.xyxy.numpy().astype(np.float32)
    out: list[tuple[Optional[int], np.ndarray]] = []

    use_masks = prefer_mask_box and getattr(r_cpu, "masks", None) is not None and getattr(r_cpu.masks, "data", None) is not None
    masks = None
    if use_masks:
        try:
            masks = r_cpu.masks.data.cpu().numpy()
        except Exception:
            masks = None

    for i, box in enumerate(boxes_xyxy):
        final_box = box
        if masks is not None and i < len(masks):
            m = masks[i]
            if m.ndim == 3:
                m = m.squeeze(0)
            tight = mask_to_box(m > 0)
            if tight is not None:
                final_box = tight.astype(np.float32)

        final_box = clip_box(final_box, img_h, img_w)
        rid = ids[i] if ids is not None and i < len(ids) else None
        out.append((rid, final_box))

    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate_pseudo_labels(
    video_path: str,
    yolo_model_path: str = "yolo26x.pt",
    sam3_model: str = "sam3.pt",
    out_dir: str = "pseudo_labels",
    crop_classifier: Optional[Callable] = None,
    confusing_classes: set[int] = frozenset({1}),  # bicycle vs motorcycle, etc.
    # Algorithm defaults
    N: int = 10,
    W: int = 30,
    tau_det: float = 0.60,
    tau_nms: float = 0.50,
    tau_len: int = 3,
    tau_conf: float = 0.50,
    tau_tiou: float = 0.40,
    max_gap: int = 1,
    min_matched_keyframes: int = 1,
    device: str = "cuda",
    batch_size: int = 32,
    sam_assoc_iou: float = 0.20,
    remap_train_labels: bool = True,  # preserves raw IDs in JSON regardless
    val_ratio: float = 0.2,
) -> list[CandidateTrack]:
    """
    Full pseudo-label generation pipeline.

    Important:
      - JSON export preserves raw/original class IDs (e.g. 0,1,2,7).
      - YOLO .txt export remaps to contiguous IDs by default for training safety.

    Returns:
        List of accepted CandidateTrack objects from the final round.
    """
    if not SAM3_AVAILABLE:
        raise ImportError("SAM3 is required. Install/upgrade ultralytics to a version with SAM3VideoPredictor.")

    next_global_obj_id = 0
    frame_dir = os.path.join(out_dir, "_frames")

    # Step 1: extract or reuse frames
    frame_paths = sorted(Path(frame_dir).glob("*.jpg")) if os.path.isdir(frame_dir) else []
    if frame_paths:
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"Failed to read cached frame: {frame_paths[0]}")
        img_h, img_w = first.shape[:2]
        fps = 30.0
        n_frames = len(frame_paths)
        print(f"[1/6] Re-using {n_frames} existing frames.")
    else:
        print("[1/6] Extracting frames …")
        img_h, img_w, fps, n_frames = extract_frames(video_path, frame_dir)
        print(f"      {n_frames} frames  {img_w}×{img_h}  @ {fps:.2f} fps")

    key_frames = list(range(0, n_frames, N))

    # Step 2: YOLO on key frames
    print(f"[2/6] Running YOLO detector on {len(key_frames)} key frames …")
    detector = YOLO(yolo_model_path)
    yolo_dets: dict[int, list[Detection]] = {}

    img_paths = [os.path.join(frame_dir, f"{fi:06d}.jpg") for fi in key_frames]
    n_batches = (len(key_frames) + batch_size - 1) // batch_size

    for b in tqdm(range(0, len(key_frames), batch_size), total=n_batches, desc="YOLO detect", unit="batch"):
        batch_kf = key_frames[b:b + batch_size]
        batch_imgs = img_paths[b:b + batch_size]

        preds = detector.predict(
            batch_imgs,
            conf=tau_det * 0.8,
            classes=TRACK_CLASSES,
            verbose=False,
            device=device,
        )

        for fi, pred in zip(batch_kf, preds):
            raw: list[Detection] = []
            if pred.boxes is not None and len(pred.boxes):
                xyxy = pred.boxes.xyxy.cpu().numpy()
                confs = pred.boxes.conf.cpu().numpy()
                clss = pred.boxes.cls.int().cpu().numpy()
                for box, conf, cls in zip(xyxy, confs, clss):
                    raw.append(
                        Detection(
                            frame_idx=fi,
                            box=box.astype(np.float32),
                            cls=int(cls),
                            conf=float(conf),
                        )
                    )
            yolo_dets[fi] = filter_detections(raw, img_h, img_w, tau_det, tau_nms)

    total_dets = sum(len(v) for v in yolo_dets.values())
    print(f"      {total_dets} detections after filtering.")

    # Step 3: init SAM3
    print("[3/6] Initialising SAM3 (Ultralytics) …")
    sam3 = SAM3VideoPredictor(
        overrides=dict(
            model=sam3_model,
            conf=0.0,
            task="segment",
            mode="predict",
            half=(device != "cpu"),
            save=False,
            verbose=False,
            device=device,
        )
    )

    def _collect_sam3(
        frame_indices: list[int],
        frame_paths_local: list[str],
        active_tracks: dict[int, CandidateTrack],
        init_boxes: list[list[float]],
    ) -> None:
        """
        Run SAM3 on an ordered list of frames with box prompts at frame 0.

        We do not fully trust returned ordering/IDs, so we:
          1) use result IDs if they align to known active tracks
          2) otherwise use IoU-based association fallback
        """
        if not frame_indices or not frame_paths_local or not init_boxes:
            return

        # SAM3VideoPredictor requires dataset.mode == "video".
        # Write frames to a temp video file so ultralytics treats it as video.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_video = os.path.join(tmpdir, "clip.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_video, fourcc, fps, (img_w, img_h))
            for src in frame_paths_local:
                frame = cv2.imread(src)
                writer.write(frame)
            writer.release()
            results = list(sam3(source=tmp_video, bboxes=init_boxes, stream=True))

        for path_idx, r in enumerate(results):
            frame_idx = frame_indices[path_idx]
            extracted = _extract_result_boxes(r, img_h, img_w, prefer_mask_box=True)
            if not extracted:
                continue

            assigned: set[int] = set()

            for rid, box in extracted:
                tid: Optional[int] = None

                # First try direct match if returned ID corresponds to active track
                if rid is not None and rid in active_tracks:
                    tid = rid

                # Fallback: IoU-based association to last known box
                if tid is None:
                    tid = _best_iou_match(box, active_tracks, frame_idx, min_iou=sam_assoc_iou)

                if tid is None or tid in assigned:
                    continue

                track = active_tracks[tid]

                # Deduplicate same frame
                if any(f.frame_idx == frame_idx for f in track.frames):
                    continue

                conf = 0.0
                if frame_idx in yolo_dets:
                    for d in yolo_dets[frame_idx]:
                        if d.cls == track.cls and box_iou(box, d.box) >= 0.3:
                            conf = max(conf, d.conf)

                track.frames.append(TrackFrame(frame_idx=frame_idx, box=box, conf=conf))
                assigned.add(tid)

    # Step 4: propagate
    print(f"[4/6] Propagating tracks (N={N}, W={W}) …")
    candidate_tracks: list[CandidateTrack] = []

    for kf in tqdm(key_frames, desc="SAM3 propagate", unit="kf"):
        kf_dets = yolo_dets.get(kf, [])
        if not kf_dets:
            continue

        # Create globally unique tracks
        active_tracks: dict[int, CandidateTrack] = {}
        init_boxes: list[list[float]] = []

        for det in kf_dets:
            tid = next_global_obj_id
            next_global_obj_id += 1
            active_tracks[tid] = CandidateTrack(
                obj_id=tid,
                cls=det.cls,
                init_conf=det.conf,
                key_frame=kf,
                frames=[TrackFrame(frame_idx=kf, box=det.box.copy(), conf=det.conf)],
            )
            init_boxes.append(det.box.tolist())

        # For the first SAM frame, result order usually corresponds to prompt order.
        # To maximize chance of direct-ID match, rebuild a deterministic local map.
        ordered_track_ids = list(active_tracks.keys())
        remapped_tracks = {i: active_tracks[tid] for i, tid in enumerate(ordered_track_ids)}

        # Forward pass
        fwd_end = min(kf + W, n_frames - 1)
        fwd_indices = list(range(kf, fwd_end + 1))
        fwd_paths = [os.path.join(frame_dir, f"{fi:06d}.jpg") for fi in fwd_indices]
        _collect_sam3(fwd_indices, fwd_paths, remapped_tracks, init_boxes)

        # Backward pass (reverse time order)
        if kf > 0:
            bwd_start = max(0, kf - W)
            bwd_indices = list(range(kf, bwd_start - 1, -1))
            bwd_paths = [os.path.join(frame_dir, f"{fi:06d}.jpg") for fi in bwd_indices]
            _collect_sam3(bwd_indices, bwd_paths, remapped_tracks, init_boxes)

        # Convert back to original global IDs
        for local_id, tr in remapped_tracks.items():
            global_id = ordered_track_ids[local_id]
            tr.obj_id = global_id
            tr.frames.sort(key=lambda f: f.frame_idx)
            # Remove any accidental duplicate frame_idx, keep higher conf
            dedup: dict[int, TrackFrame] = {}
            for tf in tr.frames:
                prev = dedup.get(tf.frame_idx)
                if prev is None or tf.conf > prev.conf:
                    dedup[tf.frame_idx] = tf
            tr.frames = [dedup[i] for i in sorted(dedup)]
            candidate_tracks.append(tr)

    print(f"      {len(candidate_tracks)} candidate tracks collected.")

    # Step 5: validate
    print("[5/6] Validating tracks …")
    accepted_tracks = [
        t for t in candidate_tracks
        if _validate_track(
            t,
            img_h,
            img_w,
            tau_len,
            tau_conf,
            tau_tiou,
            max_gap,
            min_matched_keyframes,
            crop_classifier,
            confusing_classes,
            frame_dir,
        )
    ]
    print(f"      {len(accepted_tracks)} / {len(candidate_tracks)} accepted.")

    # Step 6: deduplicate
    print("[6/6] Merging duplicate tracks …")
    accepted_tracks = merge_duplicate_tracks(accepted_tracks)
    print(f"      {len(accepted_tracks)} tracks after deduplication.")

    # Export
    print("[export] Exporting pseudo-labels …")
    video_stem = Path(video_path).stem
    json_out = os.path.join(out_dir, "sam3", f"{video_stem}.json")
    export_pseudo_labels(
        accepted_tracks,
        frame_dir,
        out_dir,
        img_h,
        img_w,
        json_path=json_out,
        export_yolo_txt=True,
        remap_for_training=remap_train_labels,
        val_ratio=val_ratio,
    )

    return accepted_tracks


# ── Legacy simple tracker ─────────────────────────────────────────────────────

def track_video(
    source: str | int = 0,
    model_path: str = "yolo26x.pt",
    out_path: str = "predictions.json",
    save_video: bool = True,
):
    """
    Simple YOLO tracking with BotSORT.
    Tracks objects in a video, saves results to JSON and an annotated video.
    """
    model = YOLO(model_path)
    base_path = Path(out_path).with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    video_out = str(base_path.with_suffix(".mp4"))

    cap = cv2.VideoCapture(source if isinstance(source, str) else source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        classes=TRACK_CLASSES,
        verbose=False,
        save=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    all_predictions = []
    writer = None
    for i, r in enumerate(tqdm(results, desc="track", unit="f")):
        boxes = r.boxes
        frame_pred = {"frame_idx": i}
        if boxes is not None and len(boxes) > 0:
            frame_pred["boxes_xyxy"] = boxes.xyxy.cpu().tolist()
            frame_pred["scores"] = boxes.conf.cpu().tolist()
            frame_pred["classes"] = boxes.cls.int().cpu().tolist()
            frame_pred["ids"] = boxes.id.int().cpu().tolist() if boxes.id is not None else []
        else:
            frame_pred.update(boxes_xyxy=[], scores=[], classes=[], ids=[])

        frame_pred["n_masks"] = len(r.masks.data) if r.masks is not None else 0
        all_predictions.append(frame_pred)

        if save_video:
            frame = r.plot()
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    video_out,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
            writer.write(frame)

    if writer is not None:
        writer.release()

    pred_path = str(base_path.with_suffix(".json"))
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"Saved {len(all_predictions)} frames to {pred_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",     choices=["track", "pseudo"], default="pseudo")
    ap.add_argument("--source",   default="/home/brianko/Visual-Preference/test2/2_09_084511_3min.mp4")
    ap.add_argument("--model",    default="yolo26x.pt")
    ap.add_argument("--out-path", default="yolo/predictions.json")
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()

    if args.mode == "track":
        track_video(
            source=args.source,
            model_path=args.model,
            out_path=args.out_path,
            save_video=args.save_video,
        )
    else:
        generate_pseudo_labels(
            video_path="/home/brianko/Visual-Preference/test2/2_09_084511_3min.mp4",
            yolo_model_path="yolo26x.pt",
            sam3_model="sam3.pt",
            out_dir="pseudo_labels_/2_09_084511",
            N=10,
            W=30,
            tau_det=0.60,
            tau_nms=0.50,
            tau_len=3,
            tau_conf=0.50,
            tau_tiou=0.40,
            max_gap=1,
            min_matched_keyframes=1,
            device="cuda",
            remap_train_labels=True,  # JSON keeps raw IDs; YOLO txt gets contiguous IDs
            val_ratio=0.2,
        )