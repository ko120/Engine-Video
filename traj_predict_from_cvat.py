"""
Trajectory prediction evaluated against CVAT ground-truth annotations.

Reads a CVAT annotations.xml, then for every track and every eligible frame:
  1. Linear predictor   – constant-velocity model over the last `--lookback`
                          frames, forecasting the next `--horizon` frames.
  2. Kalman predictor   – constant-velocity Kalman filter over the same history,
                          forecasting the next `--horizon` frames.
  3. Ground-truth       – the actual future positions recorded in the XML.

Outputs
-------
  • A new CVAT XML with four track layers per object:
      bbox track               – original ground-truth bounding boxes
      polyline "lin_pred_*"    – linear-predicted future path
      polyline "kalman_pred_*" – Kalman-predicted future path
      polyline "gt_*"          – ground-truth future path

  • Console metrics:
      per-track ADE / FDE for linear and Kalman
      dataset-wide averages for linear and Kalman

Usage
-----
    python traj_predict_from_cvat.py \
        --input  data/easy/annotations.xml \
        --output predictions.xml \
        --horizon 30 --lookback 10 --poly-stride 5
"""

import argparse
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
from datetime import datetime, timezone
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from filterpy.kalman import KalmanFilter


# ── XML parsing ────────────────────────────────────────────────────────────────

def parse_cvat_xml(path):
    """
    Parse a CVAT interpolation-mode XML.

    Returns
    -------
    meta : dict   – width, height, total_frames, task_name, label_colors
    tracks : list of dicts, each with keys:
        id, label, frames (sorted list of (frame_idx, xtl, ytl, xbr, ybr))
    """
    tree = ET.parse(path)
    root = tree.getroot()

    orig = root.find(".//original_size")
    width = int(orig.find("width").text)
    height = int(orig.find("height").text)

    size_el = root.find(".//size")
    task_name_el = root.find(".//name")
    total_frames = int(size_el.text) if size_el is not None else None
    task_name = task_name_el.text if task_name_el is not None else "annotations"

    label_colors = {}
    for lbl in root.findall(".//label"):
        name = lbl.find("name").text
        color = lbl.find("color")
        label_colors[name] = color.text if color is not None else "#ffffff"

    tracks = []
    for track_el in root.findall("track"):
        tid = int(track_el.get("id"))
        label = track_el.get("label")

        frames = []
        for box in track_el.findall("box"):
            if box.get("outside", "0") == "1":
                continue
            frames.append((
                int(box.get("frame")),
                float(box.get("xtl")),
                float(box.get("ytl")),
                float(box.get("xbr")),
                float(box.get("ybr")),
            ))

        frames.sort(key=lambda x: x[0])
        if frames:
            tracks.append({"id": tid, "label": label, "frames": frames})

    if total_frames is None:
        total_frames = max(f[0] for t in tracks for f in t["frames"]) + 1

    meta = {
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "task_name": task_name,
        "label_colors": label_colors,
    }
    return meta, tracks


# ── geometry helpers ──────────────────────────────────────────────────────────

def box_center(xtl, ytl, xbr, ybr):
    return ((xtl + xbr) / 2.0, (ytl + ybr) / 2.0)


def frame_to_center(track):
    """Returns {frame_idx: (cx, cy)} for quick look-up."""
    return {f[0]: box_center(f[1], f[2], f[3], f[4]) for f in track["frames"]}


# ── predictors ────────────────────────────────────────────────────────────────

def linear_predict(history, horizon):
    """
    Constant-velocity prediction using average velocity over history.

    Parameters
    ----------
    history : list of (x, y) – observed positions (oldest first)
    horizon : int – number of future steps to predict

    Returns
    -------
    list of (x, y) of length `horizon`
    """
    if len(history) >= 2:
        x0, y0 = history[0]
        x1, y1 = history[-1]
        dt = len(history) - 1
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
    else:
        x1, y1 = history[-1]
        vx = vy = 0.0

    return [(x1 + k * vx, y1 + k * vy) for k in range(1, horizon + 1)]


def init_kalman_filter(history, process_var=1.0, meas_var=4.0):
    """
    Create a 2D constant-velocity Kalman filter.

    State:
        [x, y, vx, vy]^T

    Measurement:
        [x, y]^T
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State transition (dt = 1 frame)
    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    # Measurement function
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    # Initialize state from history
    x_last, y_last = history[-1]

    if len(history) >= 2:
        x_prev, y_prev = history[-2]
        vx0 = x_last - x_prev
        vy0 = y_last - y_prev
    else:
        vx0 = 0.0
        vy0 = 0.0

    kf.x = np.array([x_last, y_last, vx0, vy0], dtype=float)

    # Initial covariance: more uncertain in velocity than position
    kf.P = np.array([
        [10, 0,   0,   0],
        [0,  10,  0,   0],
        [0,  0, 100,   0],
        [0,  0,   0, 100],
    ], dtype=float)

    # Measurement noise
    kf.R = np.array([
        [meas_var, 0],
        [0, meas_var],
    ], dtype=float)

    # Process noise
    q = process_var
    kf.Q = np.array([
        [0.25*q, 0,      0.5*q, 0],
        [0,      0.25*q, 0,     0.5*q],
        [0.5*q,  0,      1.0*q, 0],
        [0,      0.5*q,  0,     1.0*q],
    ], dtype=float)

    return kf


def kalman_predict(history, horizon, process_var=1.0, meas_var=4.0):
    """
    Kalman-filter prediction using the same history window.

    Parameters
    ----------
    history : list of (x, y) – observed positions (oldest first)
    horizon : int – number of future steps to predict

    Returns
    -------
    list of (x, y) of length `horizon`
    """
    if len(history) == 0:
        return []
    if len(history) == 1:
        x, y = history[-1]
        return [(x, y) for _ in range(horizon)]

    kf = init_kalman_filter(history, process_var=process_var, meas_var=meas_var)

    # Re-run filter through history in temporal order
    # We initialized at last point for a reasonable state guess, but for a clean
    # history fit we reset state using first observation.
    x0, y0 = history[0]
    if len(history) >= 2:
        x1, y1 = history[1]
        vx0 = x1 - x0
        vy0 = y1 - y0
    else:
        vx0 = vy0 = 0.0

    kf.x = np.array([x0, y0, vx0, vy0], dtype=float)

    for i, (x, y) in enumerate(history):
        if i > 0:
            kf.predict()
        kf.update(np.array([x, y], dtype=float))

    preds = []
    for _ in range(horizon):
        kf.predict()
        preds.append((float(kf.x[0]), float(kf.x[1])))

    return preds


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(tracks, horizon, lookback, kalman_process_var=1.0, kalman_meas_var=4.0):
    """
    For every frame t in every track that has `horizon` future GT frames:
      - build a linear prediction from the last `lookback` observed centres
      - build a Kalman prediction from the same observed centres
      - collect the ground-truth future centres
      - compute per-instance ADE and FDE

    Returns
    -------
    per_track : dict
    overall   : dict
    """
    per_track = {}

    all_lin_ade, all_lin_fde = [], []
    all_kal_ade, all_kal_fde = [], []

    for track in tracks:
        tid = track["id"]
        fc = frame_to_center(track)
        fidxs = sorted(fc.keys())

        lin_ades, lin_fdes = [], []
        kal_ades, kal_fdes = [], []

        for i, t in enumerate(fidxs):
            future_fidxs = [f for f in fidxs if f > t][:horizon]
            if len(future_fidxs) < horizon:
                continue

            past_fidxs = fidxs[max(0, i - lookback + 1): i + 1]
            history = [fc[f] for f in past_fidxs]

            lin_pred = linear_predict(history, horizon)
            kal_pred = kalman_predict(
                history,
                horizon,
                process_var=kalman_process_var,
                meas_var=kalman_meas_var,
            )
            gt = [fc[f] for f in future_fidxs]

            lin_dists = [np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(lin_pred, gt)]
            kal_dists = [np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(kal_pred, gt)]

            lin_ades.append(float(np.mean(lin_dists)))
            lin_fdes.append(float(lin_dists[-1]))

            kal_ades.append(float(np.mean(kal_dists)))
            kal_fdes.append(float(kal_dists[-1]))

        if lin_ades:
            per_track[tid] = {
                "label": track["label"],
                "n": len(lin_ades),
                "lin_ade": float(np.mean(lin_ades)),
                "lin_fde": float(np.mean(lin_fdes)),
                "kal_ade": float(np.mean(kal_ades)),
                "kal_fde": float(np.mean(kal_fdes)),
            }

            all_lin_ade.extend(lin_ades)
            all_lin_fde.extend(lin_fdes)
            all_kal_ade.extend(kal_ades)
            all_kal_fde.extend(kal_fdes)

    overall = {
        "lin_ade": float(np.mean(all_lin_ade)) if all_lin_ade else float("nan"),
        "lin_fde": float(np.mean(all_lin_fde)) if all_lin_fde else float("nan"),
        "kal_ade": float(np.mean(all_kal_ade)) if all_kal_ade else float("nan"),
        "kal_fde": float(np.mean(all_kal_fde)) if all_kal_fde else float("nan"),
    }
    return per_track, overall


# ── XML output ────────────────────────────────────────────────────────────────

def build_output_xml(meta, tracks, horizon, lookback, poly_stride,
                     kalman_process_var=1.0, kalman_meas_var=4.0,
                     box_threshold=0.5):
    """
    Produces a CVAT-compatible XML with four track layers per object:
      • bbox track
      • lin_pred_<label>
      • kalman_pred_<label>
      • gt_<label>
    """
    now = datetime.now(timezone.utc).isoformat()

    orig_labels = sorted({t["label"] for t in tracks})
    lc = meta["label_colors"]

    root = Element("annotations")
    SubElement(root, "version").text = "1.1"

    meta_el = SubElement(root, "meta")
    job_el = SubElement(meta_el, "job")
    SubElement(job_el, "id").text = "1"
    SubElement(job_el, "name").text = meta["task_name"] + "_traj"
    SubElement(job_el, "size").text = str(meta["total_frames"])
    SubElement(job_el, "mode").text = "interpolation"
    SubElement(job_el, "overlap").text = "0"
    SubElement(job_el, "bugtracker")
    SubElement(job_el, "created").text = now
    SubElement(job_el, "updated").text = now
    SubElement(job_el, "start_frame").text = "0"
    SubElement(job_el, "stop_frame").text = str(meta["total_frames"] - 1)
    SubElement(job_el, "frame_filter")

    labels_el = SubElement(job_el, "labels")

    for lname in orig_labels:
        lbl = SubElement(labels_el, "label")
        SubElement(lbl, "name").text = lname
        SubElement(lbl, "color").text = lc.get(lname, "#ffffff")
        SubElement(lbl, "type").text = "any"
        SubElement(lbl, "attributes")

    for lname in orig_labels:
        lbl = SubElement(labels_el, "label")
        SubElement(lbl, "name").text = f"lin_pred_{lname}"
        SubElement(lbl, "color").text = "#ff4444"   # red
        SubElement(lbl, "type").text = "any"
        SubElement(lbl, "attributes")

    for lname in orig_labels:
        lbl = SubElement(labels_el, "label")
        SubElement(lbl, "name").text = f"kalman_pred_{lname}"
        SubElement(lbl, "color").text = "#4444ff"   # blue
        SubElement(lbl, "type").text = "any"
        SubElement(lbl, "attributes")

    for lname in orig_labels:
        lbl = SubElement(labels_el, "label")
        SubElement(lbl, "name").text = f"gt_{lname}"
        SubElement(lbl, "color").text = "#44ff44"   # green
        SubElement(lbl, "type").text = "any"
        SubElement(lbl, "attributes")

    segs = SubElement(job_el, "segments")
    seg = SubElement(segs, "segment")
    SubElement(seg, "id").text = "1"
    SubElement(seg, "start").text = "0"
    SubElement(seg, "stop").text = str(meta["total_frames"] - 1)
    SubElement(seg, "url")

    orig_el = SubElement(job_el, "original_size")
    SubElement(orig_el, "width").text = str(meta["width"])
    SubElement(orig_el, "height").text = str(meta["height"])
    SubElement(meta_el, "dumped").text = now

    cvat_id = 0

    for track in tracks:
        label = track["label"]
        fc = frame_to_center(track)
        fidxs = sorted(fc.keys())
        last_f = fidxs[-1]

        # (1) original bbox track
        bbox_el = SubElement(root, "track",
                             id=str(cvat_id), label=label,
                             source="file", z_order="0")
        cvat_id += 1

        last_box = None
        for (f, xtl, ytl, xbr, ybr) in track["frames"]:
            box = (xtl, ytl, xbr, ybr)
            changed = (last_box is None or
                       any(abs(box[i] - last_box[i]) > box_threshold for i in range(4)))
            if changed:
                SubElement(
                    bbox_el, "box",
                    frame=str(f), outside="0", occluded="0",
                    keyframe="1",
                    xtl=f"{xtl:.2f}", ytl=f"{ytl:.2f}",
                    xbr=f"{xbr:.2f}", ybr=f"{ybr:.2f}",
                    z_order="0"
                )
                last_box = box

        last_row = track["frames"][-1]
        if last_f + 1 < meta["total_frames"]:
            SubElement(
                bbox_el, "box",
                frame=str(last_f + 1), outside="1", occluded="0",
                keyframe="1",
                xtl=f"{last_row[1]:.2f}", ytl=f"{last_row[2]:.2f}",
                xbr=f"{last_row[3]:.2f}", ybr=f"{last_row[4]:.2f}",
                z_order="0"
            )

        def write_poly_track(elem_label, pts_fn, z_order):
            nonlocal cvat_id
            el = SubElement(
                root, "track",
                id=str(cvat_id), label=elem_label,
                source="auto", z_order=str(z_order)
            )
            cvat_id += 1
            last_pts_str = ""

            for i, t in enumerate(fidxs):
                on_stride = (i % poly_stride == 0)
                is_last = (i == len(fidxs) - 1)
                if not (on_stride or is_last):
                    continue

                pts = pts_fn(i, t)
                if pts:
                    cx, cy = fc[t]
                    all_pts = [(cx, cy)] + pts
                    last_pts_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in all_pts)
                    SubElement(
                        el, "polyline",
                        frame=str(t), outside="0", occluded="0",
                        keyframe="1", points=last_pts_str,
                        z_order=str(z_order)
                    )

            if last_f + 1 < meta["total_frames"] and last_pts_str:
                SubElement(
                    el, "polyline",
                    frame=str(last_f + 1), outside="1", occluded="0",
                    keyframe="1", points=last_pts_str,
                    z_order=str(z_order)
                )

        # (2) linear prediction
        def lin_pred_pts(i, t):
            past = fidxs[max(0, i - lookback + 1): i + 1]
            history = [fc[f] for f in past]
            return linear_predict(history, horizon)

        write_poly_track(f"lin_pred_{label}", lin_pred_pts, z_order=1)

        # (3) kalman prediction
        def kalman_pred_pts(i, t):
            past = fidxs[max(0, i - lookback + 1): i + 1]
            history = [fc[f] for f in past]
            return kalman_predict(
                history,
                horizon,
                process_var=kalman_process_var,
                meas_var=kalman_meas_var,
            )

        write_poly_track(f"kalman_pred_{label}", kalman_pred_pts, z_order=2)

        # (4) ground-truth future
        def gt_pts(i, t):
            future = [f for f in fidxs if f > t][:horizon]
            return [fc[f] for f in future]

        write_poly_track(f"gt_{label}", gt_pts, z_order=3)

    return root


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Trajectory prediction + GT comparison from CVAT XML."
    )
    ap.add_argument("--input", default="/content/engineVideo/label/annotations_easy.xml",
                    help="Input CVAT annotations XML")
    ap.add_argument("--output", default=None,
                    help="Output CVAT XML (default: <input_dir>/predictions.xml)")
    ap.add_argument("--horizon", type=int, default=30,
                    help="Frames ahead to predict (default: 30)")
    ap.add_argument("--lookback", type=int, default=10,
                    help="Observation frames for velocity estimate/filtering (default: 10)")
    ap.add_argument("--poly-stride", type=int, default=5,
                    help="Write polyline keyframe every N frames (default: 5)")
    ap.add_argument("--box-threshold", type=float, default=0.5,
                    help="Min pixel change for a new bbox keyframe (default: 0.5)")
    ap.add_argument("--kalman-process-var", type=float, default=1.0,
                    help="Kalman process noise variance (default: 1.0)")
    ap.add_argument("--kalman-meas-var", type=float, default=4.0,
                    help="Kalman measurement noise variance (default: 4.0)")
    ap.add_argument("--no-xml", action="store_true",
                    help="Skip XML output, only print metrics")
    args = ap.parse_args()

    if args.output is None:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(input_dir, "predictions.xml")

    print(f"Parsing {args.input} ...")
    meta, tracks = parse_cvat_xml(args.input)
    print(f"  {len(tracks)} tracks  |  {meta['total_frames']} frames  |  {meta['width']}x{meta['height']} px")

    print(
        f"\nComputing metrics "
        f"(lookback={args.lookback}, horizon={args.horizon}, "
        f"kalman_process_var={args.kalman_process_var}, "
        f"kalman_meas_var={args.kalman_meas_var}) ..."
    )
    per_track, overall = compute_metrics(
        tracks,
        args.horizon,
        args.lookback,
        kalman_process_var=args.kalman_process_var,
        kalman_meas_var=args.kalman_meas_var,
    )

    print(
        f"\n{'Track':>6}  {'Label':<12}  {'Instances':>9}  "
        f"{'Lin ADE':>10}  {'Lin FDE':>10}  {'Kal ADE':>10}  {'Kal FDE':>10}"
    )
    print("-" * 85)
    for tid in sorted(per_track):
        m = per_track[tid]
        print(
            f"  {tid:>4}  {m['label']:<12}  {m['n']:>9}  "
            f"{m['lin_ade']:>10.3f}  {m['lin_fde']:>10.3f}  "
            f"{m['kal_ade']:>10.3f}  {m['kal_fde']:>10.3f}"
        )
    print("-" * 85)
    print(
        f"{'Overall':<22}  {'':>9}  "
        f"{overall['lin_ade']:>10.3f}  {overall['lin_fde']:>10.3f}  "
        f"{overall['kal_ade']:>10.3f}  {overall['kal_fde']:>10.3f}"
    )

    if not args.no_xml:
        print(f"\nBuilding output XML ...")
        xml_root = build_output_xml(
            meta, tracks,
            args.horizon, args.lookback, args.poly_stride,
            kalman_process_var=args.kalman_process_var,
            kalman_meas_var=args.kalman_meas_var,
            box_threshold=args.box_threshold,
        )

        raw = tostring(xml_root, encoding="unicode")
        dom = xml.dom.minidom.parseString(raw)
        pretty = ('<?xml version="1.0" encoding="utf-8"?>\n'
                  + dom.toprettyxml(indent="  ").split("\n", 1)[1])

        with open(args.output, "w", encoding="utf-8") as f:
            f.write(pretty)

        out_tree = ET.parse(args.output)
        n_bbox = sum(1 for e in out_tree.getroot().iter("box") if e.get("outside") == "0")
        n_poly = sum(1 for e in out_tree.getroot().iter("polyline") if e.get("outside") == "0")
        print(f"  Written -> {args.output}")
        print(f"  Bbox keyframes:     {n_bbox}")
        print(f"  Polyline keyframes: {n_poly}  (lin_pred + kalman_pred + gt combined)")


if __name__ == "__main__":
    main()