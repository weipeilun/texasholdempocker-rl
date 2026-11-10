"""Measure how each bounded SO-ARM viewpoint changes the table image."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .face_track import TICKS_PER_DEG, load_config, open_camera, rotate_frame
from .table_scan import AXES, LIFT, TableMotion, connect


def measure(cv2, reference, current):
    """Return where the new image is centered in the reference and its apparent scale."""
    import numpy as np

    height, width = reference.shape[:2]
    size = (960, round(960 * height / width))
    detector = cv2.ORB_create(2200)

    def features(frame):
        gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
        return detector.detectAndCompute(gray, None)

    before, a = features(reference)
    after, b = features(current)
    if a is None or b is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a, b, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.73 * pair[1].distance]
    if len(good) < 15:
        return None
    source = np.float32([before[m.queryIdx].pt for m in good])
    dest = np.float32([after[m.trainIdx].pt for m in good])
    affine, mask = cv2.estimateAffinePartial2D(source, dest, ransacReprojThreshold=3.0)
    if affine is None or int(mask.sum()) < 12:
        return None
    center = cv2.invertAffineTransform(affine) @ np.array([size[0] / 2, size[1] / 2, 1])
    return {
        "center": [round(float(center[i] / size[i]), 3) for i in range(2)],
        "scale": round(float(np.linalg.norm(affine[:, 0])), 3),
        "matches": int(mask.sum()),
    }


def calibrate(arm, cap, cv2, rotate, *, close_views=False):
    home = arm.read_all()
    candidates = []
    axes = [(AXES[1], (7, 14)), (AXES[0], (7, 14))]
    if close_views and LIFT in home:
        axes.append((LIFT, (4,)))
    for sid, steps in axes:
        low, high = arm.read_limits(sid)
        for sign in (-1, 1):
            for degrees in steps:
                goal = home[sid] + sign * round(degrees * TICKS_PER_DEG)
                if low + 20 <= goal <= high - 20:
                    candidates.append((sid, goal))
    if not candidates:
        raise ValueError("当前姿态没有安全试探余量")

    def frame():
        # Discard frames buffered during movement.
        for _ in range(4):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError("校准时相机取帧失败")
        return rotate_frame(cv2, image, rotate)

    motion = TableMotion(arm, home, candidates)

    def move(view):
        motion.go(view)
        deadline = time.monotonic() + 10
        while motion.step():
            if time.monotonic() > deadline:
                raise RuntimeError("校准时关节到位超时")
            time.sleep(0.08)
        time.sleep(0.45)

    try:
        reference = frame()
        views, effects = [], []
        for sid, target in candidates:
            move((sid, target))
            effect = measure(cv2, reference, frame())
            degrees = round((target - home[sid]) / TICKS_PER_DEG, 1)
            print(f"关节 {sid} {degrees:+.1f}°，读回 {arm.read_pos(sid)}，视野变化 {effect}", flush=True)
            if effect:
                views.append(home | {sid: target})
                effects.append(effect)
        move((AXES[0], home[AXES[0]]))
        if not views:
            raise RuntimeError("无法从画面中稳定测出各动作的视野变化")
        return {"home": home, "views": views, "effects": effects}
    finally:
        motion.stop()


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", required=True)
    parser.add_argument("--out", default="~/.splendor_offline/arm_poses.json")
    parser.add_argument("--close-views", action="store_true", help="额外测量肩部升降 ±4°")
    args = parser.parse_args()
    cfg = load_config()
    cap = open_camera(cv2, args.camera, 2592, 1944)
    if cap is None:
        parser.error("无法打开牌桌相机")
    try:
        arm = connect(args.port, cfg["baudrate"])
        try:
            profile = calibrate(arm, cap, cv2, cfg["rotate"], close_views=args.close_views)
        finally:
            arm.bus.close()
    finally:
        cap.release()
    path = Path(args.out).expanduser()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(f"已记录 {len(profile['views'])} 个实测取景动作至 {path}", flush=True)


if __name__ == "__main__":
    main()
