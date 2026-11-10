"""Own the Mac camera and the calibrated table scan. Only the worker commands the arm."""
from __future__ import annotations

import threading
from pathlib import Path
import time

from so_arm import table_scan
from so_arm.face_track import JOINTS, TICKS_PER_DEG, open_camera, rotate_frame

from .table import UNKNOWN, validate

REGIONS = {
    "market": "公共牌与贵族",
    **{f"seat{i}": f"玩家{i}" for i in range(1, 5)},
}
SETTLE_SECONDS = 0.55
MOVE_TIMEOUT = 12



def assess_table(data, table, current_player=None):
    raw = data.get("coverage")
    raw = raw if isinstance(raw, dict) else {}
    coverage = {region: raw.get(region) for region in REGIONS}
    coverage = {region: value if value in ("clear", "cropped", "blocked", "unreadable") else "unknown"
                for region, value in coverage.items()}
    table.issues.extend(validate(table, current_player=current_player))
    complete = (
        data.get("visible") is True and not table.issues
        and all(value == "clear" for value in coverage.values())
        and table.bank is not None and table.nobles is not None and table.market is not None
        and all(UNKNOWN not in row for row in table.market)
        and all(value is not None for seat in table.seats for value in vars(seat).values())
    )
    if not complete:
        table.visible = False
        table.issues.extend(f"{REGIONS[k]}：{label}" for k, v in coverage.items()
                            if (label := {"cropped": "出画", "blocked": "被遮挡",
                                         "unreadable": "看不清", "unknown": "无法确认"}.get(v)))
        if not table.issues:
            table.issues.append("有牌局字段未能完整识别")
    return coverage, complete



def view_score(coverage, table):
    cards = sum(card != UNKNOWN for row in (table.market or []) for card in row)
    fields = sum(value is not None for seat in table.seats for value in vars(seat).values())
    return sum(value == "clear" for value in coverage.values()), cards + fields


class ArmCamera:
    def __init__(self, camera, port, baudrate, rotate, poses):
        self.camera, self.port, self.baudrate = camera, port, baudrate
        self.rotate, self.poses = rotate, poses
        self.lock = threading.RLock()
        self.worker = None
        self.running = self.auto = False
        self.operator = None
        self.message = "机械臂相机已关闭"
        self.error = None
        self.jpeg = self.frame_time = None
        self.seq = self.revision = self.capture_epoch = 0
        self.ready_at = 0.
        self.coverage = {}
        self.motion = None
        self.targets = []
        self.searching = self.retry = False
        self.bad_frames = self.good_frames = 0
        self.deadline = 0.
        self.trial = None
        self.best = None
        self.returning = False
        self.effects = {}
        self.next_view = None

    def start(self, operator):
        with self.lock:
            if self.worker and self.worker.is_alive():
                if not self.running:
                    raise ValueError("相机正在关闭，请稍后重试")
                return
            self.running = True
            self.operator = operator
            self.error = None
            self.message = "正在打开机械臂相机…"
            self.revision += 1
            self.capture_epoch += 1
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()

    def close(self):
        with self.lock:
            self.running = False
            self.auto = False
            self.revision += 1
            self.capture_epoch += 1
            self.jpeg = self.frame_time = None
            self.message = "机械臂相机已关闭"

    def set_auto(self, enabled):
        with self.lock:
            if enabled:
                if not self.running or self.jpeg is None or time.time() - self.frame_time > 2:
                    raise ValueError("请先打开并确认机械臂相机画面")
                if self.motion and self.motion.moving:
                    raise ValueError("正在停止或调整机械臂，请稍后重试")
            self.auto = enabled
            self.searching = self.retry = self.returning = False
            self.best = self.next_view = None
            self.capture_epoch += 1
            self.bad_frames = self.good_frames = 0
            self.coverage = {}
            self.error = None
            self.revision += 1
            self.message = "正在检查牌桌画面…" if enabled else "自动取景已停止，保持当前姿势"

    def preview(self):
        with self.lock:
            return self.jpeg, self.frame_time, self.seq, f"机械臂相机 {self.camera}"

    def view(self):
        with self.lock:
            return {"running": self.running, "auto": self.auto, "message": self.message,
                    "error": self.error, "coverage": {REGIONS[k]: v for k, v in self.coverage.items()}}

    def sample(self):
        with self.lock:
            if (not self.running or self.jpeg is None or time.time() - self.frame_time > 2
                    or (self.auto and self.motion is None)
                    or (self.motion and self.motion.moving) or self.retry or self.frame_time < self.ready_at):
                return None
            return self.preview(), self.revision

    def _actions(self):
        if not self.auto or self.motion is None:
            return {}
        remaining = self.targets if self.searching and self.targets is not None else self.motion.targets
        return {f"view_{i + 1}": target for i, target in enumerate(self.motion.targets)
                if target in remaining and target != self.trial}

    def plan(self, missing):
        with self.lock:
            if not self.auto or self.motion is None:
                return None
            actions = self._actions()
            names = {sid: name for name, sid in JOINTS.items()}
            profile = [
                {"id": key, "joint": names[target[0]],
                 "delta_degrees": round((target[1] - self.motion.home[target[0]]) / TICKS_PER_DEG, 1),
                 **self.effects.get(target, {})}
                for key, target in actions.items()
            ]
            current = next((f"view_{i + 1}" for i, target in enumerate(self.motion.targets)
                            if target == self.trial), "between_views" if self.trial is None else "home")
            return {"current_view": current,
                    "current_effect": self.effects.get(self.trial) if current != "home" else
                    {"center": [0.5, 0.5], "scale": 1},
                    "missing": missing, "available_actions": profile}

    def accept(self, revision, coverage, complete, drive, score=None, suggestion=None):
        """Accept only results for the current stationary pose; count separate frames."""
        with self.lock:
            if not self.running or self.revision != revision or self.sample() is None:
                return False
            self.coverage = coverage
            if not self.auto:
                return True
            if not drive:
                self.bad_frames = self.good_frames = 0
                self.message = "识别结果已过时，等待当前画面的复核"
                return True
            score = score or (sum(value == "clear" for value in coverage.values()), 0)
            if self.best is None or score > self.best[0]:
                self.best = (score, self.trial)
            self.bad_frames = self.bad_frames + 1 if not complete else 0
            self.good_frames = self.good_frames + 1 if complete else 0
            proposed = self._actions().get(suggestion) if isinstance(suggestion, str) else None
            if self.good_frames >= 2:
                self.searching = False
                self.targets.clear()
                self.best = None
                self.message = "已看全牌桌，保持当前角度"
            elif not complete and (proposed is not None or self.bad_frames >= (1 if self.searching else 2)):
                if not self.searching:
                    self.searching = True
                    self.targets = None  # initialize the sweep from the taught list in the worker
                self.retry = True
                self.next_view = proposed
                self.revision += 1
                self.bad_frames = self.good_frames = 0
                self.message = "画面不全，正在调整取景…"
            elif not complete:
                self.message = "有区域尚未读清，复核后会尝试下一角度…"
            return True

    def _stop_motion(self):
        if self.motion and self.motion.moving:
            self.motion.stop()
            self.trial = None
            self.ready_at = time.time() + SETTLE_SECONDS
            self.revision += 1

    def _update_motion(self):
        if not self.auto:
            self._stop_motion()
            return
        if self.motion is None:
            arm = table_scan.connect(self.port, self.baudrate)
            try:
                if self.poses and Path(self.poses).expanduser().exists():
                    home, views, self.effects = table_scan.load_profile(self.poses)
                else:
                    home, views = table_scan.auto_poses(arm)
                    self.effects = {}
                self.motion = table_scan.TableMotion(arm, home, views)
                self.trial = self.motion.current
            except BaseException:
                arm.bus.close()
                raise
            self.ready_at = time.time() + SETTLE_SECONDS
            self.revision += 1
        if self.retry:
            if self.targets is None:
                self.targets = [target for target in self.motion.targets if target != self.trial]
            if self.targets:
                if self.next_view in self.targets:
                    self.trial = self.next_view
                    self.targets.remove(self.trial)
                    self.message = "正在前往 Terra 选定的取景位置…"
                else:
                    self.trial = self.targets.pop(0)
                self.next_view = None
            else:
                home_view = (table_scan.AXES[0], self.motion.home[table_scan.AXES[0]])
                best_view = (self.best[1] if self.best else None) or self.trial or home_view
                self.returning = True
                self.trial = best_view
                self.message = "未找到完整画面，正在回到识别最清楚的角度…"
            self.motion.go(self.trial)
            self.deadline = time.monotonic() + MOVE_TIMEOUT
            self.retry = False
        if self.motion and self.motion.moving:
            if time.monotonic() > self.deadline:
                raise ValueError("关节未在预期时间内到位")
            if not self.motion.step():
                self.ready_at = time.time() + SETTLE_SECONDS
                if self.returning:
                    self.auto = self.searching = self.returning = False
                    self.message = "已保留可读信息最多的角度，仍有区域读不清。请查看下方区域提示。"
                else:
                    self.message = "已到达候选角度，等待画面稳定…"

    def _run(self):
        cap = None
        try:
            import cv2

            cap = open_camera(cv2, self.camera, 2592, 1944)
            if cap is None:
                raise ValueError(f"打不开相机 {self.camera}，请确认 USB 连接、相机编号和 macOS 相机权限")
            while True:
                with self.lock:
                    if not self.running:
                        break
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("机械臂相机读帧失败，请检查连接后重新打开")
                frame = rotate_frame(cv2, frame, self.rotate)
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise ValueError("无法编码机械臂相机画面")
                with self.lock:
                    if not self.running:
                        break
                    if self.jpeg is None:
                        self.message = "相机已打开，请确认这是机械臂上的画面"
                    self.jpeg, self.frame_time = jpg.tobytes(), time.time()
                    self.seq += 1
                    try:
                        self._update_motion()
                    except Exception as exc:
                        self.auto = self.retry = self.searching = False
                        self.revision += 1
                        self.error = f"机械臂停止取景：{exc}"
                        self.message = "自动取景已停止"
                        try:
                            self._stop_motion()
                        finally:
                            if self.motion:
                                self.motion.arm.bus.close()
                                self.motion = None
                time.sleep(max(0., 0.08 - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.close()
                self.error = str(exc)
        finally:
            with self.lock:
                if self.motion:
                    try:
                        self._stop_motion()
                    except Exception as exc:
                        self.error = f"无法确认关节已停止：{exc}"
                    finally:
                        self.motion.arm.bus.close()
                        self.motion = None
            if cap is not None:
                cap.release()
