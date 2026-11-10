"""Teach safe camera viewpoints: python -m so_arm.table_scan --port /dev/cu.usbmodem…"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .face_track import Arm, JOINTS, TICKS_PER_DEG, TICKS_PER_REV, load_config, open_bus

AXES = (JOINTS["shoulder_pan"], JOINTS["wrist_flex"])
LIFT = JOINTS["shoulder_lift"]
SCAN_AXES = (*AXES, LIFT)
POSE_TOLERANCE = round(1.5 * TICKS_PER_DEG)
MAX_SWING = round(15 * TICKS_PER_DEG)
STEP = round(1.2 * TICKS_PER_DEG)
ARRIVAL_TOLERANCE = 5  # about 0.44°, including the servo deadband
MAX_VIEWS = 12


def connect(port, baudrate):
    try:
        bus = open_bus(port, baudrate)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    try:
        bus.ser.write_timeout = 0.1
        ids = [sid for sid in JOINTS.values() if bus.ping(sid)]
        if not all(sid in ids for sid in AXES):
            raise ValueError("肩部旋转或腕部俯仰舵机没有回应")
        arm = Arm(bus, ids)
        arm.check_position_mode()
        return arm
    except BaseException:
        bus.close()
        raise


def moving_axis(home, view):
    changed = [sid for sid in home if abs(home[sid] - view[sid]) > POSE_TOLERANCE]
    if len(changed) != 1 or changed[0] not in SCAN_AXES:
        raise ValueError("每个候选姿势只允许改变一个已支持的取景关节")
    sid = changed[0]
    limit = round(5 * TICKS_PER_DEG) if sid == LIFT else MAX_SWING
    if abs(home[sid] - view[sid]) > limit:
        raise ValueError(f"舵机 {sid} 的候选姿势超出允许的偏转角")
    return sid


def load_profile(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise ValueError("请先校准桌面取景姿势（见线下对局说明）")
    data = json.loads(path.read_text(encoding="utf-8"))

    def parse_pose(raw):
        pose = {int(sid): tick for sid, tick in raw.items()}
        if not set(AXES) <= pose.keys() <= set(JOINTS.values()) or any(
            type(tick) is not int or not 0 <= tick < TICKS_PER_REV for tick in pose.values()
        ):
            raise ValueError("桌面取景姿势文件中的关节读数无效")
        return pose

    home = parse_pose(data["home"])
    views = [parse_pose(pose) for pose in data["views"]]
    if not 1 <= len(views) <= MAX_VIEWS or any(view.keys() != home.keys() for view in views):
        raise ValueError("桌面取景需要 1–12 个候选姿势，且舵机编号须与起始姿势一致")
    targets = []
    for view in views:
        sid = moving_axis(home, view)
        targets.append((sid, view[sid]))
    raw_effects = data.get("effects") or []
    effects = {}
    for index, target in enumerate(targets):
        item = raw_effects[index] if index < len(raw_effects) else None
        if not isinstance(item, dict):
            continue
        center, scale = item.get("center"), item.get("scale")
        if (isinstance(center, list) and len(center) == 2
                and all(type(v) in (int, float) and -1 <= v <= 2 for v in center)
                and type(scale) in (int, float) and 0.3 <= scale <= 3):
            effects[target] = {"center": center, "scale": scale}
    return home, targets, effects


def load_poses(path):
    home, targets, _ = load_profile(path)
    return home, targets


def auto_poses(arm):
    """Anchor at the current pose and search small single-axis offsets."""
    home = arm.read_all()
    limits = {sid: arm.read_limits(sid) for sid in AXES}
    views = []
    # Wrist tilt first, since it changes the look-down angle with little travel.
    for degrees in (3, 7):
        for sid in reversed(AXES):
            lo, hi = limits[sid]
            for sign in (-1, 1):
                target = home[sid] + sign * round(degrees * TICKS_PER_DEG)
                if lo + POSE_TOLERANCE <= target <= hi - POSE_TOLERANCE:
                    views.append((sid, target))
    if not views:
        raise ValueError("当前位置附近没有可用于自动取景的关节余量")
    return home, views


class TableMotion:
    """An out-and-back sweep over bounded single-joint paths."""

    def __init__(self, arm, home, targets):
        self.arm, self.home, self.targets = arm, home, targets
        if set(arm.ids) != home.keys():
            raise ValueError("连接的舵机与校准记录不一致，请重新校准")
        for sid, tick in home.items():
            low, high = arm.read_limits(sid)
            if not all(low <= target <= high for target in
                       [tick] + [goal for axis, goal in targets if axis == sid]):
                raise ValueError(f"舵机 {sid} 的取景路径超出内部限位")
        actual = self.read_checked()
        self.axis = None
        self.current = (AXES[0], home[AXES[0]])
        displaced = [sid for sid, tick in home.items() if abs(actual[sid] - tick) > POSE_TOLERANCE]
        if displaced:
            sid = displaced[0]
            path = [view for view in targets if view[0] == sid]
            ticks = [home[sid], *(view[1] for view in path)]
            if (len(displaced) != 1 or not path
                    or not min(ticks) - POSE_TOLERANCE <= actual[sid] <= max(ticks) + POSE_TOLERANCE):
                raise ValueError("机械臂不在已标定路径内，请回到起始位置")
            self.axis = sid
            near = min(path, key=lambda view: abs(actual[sid] - view[1]))
            self.current = near if abs(actual[sid] - near[1]) <= POSE_TOLERANCE else None
        arm.hold_here()
        arm.set_motion_profile(sorted(set(sid for sid, _ in targets)), acceleration=25, speed=180)
        self.pending = []
        self.moving = False

    def read_checked(self):
        positions = self.arm.read_all()
        if any(self.arm.bus.last_error.get(sid) for sid in positions):
            raise ValueError("舵机报告异常，已停止取景")
        return positions

    def go(self, view):
        if view not in self.targets and (view[0] not in SCAN_AXES or view[1] != self.home.get(view[0])):
            raise ValueError("取景动作不在已标定路径中")
        # A change of axis goes back through the taught origin.
        self.pending = []
        if self.axis is not None and self.axis != view[0]:
            self.pending.append((self.axis, self.home[self.axis]))
        self.pending.append(view)
        self.moving = True

    def step(self):
        """Advance at most a fraction of a degree. The worker calls at most 10 times/sec."""
        sid, goal = self.pending[0]
        actual = self.read_checked()
        if any(abs(actual[i] - tick) > POSE_TOLERANCE for i, tick in self.home.items() if i != sid):
            raise ValueError("其他关节偏离校准位置，已停止取景")
        tick = actual[sid]
        taught = [self.home[sid], *(p for axis, p in self.targets if axis == sid)]
        if not min(taught) - POSE_TOLERANCE <= tick <= max(taught) + POSE_TOLERANCE:
            raise ValueError("关节偏离已校准的活动范围，已停止取景")
        self.axis = sid
        if abs(goal - tick) <= ARRIVAL_TOLERANCE:
            self.pending.pop(0)
        else:
            self.arm.set_goals({sid: tick + max(-STEP, min(STEP, goal - tick))})
        self.moving = bool(self.pending)
        return self.moving

    def stop(self):
        # Leave torque enabled: releasing torque can drop the arm.
        if self.moving and self.axis is not None:
            sid = self.axis
            self.arm.set_goals({sid: self.arm.read_pos(sid)})
        self.moving = False
        self.pending.clear()


def main():
    parser = argparse.ArgumentParser(description="记录璀璨宝石牌桌的安全取景姿势")
    parser.add_argument("--port")
    parser.add_argument("--out", default="~/.splendor_offline/arm_poses.json")
    args = parser.parse_args()
    cfg = load_config()
    try:
        arm = connect(args.port or cfg["port"], cfg["baudrate"])
        try:
            print("保持机械臂有支撑、取景路径无障碍。此程序只读取位置，不切换扭矩。")
            input("把相机摆到大致看全牌桌的起始姿势，稳定后按回车：")
            home = arm.read_all()
            views = []
            while len(views) < MAX_VIEWS:
                if input("从起始姿势沿一个关节试探（最多 15°），稳定后回车；输入 s 保存：").strip().lower() == "s":
                    break
                pose = arm.read_all()
                try:
                    moving_axis(home, pose)
                except ValueError as exc:
                    print(exc)
                    continue
                views.append(pose)
                print(f"已记录 {len(views)} 个候选姿势。请沿原路回到起始姿势。")
                input("回到起始姿势后按回车：")
                actual = arm.read_all()
                if any(abs(actual[sid] - tick) > POSE_TOLERANCE for sid, tick in home.items()):
                    raise ValueError("机械臂还未回到起始姿势，请重新校准")
            if not views:
                raise ValueError("至少需要一个候选姿势")
        finally:
            arm.bus.close()
        path = Path(args.out).expanduser()
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path.write_text(json.dumps({"home": home, "views": views}, indent=2) + "\n", encoding="utf-8")
        print(f"已保存到 {path}")
    except (ValueError, OSError, RuntimeError, EOFError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
