#!/usr/bin/env python3
"""SO-ARM100 / SO-ARM101 机械臂摄像头人脸追踪。

摄像头装在机械臂上（腕部或夹爪）。脚本用 OpenCV 检测人脸，按人脸偏离画面中心的角度，
转动 shoulder_pan（左右）和 wrist_flex（上下），让人脸保持在画面中央。

直接用 Feetech STS3215 串口协议控制舵机，只依赖 pyserial + opencv-python + numpy，
不依赖 lerobot 的版本；与 lerobot 的标定互不影响（只写 RAM 寄存器：扭矩、加速度、目标位置）。

常用：
    python face_track.py --list-cameras          # 找机械臂上的摄像头编号
    python face_track.py --read                  # 读各关节当前位置（检查串口/舵机）
    python face_track.py --camera 1              # 开始（默认先暂停，按空格开始追踪）
    python face_track.py --camera 1 --dry-run    # 只看检测和指令，不动机械臂

按键（在视频窗口里）：
    空格 开始/暂停追踪    x 反转左右方向    y 反转上下方向
    r 回到起始朝向        c 切换摄像头      s 保存设置和当前姿态
    q / Esc 退出
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------------------
# Feetech STS3215 协议（SO-ARM100/101 用的舵机）
# ---------------------------------------------------------------------------

ADDR_MIN_ANGLE_LIMIT = 9      # 2 字节，EEPROM（只读取）
ADDR_MAX_ANGLE_LIMIT = 11     # 2 字节，EEPROM（只读取）
ADDR_OPERATING_MODE = 33     # 0 = 位置模式；不修改 EEPROM
ADDR_TORQUE_ENABLE = 40       # 1 字节
ADDR_ACCELERATION = 41        # 1 字节，单位 100 步/s²，0 = 最大
ADDR_GOAL_POSITION = 42       # 2 字节
ADDR_GOAL_SPEED = 46          # 2 字节，步/s，0 = 最大
ADDR_PRESENT_POSITION = 56    # 2 字节
ADDR_PRESENT_LOAD = 60        # 2 字节
ADDR_PRESENT_TEMPERATURE = 63  # 1 字节

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

TICKS_PER_REV = 4096
TICKS_PER_DEG = TICKS_PER_REV / 360.0

JOINTS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

STATUS_ERROR_BITS = {
    0: "电压",
    1: "角度传感器",
    2: "过热",
    3: "过流",
    5: "过载",
}


class ServoError(RuntimeError):
    pass


def make_packet(servo_id: int, inst: int, params=()) -> bytes:
    body = [servo_id & 0xFF, len(params) + 2, inst & 0xFF, *[p & 0xFF for p in params]]
    checksum = (~sum(body)) & 0xFF
    return bytes([0xFF, 0xFF, *body, checksum])


def encode_le(value: int, size: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(size)]


def decode_le(data: bytes) -> int:
    return sum(b << (8 * i) for i, b in enumerate(data))


def decode_signed(value: int, sign_bit: int) -> int:
    """STS 系列用符号位（不是补码）表示负数。"""
    if value & (1 << sign_bit):
        return -(value & ((1 << sign_bit) - 1))
    return value


class FeetechBus:
    """最小的 STS3215 串口总线实现：ping / read / write / sync_write。"""

    def __init__(self, port: str | None = None, baudrate: int = 1_000_000,
                 reply_timeout: float = 0.03, ser=None):
        if ser is None:
            try:
                import serial  # pyserial
            except ImportError:
                sys.exit("缺少 pyserial：pip install pyserial")
            ser = serial.Serial(port, baudrate=baudrate, timeout=0.005)
        self.ser = ser
        self.port = port
        self.reply_timeout = reply_timeout
        self.last_error: dict[int, int] = {}
        self._warned_at: dict[int, float] = {}

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    # -- 底层收发 --
    def _read_exact(self, n: int, deadline: float) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if chunk:
                buf += chunk
            elif time.monotonic() > deadline:
                raise ServoError("读取超时")
        return buf

    def _read_status(self, servo_id: int, sent: bytes = b"") -> bytes:
        deadline = time.monotonic() + self.reply_timeout
        while True:
            raw, params = self._read_one_packet(servo_id, deadline)
            if raw != sent:  # 有些转接板会把发出的包回显，跳过
                return params

    def _read_one_packet(self, servo_id: int, deadline: float):
        prev = b""
        while True:  # 找包头 FF FF
            b = self._read_exact(1, deadline)
            if prev == b"\xff" and b == b"\xff":
                break
            prev = b
        sid = self._read_exact(1, deadline)[0]
        while sid == 0xFF:  # 多余的 FF
            sid = self._read_exact(1, deadline)[0]
        length = self._read_exact(1, deadline)[0]
        if length < 2:
            raise ServoError(f"舵机 {servo_id} 回包长度错误")
        rest = self._read_exact(length, deadline)
        error, params, checksum = rest[0], rest[1:-1], rest[-1]
        if (~(sid + length + sum(rest[:-1]))) & 0xFF != checksum:
            raise ServoError(f"舵机 {servo_id} 回包校验错误")
        raw = bytes([0xFF, 0xFF, sid, length]) + rest
        if sid != servo_id:
            raise ServoError(f"期望舵机 {servo_id} 回包，收到 {sid}")
        if raw != getattr(self, "_last_sent", None):
            self._note_error(sid, error)
        return raw, params

    def _note_error(self, sid: int, error: int):
        self.last_error[sid] = error
        if error:
            now = time.monotonic()
            if now - self._warned_at.get(sid, 0) > 2.0:
                self._warned_at[sid] = now
                names = [n for bit, n in STATUS_ERROR_BITS.items() if error & (1 << bit)]
                print(f"[警告] 舵机 {sid} 报告异常：{'、'.join(names) or hex(error)}", file=sys.stderr)

    def _txrx(self, servo_id: int, inst: int, params=(), retries: int = 2) -> bytes:
        packet = make_packet(servo_id, inst, params)
        last: Exception | None = None
        for _ in range(retries + 1):
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            self.ser.write(packet)
            self._last_sent = packet
            try:
                return self._read_status(servo_id, packet)
            except ServoError as e:
                last = e
        raise ServoError(f"舵机 {servo_id} 无响应（{last}）")

    # -- 指令 --
    def ping(self, servo_id: int, retries: int = 1) -> bool:
        try:
            self._txrx(servo_id, INST_PING, (), retries=retries)
            return True
        except ServoError:
            return False

    def read(self, servo_id: int, addr: int, size: int) -> int:
        data = self._txrx(servo_id, INST_READ, (addr, size))
        if len(data) != size:
            raise ServoError(f"舵机 {servo_id} 读地址 {addr} 返回 {len(data)} 字节")
        return decode_le(data)

    def write(self, servo_id: int, addr: int, value: int, size: int):
        self._txrx(servo_id, INST_WRITE, (addr, *encode_le(value, size)))

    def sync_write(self, addr: int, size: int, values: dict[int, int]):
        params = [addr, size]
        for sid, v in values.items():
            params += [sid, *encode_le(v, size)]
        self.ser.write(make_packet(BROADCAST_ID, INST_SYNC_WRITE, params))  # 广播无回包


def find_ports() -> list[str]:
    pats = ["/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*",
            "/dev/ttyACM*", "/dev/ttyUSB*"]
    ports: list[str] = []
    for p in pats:
        ports += sorted(glob.glob(p))
    return ports


def open_bus(port: str | None, baudrate: int, probe_id: int = 1) -> FeetechBus:
    if port:
        return FeetechBus(port, baudrate)
    ports = find_ports()
    if not ports:
        sys.exit("没找到串口。确认机械臂驱动板已上电并用 USB 接到电脑（ls /dev/cu.usb*），或用 --port 指定。")
    responders = []
    for p in ports:
        try:
            bus = FeetechBus(p, baudrate)
        except Exception as e:
            print(f"  {p}: 打不开（{e}）")
            continue
        if bus.ping(probe_id):
            responders.append(bus)
        else:
            bus.close()
    if len(responders) == 1:
        print(f"串口：{responders[0].port}")
        return responders[0]
    for b in responders:
        b.close()
    if not responders:
        sys.exit(f"这些串口上都没有舵机 {probe_id} 回应：{ports}\n检查舵机电源（驱动板要接 5V/12V 电源，不只是 USB）。")
    sys.exit("有多个串口都有舵机回应（可能主臂和从臂都接着），请用 --port 指定从臂：\n  "
             + "\n  ".join(b.port for b in responders))


# ---------------------------------------------------------------------------
# 机械臂
# ---------------------------------------------------------------------------

class Arm:
    def __init__(self, bus: FeetechBus, ids: list[int]):
        self.bus = bus
        self.ids = ids

    def read_pos(self, sid: int) -> int:
        pos = decode_signed(self.bus.read(sid, ADDR_PRESENT_POSITION, 2), 15)
        if not 0 <= pos < TICKS_PER_REV:
            raise ServoError(f"舵机 {sid} 位置读数异常：{pos}（应在 0–4095）")
        return pos

    def read_all(self) -> dict[int, int]:
        return {sid: self.read_pos(sid) for sid in self.ids}

    def read_limits(self, sid: int) -> tuple[int, int]:
        lo = self.bus.read(sid, ADDR_MIN_ANGLE_LIMIT, 2)
        hi = self.bus.read(sid, ADDR_MAX_ANGLE_LIMIT, 2)
        if lo >= hi:  # 0/0 表示多圈模式或未设置
            return 0, TICKS_PER_REV - 1
        return lo, hi

    def check_position_mode(self, ids=None):
        for sid in ids or self.ids:
            mode = self.bus.read(sid, ADDR_OPERATING_MODE, 1)
            if mode != 0:
                raise ServoError(f"舵机 {sid} 工作模式为 {mode}，不是位置模式（0）；停止操作，请先检查舵机配置")

    def hold_here(self, ids=None):
        """先把目标位置设成当前位置再上扭矩，避免上电瞬间跳动。"""
        selected = list(ids or self.ids)
        self.check_position_mode(selected)
        # 全部先检查，避免某个舵机不安全时其他关节已经锁定。
        positions = {}
        for sid in selected:
            a, b = self.read_pos(sid), self.read_pos(sid)
            if abs(a - b) > 20:
                raise ServoError(f"舵机 {sid} 两次读数差太多（{a} / {b}），不上扭矩")
            lo, hi = self.read_limits(sid)
            if not lo <= b <= hi:
                raise ServoError(f"舵机 {sid} 当前位置 {b} 不在内部限位 {lo}–{hi} 内；请调整姿态或检查标定")
            positions[sid] = b
        for sid, b in positions.items():
            self.bus.write(sid, ADDR_GOAL_POSITION, b, 2)
            goal = self.bus.read(sid, ADDR_GOAL_POSITION, 2)
            if goal != b:
                raise ServoError(f"舵机 {sid} 目标位置写入不一致（写 {b} 读回 {goal}），不上扭矩")
            self.bus.write(sid, ADDR_TORQUE_ENABLE, 1, 1)

    def release(self, ids=None):
        for sid in ids or self.ids:
            self.bus.write(sid, ADDR_TORQUE_ENABLE, 0, 1)

    def set_motion_profile(self, ids, acceleration: int, speed: int):
        for sid in ids:
            self.bus.write(sid, ADDR_ACCELERATION, acceleration, 1)
            self.bus.write(sid, ADDR_GOAL_SPEED, speed, 2)

    def set_goals(self, goals: dict[int, int]):
        self.bus.sync_write(ADDR_GOAL_POSITION, 2, {sid: clamp_tick(v) for sid, v in goals.items()})

    def move_smooth(self, targets: dict[int, int], seconds: float = 3.0):
        start = {sid: self.read_pos(sid) for sid in targets}
        steps = max(1, int(seconds * 50))
        for i in range(1, steps + 1):
            a = i / steps
            a = a * a * (3 - 2 * a)  # 缓入缓出
            self.set_goals({sid: round(start[sid] + (targets[sid] - start[sid]) * a) for sid in targets})
            time.sleep(seconds / steps)


def clamp_tick(v) -> int:
    return int(max(0, min(TICKS_PER_REV - 1, round(v))))


# ---------------------------------------------------------------------------
# 控制器（纯计算，便于测试）
# ---------------------------------------------------------------------------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class AxisController:
    """积分型视觉伺服：目标角 += ki × 误差角 × dt。

    - 摄像头在机械臂上，误差角直接对应关节需要转的角度，积分控制能消掉重力下垂造成的静差；
    - 每秒最大角速度限制、目标领先实际位置的最大角度（防积分饱和）、关节范围限制。
    单位：度。
    """

    def __init__(self, ki=2.5, max_rate=120.0, max_lead=12.0, deadband=1.5,
                 lo=-math.inf, hi=math.inf):
        self.ki, self.max_rate, self.max_lead, self.deadband = ki, max_rate, max_lead, deadband
        self.lo, self.hi = lo, hi
        self.goal: float | None = None

    def reset(self, pos: float):
        self.goal = clamp(pos, self.lo, self.hi)

    def _finish(self, g: float, present: float) -> float:
        g = clamp(g, present - self.max_lead, present + self.max_lead)
        self.goal = clamp(g, self.lo, self.hi)
        return self.goal

    def update(self, err: float, present: float, dt: float) -> float:
        if self.goal is None:
            self.reset(present)
        if abs(err) <= self.deadband:
            err = 0.0
        else:
            err -= math.copysign(self.deadband, err)
        step = clamp(self.ki * err * dt, -self.max_rate * dt, self.max_rate * dt)
        return self._finish(self.goal + step, present)

    def toward(self, target: float, rate: float, present: float, dt: float) -> float:
        if self.goal is None:
            self.reset(present)
        step = clamp(target - self.goal, -rate * dt, rate * dt)
        return self._finish(self.goal + step, present)


class DirectionGuard:
    """方向配反时自动纠正：关节已经朝"减小误差"的方向转了 move_deg 度，
    误差反而变大 grow_deg 度以上，说明方向反了。"""

    def __init__(self, move_deg=8.0, grow_deg=3.0):
        self.move_deg, self.grow_deg = move_deg, grow_deg
        self.start = None

    def reset(self):
        self.start = None

    def check(self, present: float, err_abs: float) -> bool:
        if self.start is None:
            self.start = (present, err_abs)
            return False
        p0, e0 = self.start
        if abs(present - p0) < self.move_deg:
            return False
        self.start = (present, err_abs)
        return err_abs - e0 >= self.grow_deg


def pixel_to_angle(offset_px: float, half_size_px: float, half_fov_deg: float) -> float:
    """像素偏移 → 角度（针孔相机模型）。"""
    return math.degrees(math.atan(offset_px / half_size_px * math.tan(math.radians(half_fov_deg))))


def vertical_fov(hfov_deg: float, w: int, h: int) -> float:
    return 2 * math.degrees(math.atan(math.tan(math.radians(hfov_deg / 2)) * h / w))


class TargetSelector:
    """多张脸时优先跟住上一次的那张，否则选最大的；对中心做平滑。"""

    def __init__(self, alpha=0.6, max_jump=0.25):
        self.alpha, self.max_jump = alpha, max_jump
        self.center: tuple[float, float] | None = None

    def reset(self):
        self.center = None

    def update(self, faces, w: int, h: int):
        if not faces:
            return None
        pick = None
        if self.center is not None:
            cx0, cy0 = self.center
            best = min(faces, key=lambda f: math.hypot(f[0] + f[2] / 2 - cx0, f[1] + f[3] / 2 - cy0))
            d = math.hypot(best[0] + best[2] / 2 - cx0, best[1] + best[3] / 2 - cy0)
            if d < self.max_jump * max(w, h):
                pick = best
        if pick is None:
            pick = max(faces, key=lambda f: f[2] * f[3])
            self.center = None
        cx, cy = pick[0] + pick[2] / 2, pick[1] + pick[3] / 2
        if self.center is None:
            self.center = (cx, cy)
        else:
            a = self.alpha
            self.center = (a * cx + (1 - a) * self.center[0], a * cy + (1 - a) * self.center[1])
        return pick, self.center


# ---------------------------------------------------------------------------
# 人脸检测
# ---------------------------------------------------------------------------

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


class FaceDetector:
    """默认 YuNet；Haar 仅在明确指定时使用。"""

    def __init__(self, cv2, kind="auto", yunet_path=None, proc_width=640, min_score=0.7):
        self.cv2 = cv2
        self.proc_width = proc_width
        self.kind = None
        yunet_path = yunet_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", YUNET_NAME)
        if kind in ("auto", "yunet"):
            if not hasattr(cv2, "FaceDetectorYN"):
                sys.exit("当前 OpenCV 不支持 YuNet，请升级：pip install -U opencv-python")
            if not os.path.exists(yunet_path):
                self._download(yunet_path)
            try:
                self.yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 240), min_score, 0.3, 50)
            except Exception as e:
                sys.exit(f"YuNet 模型加载失败：{e}。请检查模型文件和 OpenCV 版本。")
            self.kind = "yunet"
        else:
            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self.haar = cv2.CascadeClassifier(path)
            if self.haar.empty():
                sys.exit(f"Haar 模型加载失败：{path}")
            self.kind = "haar"

    @staticmethod
    def _download(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"下载 YuNet 模型到 {path} …")
        try:
            urllib.request.urlretrieve(YUNET_URL, path + ".part")
            os.replace(path + ".part", path)
        except Exception as e:
            sys.exit(f"YuNet 模型下载失败（{e}）。请手动下载 {YUNET_URL} 放到 {path}")

    def detect(self, frame):
        """返回 [(x, y, w, h, score), ...]，坐标是原图像素。"""
        cv2 = self.cv2
        h, w = frame.shape[:2]
        s = self.proc_width / w if w > self.proc_width else 1.0
        small = cv2.resize(frame, (round(w * s), round(h * s))) if s != 1.0 else frame
        out = []
        if self.kind == "yunet":
            self.yunet.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self.yunet.detect(small)
            if faces is not None:
                for f in faces:
                    out.append((f[0] / s, f[1] / s, f[2] / s, f[3] / s, float(f[14])))
        else:
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            m = max(24, small.shape[1] // 12)
            for (x, y, fw, fh) in self.haar.detectMultiScale(gray, 1.15, 5, minSize=(m, m)):
                out.append((x / s, y / s, fw / s, fh / s, 1.0))
        return out


# ---------------------------------------------------------------------------
# 摄像头
# ---------------------------------------------------------------------------

def open_camera(cv2, index: int, width: int, height: int):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 尽量减少延迟
    except Exception:
        pass
    for _ in range(10):  # 预热，丢掉开头的黑帧
        ok, _ = cap.read()
        if ok:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


def rotate_frame(cv2, frame, rotate: int):
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def list_cameras(cv2, max_index=6):
    import numpy as np
    tiles = []
    for i in range(max_index):
        cap = open_camera(cv2, i, 640, 480)
        if cap is None:
            continue
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        print(f"  摄像头 {i}: {frame.shape[1]}x{frame.shape[0]}")
        t = cv2.resize(frame, (480, 360))
        cv2.putText(t, f"camera {i}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        tiles.append(t)
    if not tiles:
        print("没有可用的摄像头。macOS 上要给终端（Terminal/iTerm/VS Code）摄像头权限：系统设置 → 隐私与安全性 → 摄像头。")
        return
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    grid = np.vstack(rows)
    cv2.imshow("cameras (any key to close)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_track_config.json")
DEFAULTS = {
    "port": None, "baudrate": 1_000_000, "camera": 0, "width": 640, "height": 480, "rotate": 0,
    "hfov": 65.0, "pan_joint": "shoulder_pan", "tilt_joint": "wrist_flex",
    "pan_sign": 1, "tilt_sign": 1, "pan_range": 75.0, "tilt_range": 45.0,
    "ki": 2.5, "max_rate": 120.0, "deadband": 1.5, "max_lead": 12.0,
    "acceleration": 60, "return_after": 6.0, "detector": "auto", "saved_pose": None,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg: dict):
    keep = {k: cfg.get(k) for k in DEFAULTS}
    with open(CONFIG_PATH, "w") as f:
        json.dump(keep, f, indent=2, ensure_ascii=False)
    print(f"已保存设置：{CONFIG_PATH}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SO-ARM100/101 摄像头人脸追踪",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--port", help="串口，如 /dev/cu.usbmodem5A7A0185681；默认自动找")
    p.add_argument("--baudrate", type=int)
    p.add_argument("--camera", type=int, help="摄像头编号（用 --list-cameras 看）")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], help="摄像头装歪时旋转画面")
    p.add_argument("--hfov", type=float, help="摄像头水平视场角（度），默认 65")
    p.add_argument("--pan-joint", choices=list(JOINTS))
    p.add_argument("--tilt-joint", choices=list(JOINTS))
    p.add_argument("--pan-sign", type=int, choices=[-1, 1])
    p.add_argument("--tilt-sign", type=int, choices=[-1, 1])
    p.add_argument("--pan-range", type=float, help="左右最多离开起始朝向多少度，默认 75")
    p.add_argument("--tilt-range", type=float, help="上下最多离开起始朝向多少度，默认 45")
    p.add_argument("--ki", type=float, help="追踪增益（1/秒），默认 2.5（左右轴另乘 1.4）；抖动就调小，跟得慢就调大")
    p.add_argument("--max-rate", type=float, help="最大转速（度/秒），默认 120")
    p.add_argument("--deadband", type=float, help="死区（度），默认 1.5")
    p.add_argument("--acceleration", type=int, help="舵机加速度寄存器 0-254，越小越柔和，默认 60")
    p.add_argument("--return-after", type=float, help="丢失人脸多少秒后回到起始朝向，0 = 不回，默认 6")
    p.add_argument("--detector", choices=["auto", "yunet", "haar"], help="默认使用 YuNet；haar 可手动选择")
    p.add_argument("--no-auto-direction", action="store_true", help="关闭上下方向自动纠正（左右方向始终手动切换）")
    p.add_argument("--autostart", action="store_true", help="启动后直接开始追踪（默认先暂停）")
    p.add_argument("--dry-run", action="store_true", help="不连机械臂，只看检测效果")
    p.add_argument("--pose-by-hand", action="store_true", help="先松开扭矩，手动摆好姿势再锁定")
    p.add_argument("--goto-saved-pose", action="store_true", help="先缓慢移动到按 s 保存过的姿态")
    p.add_argument("--release-on-exit", action="store_true", help="退出时松开扭矩（机械臂会落下！）")
    p.add_argument("--list-cameras", action="store_true", help="列出摄像头并显示画面")
    p.add_argument("--read", action="store_true", help="读取各关节位置后退出")
    p.add_argument("--release", action="store_true", help="松开所有关节扭矩后退出（先扶住机械臂）")
    return p.parse_args(argv)


def merged_config(args) -> dict:
    cfg = load_config()
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def import_cv2():
    try:
        import cv2
    except ImportError:
        sys.exit("缺少 OpenCV：pip install opencv-python")
    return cv2


def connect_arm(cfg) -> Arm:
    """只使用实际接着的舵机（例如摘掉了 wrist_roll / gripper 也能用）。"""
    bus = open_bus(cfg["port"], cfg["baudrate"])
    present = [sid for sid in JOINTS.values() if bus.ping(sid)]
    missing = [n for n, sid in JOINTS.items() if sid not in present]
    lost = [n for n in (cfg["pan_joint"], cfg["tilt_joint"]) if JOINTS[n] not in present]
    if lost:
        bus.close()
        sys.exit(f"追踪要用的舵机没有回应：{lost}。检查舵机编号和连线。")
    if missing:
        print(f"未连接（忽略）：{missing}")
    return Arm(bus, present)


def print_positions(arm: Arm):
    pos = arm.read_all()
    for name, sid in JOINTS.items():
        if sid not in arm.ids:
            continue
        t = arm.bus.read(sid, ADDR_PRESENT_TEMPERATURE, 1)
        print(f"  {sid} {name:<14} 位置 {pos[sid]:5d}  ({pos[sid] / TICKS_PER_DEG:6.1f}°)  温度 {t}°C")
    return pos


def setup_pose(arm: Arm, cfg, args):
    arm.check_position_mode()
    if args.pose_by_hand:
        input("即将松开所有关节扭矩，请先用手扶住机械臂，然后按回车 …")
        arm.release()
        input("把机械臂摆到合适姿势（摄像头大致朝向人脸的位置），扶稳后按回车锁定 …")
        arm.hold_here()
    else:
        arm.hold_here()
        if args.goto_saved_pose:
            if not cfg.get("saved_pose"):
                print("还没有保存过姿态（追踪时按 s 保存），跳过")
            else:
                targets = {int(k): int(v) for k, v in cfg["saved_pose"].items() if int(k) in arm.ids}
                print("缓慢移动到保存的姿态 …")
                arm.move_smooth(targets, 3.0)
                time.sleep(0.3)


def draw_overlay(cv2, frame, faces, target, state, info_lines, deadband_px):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.rectangle(frame, (cx - deadband_px, cy - deadband_px), (cx + deadband_px, cy + deadband_px),
                  (120, 120, 120), 1)
    for (x, y, fw, fh, _) in faces:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + fw), int(y + fh)), (0, 180, 255), 1)
    if target is not None:
        (x, y, fw, fh, _), (tx, ty) = target
        cv2.rectangle(frame, (int(x), int(y)), (int(x + fw), int(y + fh)), (0, 255, 0), 2)
        cv2.circle(frame, (int(tx), int(ty)), 4, (0, 255, 0), -1)
        cv2.line(frame, (cx, cy), (int(tx), int(ty)), (0, 255, 0), 1)
    colors = {"TRACKING": (0, 220, 0), "PAUSED": (0, 200, 255), "LOST": (0, 0, 255),
              "RETURNING": (255, 160, 0), "DRY-RUN": (255, 0, 255)}
    cv2.putText(frame, state, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.get(state, (255, 255, 255)), 2)
    for i, line in enumerate(info_lines):
        y = h - 12 - 20 * (len(info_lines) - 1 - i)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def run(args):
    cfg = merged_config(args)
    cv2 = import_cv2() if not args.read and not args.release else None

    if args.list_cameras:
        list_cameras(cv2)
        return

    detector = None if args.read or args.release else FaceDetector(cv2, cfg["detector"])
    if detector:
        print(f"人脸检测：{detector.kind}")

    arm = None
    if not args.dry_run:
        arm = connect_arm(cfg)
        if args.read:
            print_positions(arm)
            arm.bus.close()
            return
        if args.release:
            input("即将松开所有关节扭矩，机械臂会落下。用手扶住后按回车 …")
            arm.release()
            arm.bus.close()
            print("已松开。")
            return
        print("各关节当前位置：")
        print_positions(arm)
        setup_pose(arm, cfg, args)
    elif args.read or args.release:
        sys.exit("--dry-run 下不能用 --read / --release")

    pan_id, tilt_id = JOINTS[cfg["pan_joint"]], JOINTS[cfg["tilt_joint"]]
    if pan_id == tilt_id:
        sys.exit("pan 和 tilt 不能是同一个关节")

    cam_index = cfg["camera"]
    cap = open_camera(cv2, cam_index, cfg["width"], cfg["height"])
    if cap is None:
        sys.exit(f"打不开摄像头 {cam_index}。用 --list-cameras 查看；macOS 需给终端摄像头权限。")

    # 起始朝向、范围
    if arm:
        arm.set_motion_profile([pan_id, tilt_id], cfg["acceleration"], 0)
        home = {pan_id: arm.read_pos(pan_id) / TICKS_PER_DEG, tilt_id: arm.read_pos(tilt_id) / TICKS_PER_DEG}
        limits = {sid: tuple(v / TICKS_PER_DEG for v in arm.read_limits(sid)) for sid in (pan_id, tilt_id)}
    else:
        home = {pan_id: 180.0, tilt_id: 180.0}
        limits = {pan_id: (0.0, 360.0), tilt_id: (0.0, 360.0)}
    margin = 2.0
    ctrls = {}
    for sid, rng in ((pan_id, cfg["pan_range"]), (tilt_id, cfg["tilt_range"])):
        # 靠近限位时保留起始位置，不能让 reset 把目标强行推离当前位置。
        lo = max(home[sid] - rng, min(home[sid], limits[sid][0] + margin))
        hi = min(home[sid] + rng, max(home[sid], limits[sid][1] - margin))
        print(f"舵机 {sid}：起始 {home[sid]:.1f}°，可追踪范围 {lo:.1f}°–{hi:.1f}°（内部限位 {limits[sid][0]:.1f}°–{limits[sid][1]:.1f}°）")
        gain = cfg["ki"] * (1.4 if sid == pan_id else 1.0)
        ctrls[sid] = AxisController(gain, cfg["max_rate"], cfg["max_lead"], cfg["deadband"], lo, hi)
        ctrls[sid].reset(home[sid])
    present = dict(home)

    selector = TargetSelector()
    tracking = bool(args.autostart)
    guards = {pan_id: DirectionGuard(), tilt_id: DirectionGuard()}
    track_start = time.monotonic()
    last_seen = time.monotonic()
    prev_t = time.monotonic()
    fps = 0.0
    read_failures = 0
    last_sent: dict[int, int] = {}
    win = "SO-ARM face tracking"
    print("窗口里按：空格 开始/暂停 | x/y 反转方向 | r 回中 | c 换摄像头 | s 保存 | q 退出")
    if not tracking:
        print("当前是暂停状态：确认画面是机械臂上的摄像头后按空格开始。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frame = rotate_frame(cv2, frame, cfg["rotate"])
            now = time.monotonic()
            dt = clamp(now - prev_t, 1e-3, 0.1)
            prev_t = now
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
            h, w = frame.shape[:2]
            hfov = cfg["hfov"] if cfg["rotate"] in (0, 180) else vertical_fov(cfg["hfov"], h, w)
            vfov = vertical_fov(hfov, w, h)

            faces = detector.detect(frame)
            target = selector.update(faces, w, h)
            if target is not None:
                last_seen = now

            # 读实际位置
            if arm:
                try:
                    present = {sid: arm.read_pos(sid) / TICKS_PER_DEG for sid in (pan_id, tilt_id)}
                    read_failures = 0
                except ServoError as e:
                    read_failures += 1
                    if read_failures > 30:
                        tracking = False
                        print(f"[错误] 连续读位置失败，已暂停：{e}", file=sys.stderr)
                        read_failures = 0
            else:
                present = {sid: ctrls[sid].goal for sid in ctrls}

            err_pan = err_tilt = 0.0
            if target is not None:
                tx, ty = target[1]
                err_pan = pixel_to_angle(tx - w / 2, w / 2, hfov / 2)
                err_tilt = pixel_to_angle(ty - h / 2, h / 2, vfov / 2)

            lost_for = now - last_seen
            if target is None or not tracking:
                for g in guards.values():
                    g.reset()
            if not tracking:
                state = "PAUSED"
                for sid in ctrls:
                    ctrls[sid].reset(present[sid])
            elif target is not None:
                state = "TRACKING"
                # 刚开始追踪的 1 秒内限速（从 25% 逐渐到全速）
                ramp = clamp(0.25 + (now - track_start), 0.25, 1.0)
                for sid, err, key, name in ((pan_id, err_pan, "pan_sign", "左右"),
                                            (tilt_id, err_tilt, "tilt_sign", "上下")):
                    if sid == tilt_id and not args.no_auto_direction and guards[sid].check(present[sid], abs(err)):
                        cfg[key] *= -1
                        ctrls[sid].reset(present[sid])
                        guards[sid].reset()
                        print(f"检测到{name}方向反了，已自动反转：{key} = {cfg[key]}（按 s 保存）")
                    c = ctrls[sid]
                    base = c.max_rate
                    c.max_rate = base * ramp
                    c.update(cfg[key] * err, present[sid], dt)
                    c.max_rate = base
            elif cfg["return_after"] > 0 and lost_for > cfg["return_after"]:
                state = "RETURNING"
                selector.reset()
                for sid in ctrls:
                    ctrls[sid].toward(home[sid], 20.0, present[sid], dt)
            else:
                state = "LOST"  # 保持不动

            if tracking:
                goals = {sid: clamp_tick(ctrls[sid].goal * TICKS_PER_DEG) for sid in ctrls}
                if arm and goals != last_sent:
                    arm.set_goals(goals)
                    last_sent = goals
            if not arm:
                state = "DRY-RUN " + state if state != "PAUSED" else "DRY-RUN"

            info = [
                f"err pan {err_pan:+5.1f} tilt {err_tilt:+5.1f} deg | fps {fps:4.1f} | {detector.kind} | cam {cam_index}",
                f"{cfg['pan_joint']} {present[pan_id] - home[pan_id]:+6.1f} (sign {cfg['pan_sign']:+d})  "
                f"{cfg['tilt_joint']} {present[tilt_id] - home[tilt_id]:+6.1f} (sign {cfg['tilt_sign']:+d})",
                "SPACE start/pause  x/y flip  r home  c camera  s save  q quit",
            ]
            db_px = int(w / 2 * math.tan(math.radians(cfg["deadband"])) / math.tan(math.radians(hfov / 2)))
            draw_overlay(cv2, frame, faces, target, state.split(" ")[0] if arm else state, info, db_px)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                tracking = not tracking
                selector.reset()
                track_start = now
                print("开始追踪" if tracking else "暂停")
            elif key == ord("x"):
                cfg["pan_sign"] *= -1
                for g in guards.values():
                    g.reset()
                print(f"左右方向反转，pan_sign = {cfg['pan_sign']}（按 s 保存）")
            elif key == ord("y"):
                cfg["tilt_sign"] *= -1
                for g in guards.values():
                    g.reset()
                print(f"上下方向反转，tilt_sign = {cfg['tilt_sign']}（按 s 保存）")
            elif key == ord("r"):
                tracking = False
                for sid in ctrls:
                    ctrls[sid].reset(home[sid])
                if arm:
                    arm.move_smooth({sid: clamp_tick(home[sid] * TICKS_PER_DEG) for sid in ctrls}, 1.5)
                print("已回到起始朝向（暂停中，按空格继续）")
            elif key == ord("c"):
                cap.release()
                for k in range(1, 7):
                    nxt = (cam_index + k) % 6
                    new = open_camera(cv2, nxt, cfg["width"], cfg["height"])
                    if new is not None:
                        cap, cam_index = new, nxt
                        break
                else:
                    cap = open_camera(cv2, cam_index, cfg["width"], cfg["height"])
                cfg["camera"] = cam_index
                tracking = False
                print(f"切换到摄像头 {cam_index}（已暂停）")
            elif key == ord("s"):
                if arm:
                    cfg["saved_pose"] = {str(k): v for k, v in arm.read_all().items()}
                save_config(cfg)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arm:
            try:
                print("回到起始朝向 …")
                arm.move_smooth({sid: clamp_tick(home[sid] * TICKS_PER_DEG) for sid in ctrls}, 1.5)
                if args.release_on_exit:
                    arm.release()
                    print("已松开扭矩。")
                else:
                    print("机械臂保持当前姿态（扭矩开着）。要松开：python face_track.py --release")
            except ServoError as e:
                print(f"[警告] 退出时控制机械臂失败：{e}", file=sys.stderr)
            arm.bus.close()


if __name__ == "__main__":
    run(parse_args())
