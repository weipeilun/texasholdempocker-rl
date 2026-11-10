"""离线测试：模拟 STS3215 舵机总线 + 闭环追踪仿真。不需要机械臂和摄像头。

    python -m pytest test_face_track.py -q      （或直接 python test_face_track.py）
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_track as ft  # noqa: E402


class FakeServoSerial:
    """解析主机发来的包，按 STS3215 寄存器表回包。"""

    def __init__(self, ids=range(1, 7)):
        self.mem = {sid: bytearray(256) for sid in ids}
        for sid in ids:
            self._set(sid, ft.ADDR_PRESENT_POSITION, 2048 + 10 * sid, 2)
            self._set(sid, ft.ADDR_GOAL_POSITION, 1000, 2)  # 旧的目标位置，不能直接上扭矩
            self._set(sid, ft.ADDR_MAX_ANGLE_LIMIT, 4095, 2)
            self.mem[sid][ft.ADDR_PRESENT_TEMPERATURE] = 31
        self.out = bytearray()
        self.log = []

    def _set(self, sid, addr, v, n):
        self.mem[sid][addr:addr + n] = bytes(ft.encode_le(v, n))

    def _get(self, sid, addr, n):
        return ft.decode_le(self.mem[sid][addr:addr + n])

    def reset_input_buffer(self):
        self.out.clear()

    def read(self, n):
        chunk, self.out[:] = bytes(self.out[:n]), self.out[n:]
        return chunk

    def close(self):
        pass

    def _reply(self, sid, params=b""):
        body = [sid, len(params) + 2, 0, *params]
        self.out += bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])

    def write(self, pkt):
        assert pkt[:2] == b"\xff\xff"
        sid, length, inst = pkt[2], pkt[3], pkt[4]
        params = pkt[5:5 + length - 2]
        assert (~sum(pkt[2:-1])) & 0xFF == pkt[-1], "checksum"
        self.log.append((sid, inst, bytes(params)))
        if inst == ft.INST_SYNC_WRITE:
            addr, n = params[0], params[1]
            for i in range(2, len(params), n + 1):
                s = params[i]
                self.mem[s][addr:addr + n] = params[i + 1:i + 1 + n]
                self._after_write(s, addr)
            return
        if sid not in self.mem:
            return  # 不存在的舵机：不回包
        if inst == ft.INST_PING:
            self._reply(sid)
        elif inst == ft.INST_READ:
            addr, n = params
            self._reply(sid, self.mem[sid][addr:addr + n])
        elif inst == ft.INST_WRITE:
            addr = params[0]
            self.mem[sid][addr:addr + len(params) - 1] = params[1:]
            self._after_write(sid, addr)
            self._reply(sid)

    def _after_write(self, sid, addr):
        # 上扭矩时舵机会立刻冲向目标位置：记录这个瞬间的跳变
        if addr == ft.ADDR_TORQUE_ENABLE and self.mem[sid][ft.ADDR_TORQUE_ENABLE]:
            jump = abs(self._get(sid, ft.ADDR_GOAL_POSITION, 2) - self._get(sid, ft.ADDR_PRESENT_POSITION, 2))
            assert jump == 0, f"舵机 {sid} 上扭矩时会跳 {jump} 步"
        if addr == ft.ADDR_GOAL_POSITION and self.mem[sid][ft.ADDR_TORQUE_ENABLE]:
            self._set(sid, ft.ADDR_PRESENT_POSITION, self._get(sid, ft.ADDR_GOAL_POSITION, 2), 2)


def test_packet_format():
    # Feetech 文档里的例子：PING 1 号舵机
    assert ft.make_packet(1, ft.INST_PING) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])
    # 读 1 号舵机当前位置（地址 56，2 字节）
    assert ft.make_packet(1, ft.INST_READ, (56, 2)) == bytes([0xFF, 0xFF, 1, 4, 2, 56, 2, 0xBE])
    assert ft.decode_signed(0x8000 | 100, 15) == -100
    assert ft.decode_signed(2048, 15) == 2048


def test_bus_and_arm():
    ser = FakeServoSerial()
    bus = ft.FeetechBus(ser=ser)
    arm = ft.Arm(bus, list(range(1, 7)))
    assert bus.ping(1) and not bus.ping(9, retries=0)
    assert arm.read_all() == {sid: 2048 + 10 * sid for sid in range(1, 7)}
    assert arm.read_limits(1) == (0, 4095)
    arm.hold_here()  # FakeServoSerial 会检查上扭矩没有跳变
    assert all(ser.mem[s][ft.ADDR_TORQUE_ENABLE] == 1 for s in range(1, 7))
    arm.set_goals({1: 2100, 4: 1900})
    assert arm.read_pos(1) == 2100 and arm.read_pos(4) == 1900
    arm.move_smooth({1: 2000}, seconds=0.05)
    assert arm.read_pos(1) == 2000
    arm.release([1])
    assert ser.mem[1][ft.ADDR_TORQUE_ENABLE] == 0


def test_hold_rejects_non_position_mode_before_writes():
    ser = FakeServoSerial()
    ser.mem[1][ft.ADDR_OPERATING_MODE] = 1
    arm = ft.Arm(ft.FeetechBus(ser=ser), [1])
    try:
        arm.hold_here()
        raise AssertionError("应当拒绝非位置模式")
    except ft.ServoError as e:
        assert "工作模式" in str(e)
    assert not any(inst == ft.INST_WRITE for _, inst, _ in ser.log)


def test_hold_rejects_position_outside_limits_before_writes():
    ser = FakeServoSerial(ids=[1, 2])
    ser._set(2, ft.ADDR_MIN_ANGLE_LIMIT, 2500, 2)
    arm = ft.Arm(ft.FeetechBus(ser=ser), [1, 2])
    try:
        arm.hold_here()
        raise AssertionError("应当拒绝限位外的姿态")
    except ft.ServoError as e:
        assert "内部限位" in str(e)
    assert not any(inst == ft.INST_WRITE for _, inst, _ in ser.log)


def test_bad_checksum_rejected():
    ser = FakeServoSerial()
    bus = ft.FeetechBus(ser=ser, reply_timeout=0.01)
    orig = ser._reply

    def corrupt(sid, params=b""):
        orig(sid, params)
        ser.out[-1] ^= 0xFF
    ser._reply = corrupt
    try:
        bus.read(1, ft.ADDR_PRESENT_POSITION, 2)
        raise AssertionError("应当报错")
    except ft.ServoError:
        pass


def simulate(sign_true, sign_cfg, face_az=25.0, face_el=-12.0, seconds=4.0, fps=30, latency_frames=3,
             sag=1.0, guard=False):
    """仿真：关节一阶滞后 + 摄像头延迟 + 重力下垂。返回每帧的 (pan 误差, tilt 误差)。"""
    hfov, w, h = 65.0, 640, 480
    vfov = ft.vertical_fov(hfov, w, h)
    ctrl_p = ft.AxisController(lo=180 - 75, hi=180 + 75)
    ctrl_t = ft.AxisController(lo=180 - 45, hi=180 + 45)
    ctrl_p.reset(180)
    ctrl_t.reset(180)
    pos_p, pos_t = 180.0, 180.0
    signs = {"p": sign_cfg, "t": sign_cfg}
    guard_p, guard_t = ft.DirectionGuard(), ft.DirectionGuard()
    history = []
    errs = []
    dt = 1 / fps
    for _ in range(int(seconds * fps)):
        # 摄像头朝向 = 初始朝向 + sign_true × 关节转角
        cam_az = sign_true * (pos_p - 180)
        cam_el = -sign_true * (pos_t - 180)  # sign=+1：关节正转 = 摄像头朝下
        history.append((cam_az, cam_el))
        seen_az, seen_el = history[max(0, len(history) - 1 - latency_frames)]
        # 人脸在图像中的位置
        tx = w / 2 + (w / 2) * math.tan(math.radians(face_az - seen_az)) / math.tan(math.radians(hfov / 2))
        ty = h / 2 + (h / 2) * math.tan(math.radians(seen_el - face_el)) / math.tan(math.radians(vfov / 2))
        ep = ft.pixel_to_angle(tx - w / 2, w / 2, hfov / 2)
        et = ft.pixel_to_angle(ty - h / 2, h / 2, vfov / 2)
        errs.append((ep, et))
        if guard:
            for key, g, ctrl, pos, e in (("p", guard_p, ctrl_p, pos_p, ep), ("t", guard_t, ctrl_t, pos_t, et)):
                if g.check(pos, abs(e)):
                    signs[key] *= -1
                    ctrl.reset(pos)
                    g.reset()
        gp = ctrl_p.update(signs["p"] * ep, pos_p, dt)
        gt = ctrl_t.update(signs["t"] * et, pos_t, dt)
        # 舵机：一阶滞后（约 80ms）+ 最大 200°/s；tilt 有重力下垂
        for g, name in ((gp, "p"), (gt, "t")):
            cur = pos_p if name == "p" else pos_t
            tgt = g - (sag if name == "t" else 0)
            step = ft.clamp((tgt - cur) * (1 - math.exp(-dt / 0.08)), -200 * dt, 200 * dt)
            if name == "p":
                pos_p += step
            else:
                pos_t += step
    return errs


def test_tracking_converges_without_big_overshoot():
    for sign in (1, -1):
        errs = simulate(sign_true=sign, sign_cfg=sign)
        final_p, final_t = errs[-1]
        assert abs(final_p) < 2.0 and abs(final_t) < 2.0, errs[-1]   # 收敛到死区附近
        # 不应越过人脸很多（超调）
        overshoot = max(-e[0] for e in errs)  # 初始误差为正，越过后变负
        assert overshoot < 3.0, overshoot
        # 1.5 秒内误差缩小到 1/4 以下
        assert abs(errs[45][0]) < abs(errs[0][0]) / 4


def test_wrong_sign_runs_away_but_stays_in_range():
    errs = simulate(sign_true=1, sign_cfg=-1, seconds=3.0)
    assert abs(errs[-1][0]) > abs(errs[0][0])  # 方向反了：越追越远（用户按 x 反转）


def test_auto_direction_fixes_wrong_sign():
    for sign in (1, -1):
        errs = simulate(sign_true=sign, sign_cfg=-sign, guard=True)
        assert abs(errs[-1][0]) < 2.0 and abs(errs[-1][1]) < 2.0, errs[-1]
        worst = max(abs(e[0]) for e in errs)
        assert worst < 25.0 + 15.0, worst   # 反方向最多跑出十几度就纠正


def test_auto_direction_no_false_flip():
    for sign in (1, -1):
        for lat in (3, 6):
            errs = simulate(sign_true=sign, sign_cfg=sign, guard=True, latency_frames=lat)
            assert max(-e[0] for e in errs) < 4.0 and abs(errs[-1][0]) < 2.0


def test_echo_adapter():
    ser = FakeServoSerial()
    orig = ser.write

    def echo_write(pkt):
        ser.out += pkt  # 转接板回显
        orig(pkt)
    ser.write = echo_write
    bus = ft.FeetechBus(ser=ser)
    arm = ft.Arm(bus, [1])
    assert arm.read_pos(1) == 2058
    assert not bus.ping(9, retries=0)  # 回显不能被当成回应


def test_bad_position_not_clamped():
    ser = FakeServoSerial()
    ser._set(1, ft.ADDR_PRESENT_POSITION, 0x8000 | 37, 2)  # 负数读数
    arm = ft.Arm(ft.FeetechBus(ser=ser), [1])
    try:
        arm.hold_here()
        raise AssertionError("应当报错")
    except ft.ServoError:
        pass
    assert ser.mem[1][ft.ADDR_TORQUE_ENABLE] == 0   # 没上扭矩
    assert ft.decode_le(ser.mem[1][ft.ADDR_GOAL_POSITION:ft.ADDR_GOAL_POSITION + 2]) == 1000  # 目标没被改成 0


def test_target_selector_sticks_to_same_face():
    sel = ft.TargetSelector()
    a = (100, 100, 50, 50, 1.0)
    b = (400, 100, 80, 80, 1.0)   # 更大的另一张脸
    pick, _ = sel.update([a], 640, 480)
    assert pick == a
    pick, _ = sel.update([b, (104, 102, 50, 50, 1.0)], 640, 480)
    assert pick[0] == 104   # 跟住原来那张，不跳到更大的脸
    sel.reset()
    pick, _ = sel.update([a, b], 640, 480)
    assert pick == b


def test_controller_limits():
    c = ft.AxisController(ki=100, max_rate=90, max_lead=12, lo=100, hi=200)
    c.reset(150)
    g = c.update(50, 150, 0.1)
    assert abs(g - 159) < 1e-9          # 被最大角速度 90°/s 限制
    for _ in range(10):
        g = c.update(50, 150, 0.1)
    assert g == 162                      # 不超前实际位置 12°
    for _ in range(100):
        g = c.update(50, g, 0.1)
    assert g == 200                      # 关节范围


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
