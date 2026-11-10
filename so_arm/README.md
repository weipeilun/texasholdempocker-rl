# SO-ARM100/101 摄像头人脸追踪

`face_track.py`：用机械臂上的摄像头检测人脸，转动 **shoulder_pan（左右）** 和 **wrist_flex（上下）**，让人脸保持在画面中央。

- 直接走 Feetech STS3215 串口协议，只依赖 `pyserial`、`opencv-python`、`numpy`，不依赖 lerobot 版本。
- 只写 RAM 寄存器（扭矩开关、加速度、目标位置），不改 EEPROM，不影响 lerobot 标定。
- 安全措施：上扭矩前先把目标位置设成当前位置（不会跳）；左右 ±75°、上下 ±45° 以内（相对起始朝向，且不超过舵机里设的角度限位）；最大 120°/s；启动默认暂停。

## 安装

```bash
pip install pyserial opencv-python numpy
```

（装了 lerobot 的环境里已经有 pyserial。）macOS 第一次运行时要给终端（Terminal / iTerm / VS Code）**摄像头权限**：系统设置 → 隐私与安全性 → 摄像头。

## 第一次使用

```bash
cd so_arm
python face_track.py --list-cameras      # 看哪个编号是机械臂上的摄像头（Mac 自带摄像头通常也在里面）
python face_track.py --read              # 确认串口和 6 个舵机都能读到
python face_track.py --camera 1 --pose-by-hand
```

`--pose-by-hand`：先松开扭矩（**扶住机械臂**），手动把机械臂摆成摄像头大致朝前的姿势，回车锁定。

锁定前会检查舵机是否处于位置模式，以及手动姿态是否在舵机内部限位内；不符合时停止并显示具体舵机编号。程序不会自动修改舵机 EEPROM。启动后终端会显示两个追踪关节的起始角度和实际允许范围；如果左右范围偏小，请检查舵机内部限位及起始姿态。

窗口出来后是 **PAUSED**。确认画面是机械臂摄像头后按 **空格** 开始追踪：

- 机械臂往**反方向**转（人脸越跑越偏）→ 按 **x**（左右）或 **y**（上下）反转方向。左右方向只通过 **x** 手动切换；上下方向仍会自动纠正，可用 `--no-auto-direction` 关闭。
- 调好后按 **s**：保存摄像头编号、方向和当前姿态到 `face_track_config.json`。

之后直接：

```bash
python face_track.py --goto-saved-pose     # 缓慢回到保存的姿态再开始
python face_track.py --goto-saved-pose --autostart
```

## 按键

| 键 | 作用 |
|---|---|
| 空格 | 开始 / 暂停追踪 |
| x / y | 反转左右 / 上下方向 |
| r | 回到起始朝向（并暂停） |
| c | 切换到下一个摄像头 |
| s | 保存设置和当前姿态 |
| q / Esc | 退出（回到起始朝向，扭矩保持） |

## 调参

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ki` | 2.5 | 追踪增益（1/秒），左右轴使用 1.4 倍以加快跟随。来回晃就调小（1.5） |
| `--max-rate` | 120 | 最大转速，度/秒 |
| `--deadband` | 1.5 | 死区，度；画面中央灰框以内不动 |
| `--hfov` | 65 | 摄像头水平视场角。广角摄像头设大一些（如 90） |
| `--rotate` | 0 | 摄像头装歪了就旋转画面（90/180/270），否则左右/上下会混在一起 |
| `--pan-joint` / `--tilt-joint` | shoulder_pan / wrist_flex | 用哪两个关节 |
| `--pan-range` / `--tilt-range` | 75 / 45 | 相对起始朝向的最大偏转角 |
| `--return-after` | 6 | 丢失人脸多少秒后慢慢回到起始朝向；0 = 原地不动 |
| `--detector` | auto | 默认使用 YuNet，模型缺失时自动下载；`haar` 可手动切回旧检测器。YuNet 加载失败时会报错，不会自动切回 Haar |
| `--port` | 自动 | 主臂和从臂都插着时要指定从臂的串口 |

## 其他命令

```bash
python face_track.py --dry-run --camera 1   # 不连机械臂，只看检测效果
python face_track.py --release              # 松开所有扭矩（机械臂会落下，先扶住）
python test_face_track.py                   # 离线测试（模拟舵机总线 + 闭环仿真）
```

## 原理

摄像头在机械臂上，人脸偏离中心的像素换算成角度（针孔模型），用积分型控制：
`目标角 += ki × 误差角 × dt`，再限制最大角速度、目标领先实际位置不超过 12°、关节范围。
积分能消掉腕关节受重力下垂造成的静差；仿真里 25° 偏差约 1 秒收敛，摄像头延迟 200ms 时超调约 3°。
