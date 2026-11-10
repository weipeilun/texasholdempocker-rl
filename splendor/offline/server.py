"""Web console of offline (physical table) play. Standard library HTTP server, like splendor/server.py.

    python -m splendor.offline [--host 127.0.0.1] [--port 8766] [--camera 0|URL] [--certfile C --keyfile K]

Pages: /  console (state, history, hints, alerts, manual controls), /screen  big screen with the robots' instructions
for their human proxies, /camera  a browser page that streams a camera to the server, /settings  vision model
(OpenAI-compatible base URL / relay, API key, model), /login.

Camera access in a browser needs a secure context: open the pages on localhost (e.g. through `ssh -L`) or start
the server with --certfile/--keyfile. A camera on this machine (USB index or RTSP/HTTP URL) works without a browser:
--camera."""

from __future__ import annotations

import argparse
import ipaddress
import json
import mimetypes
import os
import shutil
import ssl
import threading
import time
from dataclasses import asdict
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..engine import CARDS, NOBLES
from .accounts import Store
from .arm_camera import ArmCamera, assess_table, view_score
from .chatgpt import ChatGPTVision
from .robots import POLICY_LABELS, ModelProvider, make_robot
from .scan_memory import ScanMemory
from .session import OfflineSession
from .simulate import vision_json
from .table import card_text, noble_text, normalize_vision
from .vision import SUGGESTED_MODELS, TERRA_MODEL, VisionError, list_models, recognize

WEB = Path(__file__).parent / "web"
PAGES = {"/": "index.html", "/screen": "screen.html", "/camera": "camera.html", "/settings": "settings.html",
         "/login": "login.html"}
MAX_FRAME = 12_000_000


class Frames:
    """the latest camera frame (JPEG) from a browser page or a local camera"""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg, self.time, self.seq, self.source = None, None, 0, None

    def put(self, jpeg, source):
        with self.lock:
            self.jpeg, self.time, self.source = jpeg, time.time(), source
            self.seq += 1

    def latest(self):
        with self.lock:
            return self.jpeg, self.time, self.seq, self.source

    def info(self):
        with self.lock:
            return {"seq": self.seq, "time": self.time, "source": self.source,
                    "bytes": len(self.jpeg) if self.jpeg else 0}


class LocalCamera(threading.Thread):
    """frames from a camera attached to this machine (OpenCV: device index or stream URL)"""

    def __init__(self, source, frames, period=1.0):
        super().__init__(daemon=True)
        self.source, self.frames, self.period = source, frames, period
        self.error = None

    def run(self):
        import cv2
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = None
        while True:
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(src)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                ok, frame = cap.read()
                if ok:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if ok:
                        self.frames.put(buf.tobytes(), f"本机摄像头 {self.source}")
                        self.error = None
                else:
                    self.error = "读取画面失败"
                    cap.release()
                    cap = None
            except Exception as e:      # noqa: BLE001
                self.error = str(e)
                cap = None
            time.sleep(self.period)


class App:
    def __init__(self, store, provider, log_dir, allow_register=False):
        self.store, self.provider, self.log_dir = store, provider, log_dir
        self.allow_register = allow_register
        self.frames = Frames()
        self.lock = threading.RLock()
        self.session = None
        self.owner = None              # the user whose vision settings this game uses
        self.vision_enabled = True
        self.vision_force = False
        self.vision_busy = False
        self.vision_last = None        # {"time", "ok", "error", "meta"}
        self.camera = None
        self.chatgpt = None
        self.arm = None
        self.arm_reading = None
        self.scan = ScanMemory()
        threading.Thread(target=self._vision_loop, daemon=True).start()

    # ----- game
    def new_game(self, user, seats):
        if len(seats) != 4:
            raise ValueError("需要 4 个座位")
        clean, robots = [], {}
        for i, s in enumerate(seats):
            kind = s.get("kind")
            if kind not in ("human", "robot"):
                raise ValueError("座位类型应为 human 或 robot")
            policy = s.get("policy") or "model"
            name = (str(s.get("name") or "").strip() or f"玩家{i + 1}")[:16]
            clean.append({"name": name, "kind": kind, "policy": policy if kind == "robot" else None})
            if kind == "robot":
                robots[i] = make_robot(policy, self.provider, seed=int(time.time()) + i)
        with self.lock:
            self.session = OfflineSession(clean, robots, log_dir=self.log_dir)
            self.owner = user
            self.vision_force = True

    def _alert(self, key, text, level="error", details=None):
        s = self.session
        if s is not None:
            with s.lock:
                s._alert(key, text, level, details)
                s._bump()

    def _clear(self, key):
        s = self.session
        if s is not None:
            with s.lock:
                if key in s.alerts:
                    s._clear(key)
                    s._bump()

    # ----- vision
    def latest_frame(self):
        return self.arm.preview() if self.arm and self.arm.running else self.frames.latest()

    def vision_problem(self, user, settings):
        if settings["vision_provider"] == "chatgpt":
            if not self.chatgpt or not self.store.is_admin(user):
                return "ChatGPT 视觉只可在已启用此功能的本机管理员账号使用"
            if not self.chatgpt.status()["connected"]:
                return "请先在设置中连接本机 ChatGPT 账号"
        elif not settings.get("api_key"):
            return "还没有配置视觉模型的 API Key（设置页面）"
        return None

    def read_table(self, settings, jpeg, hint, *, arm_coverage, arm_plan=None):
        options = {"arm_coverage": True} if arm_coverage else {}
        if arm_plan:
            options["arm_plan"] = arm_plan
        reader = self.chatgpt.recognize if settings["vision_provider"] == "chatgpt" else recognize
        return reader(settings, jpeg, hint, **options)

    def _scan_scope(self, session, arm, user):
        if session is None:
            return None, arm.capture_epoch, user
        with session.lock:
            game = session.game
            return session, arm.capture_epoch, user, (game.turn, game.key()) if game else session.phase

    def _vision_loop(self):
        last_frame, last_call = None, 0.
        while True:
            time.sleep(0.2)
            try:
                with self.lock:
                    s = self.session
                    arm = self.arm if self.arm and self.arm.running else None
                    user = (self.owner if s is not None and s.phase != "terminal" else None)
                    user = user or (arm.operator if arm else None)
                    if user is None:
                        continue
                    settings = self.store.settings(user)
                    if arm:
                        sample = arm.sample()
                        if sample is None:
                            continue
                        (jpeg, frame_time, seq, _), revision = sample
                    else:
                        jpeg, frame_time, seq, _ = self.frames.latest()
                    now = time.time()
                    interval = float(settings.get("interval") or 5)
                    if arm and arm.auto:
                        interval = min(interval, 2)
                    if jpeg is None:
                        self._alert("no_frame", "还没有摄像头画面，请先打开相机或开始推流", "warn")
                        continue
                    self._clear("no_frame")
                    if now - frame_time > max(15., 3 * interval):
                        self._alert("stale_frame", f"摄像头画面已经 {int(now - frame_time)} 秒没有更新", "error")
                        continue
                    self._clear("stale_frame")
                    if not (self.vision_enabled or self.vision_force):
                        continue
                    problem = self.vision_problem(user, settings)
                    if problem:
                        self._alert("vision_config", problem, "error")
                        continue
                    self._clear("vision_config")
                    frame_key = (arm is not None, seq)
                    if not self.vision_force and (frame_key == last_frame or now - last_call < interval):
                        continue
                    if arm and frame_key == last_frame:
                        continue
                    self.vision_force, self.vision_busy = False, True
                    last_frame, last_call = frame_key, now
                    scope = self._scan_scope(s, arm, user) if arm else None
                    plan = None
                    if arm:
                        self.scan.prepare(scope, frame_time)
                        plan = arm.plan(self.scan.missing())
                    hint = (vision_json(s.game) if not arm and s and s.game
                            and settings.get("hint_expected") else None)
                try:
                    data, meta = self.read_table(settings, jpeg, hint, arm_coverage=bool(arm), arm_plan=plan)
                    meta["frame_time"] = frame_time
                    table = normalize_vision(data, strict=bool(arm))
                    with self.lock:
                        if s is not self.session or bool(arm) != bool(self.arm and self.arm.running):
                            continue
                        if arm:
                            with arm.lock:
                                if (arm.revision != revision or arm.sample() is None
                                        or self._scan_scope(s, arm, user) != scope):
                                    continue
                                current = s.game.current_player if s and s.game else None
                                coverage, complete = assess_table(data, table, current)
                                score = view_score(coverage, table)
                                if arm.auto or self.scan.values:
                                    table, coverage = self.scan.merge(table, coverage, current, stamp=frame_time)
                                    complete = table.visible
                                if not arm.accept(revision, coverage, complete,
                                                  self.vision_enabled and time.time() - frame_time <= 40,
                                                  score, data.get("next_view")):
                                    continue
                            meta["usable"] = complete
                            self.arm_reading = table.to_dict()
                        if table.bank is None:
                            meta["usable"] = False
                        if s is not None:
                            s.on_reading(table, meta)
                        self.vision_last = {"time": time.time(), "ok": True, "meta": meta}
                        self._clear("vision_api")
                except (VisionError, ValueError) as e:
                    self.vision_last = {"time": time.time(), "ok": False, "error": str(e)}
                    self._alert("vision_api", "调用视觉模型失败", "error", [str(e)])
                finally:
                    self.vision_busy = False
            except Exception as e:      # noqa: BLE001  (keep the loop alive; show the error)
                self.vision_last = {"time": time.time(), "ok": False, "error": f"{type(e).__name__}: {e}"}
                self.vision_busy = False

    def test_vision(self, user):
        arm = self.arm if self.arm and self.arm.running else None
        sample = arm.sample() if arm else None
        if arm and sample is None:
            raise ValueError("请等待机械臂相机画面稳定")
        jpeg, _, _, _ = sample[0] if arm else self.frames.latest()
        if jpeg is None:
            raise ValueError("还没有摄像头画面")
        settings = self.store.settings(user)
        s = self.session
        hint = vision_json(s.game) if (s is not None and s.game is not None and settings.get("hint_expected")) else None
        problem = self.vision_problem(user, settings)
        if problem:
            raise VisionError(problem)
        data, meta = self.read_table(settings, jpeg, None if arm else hint, arm_coverage=bool(arm))
        table = normalize_vision(data)
        if arm:
            assess_table(data, table)
        summary = {
            "market": [[card_text(c) for c in row] for row in table.market] if table.market else None,
            "nobles": [noble_text(n) for n in table.nobles] if table.nobles is not None else None,
            "bank": table.bank, "seats": [vars(x) for x in table.seats], "issues": table.issues,
            "visible": table.visible, "confidence": table.confidence,
        }
        return {"raw": data, "summary": summary, "meta": meta}

    def view(self, user):
        s = self.session
        jpeg, frame_time, seq, source = self.latest_frame()
        arm = self.arm.view() if self.arm else None
        if arm is not None:
            operator = self.owner if s and s.phase != "terminal" else self.arm.operator
            arm["vision_configured"] = bool(
                operator and not self.vision_problem(operator, self.store.settings(operator))
            )
            arm["reading"] = self.arm_reading if arm["running"] and s is None else None
        out = {"user": user, "admin": self.store.is_admin(user), "game": None if s is None else s.view(),
               "owner": self.owner, "vision": {"enabled": self.vision_enabled, "busy": self.vision_busy,
                                               "last": self.vision_last},
               "frame": {"seq": seq, "time": frame_time, "source": source, "bytes": len(jpeg) if jpeg else 0},
               "arm": arm, "model": self.provider.describe(),
               "camera_error": None if self.camera is None else self.camera.error,
               "now": time.time()}
        return out


def _cards_payload():
    return {"cards": {c: {**asdict(card), "text": card_text(c)} for c, card in CARDS.items()},
            "nobles": {n: {**asdict(noble), "text": noble_text(n)} for n, noble in NOBLES.items()},
            "policies": POLICY_LABELS, "models": SUGGESTED_MODELS}


def make_handler(app, secure=False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SplendorOffline/1"

        # ----- helpers
        def _token(self):
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            return cookie["sid"].value if "sid" in cookie else None

        def _user(self):
            return app.store.user_of(self._token())

        def send_json(self, data, status=200, cookie=None):
            body = json.dumps(data, ensure_ascii=False, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            if cookie is not None:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def send_bytes(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def redirect(self, where):
            self.send_response(302)
            self.send_header("Location", where)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _cookie(self, token, max_age):
            flags = "; Secure" if secure else ""
            return f"sid={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}{flags}"

        def log_message(self, fmt, *args):
            pass

        # ----- GET
        def do_GET(self):
            path = urlparse(self.path).path
            user = self._user()
            if path in PAGES:
                if path != "/login" and user is None:
                    self.redirect("/login")
                    return
                self.send_bytes((WEB / PAGES[path]).read_bytes(), "text/html; charset=utf-8")
                return
            if path.startswith("/static/"):
                name = path[len("/static/"):]
                file = (WEB / name).resolve()
                if WEB.resolve() not in file.parents or not file.is_file():
                    self.send_json({"error": "Not found"}, 404)
                    return
                ctype = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                self.send_bytes(file.read_bytes(), ctype)
                return
            if path == "/api/me":
                self.send_json({"user": user, "has_users": app.store.has_users(),
                                "can_register": app.allow_register or not app.store.has_users()})
                return
            if user is None:
                self.send_json({"error": "请先登录"}, 401)
                return
            if path == "/api/state":
                self.send_json(self._view(user))
            elif path == "/api/cards":
                self.send_json(_cards_payload())
            elif path == "/api/arm/frame.jpg":
                jpeg, _, seq, _ = app.arm.preview() if app.arm else (None, None, 0, None)
                after = parse_qs(urlparse(self.path).query).get("after", [""])[0]
                if jpeg is None or after == str(seq):
                    self.send_response(204)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Frame-Seq", str(seq))
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
            elif path == "/api/frame.jpg":
                jpeg, _, _, _ = app.latest_frame()
                if jpeg is None:
                    self.send_json({"error": "还没有画面"}, 404)
                else:
                    self.send_bytes(jpeg, "image/jpeg")
            elif path == "/api/settings":
                self.send_json(app.store.public_settings(user))
            elif path == "/api/chatgpt":
                if app.chatgpt and self._local_admin(user):
                    self.send_json(app.chatgpt.status())
                else:
                    self.send_json({"available": False, "connected": False, "signing_in": False})
            elif path == "/api/models":
                s = app.store.settings(user)
                try:
                    if s["vision_provider"] == "chatgpt":
                        if not self._local_admin(user):
                            raise VisionError("请在本机用管理员账号连接 ChatGPT")
                        problem = app.vision_problem(user, s)
                        if problem:
                            raise VisionError(problem)
                        self.send_json({"models": [TERRA_MODEL], "suggested": [TERRA_MODEL]})
                    else:
                        self.send_json({"models": list_models(s["base_url"], s["api_key"]), "suggested": SUGGESTED_MODELS})
                except VisionError as e:
                    self.send_json({"error": str(e), "suggested": SUGGESTED_MODELS}, 400)
            else:
                self.send_json({"error": "Not found"}, 404)

        # ----- POST
        def do_POST(self):
            path = urlparse(self.path).path
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip()
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            if path == "/api/frame":
                if self._user() is None:
                    self.send_json({"error": "请先登录"}, 401)
                    return
                if ctype not in ("image/jpeg", "image/jpg") or not 0 < length <= MAX_FRAME:
                    self.send_json({"error": "需要 JPEG 图片"}, 400)
                    return
                body = self.rfile.read(length)
                source = parse_qs(urlparse(self.path).query).get("source", ["浏览器摄像头"])[0][:40]
                app.frames.put(body, source)
                self.send_json({"ok": True, "seq": app.frames.info()["seq"]})
                return
            try:
                # JSON only: a cross-site form cannot send it (CSRF), like splendor/server.py
                if ctype != "application/json":
                    raise ValueError("只接受 JSON 请求")
                if not 0 <= length <= 2_000_000:
                    raise ValueError("请求体大小无效")
                data = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/login":
                    token = app.store.login(str(data.get("username", "")), str(data.get("password", "")))
                    self.send_json({"ok": True}, cookie=self._cookie(token, 7 * 24 * 3600))
                    return
                if path == "/api/register":
                    if not (app.allow_register or not app.store.has_users()):
                        raise ValueError("不允许注册新账号（请联系管理员，或用 --allow-register 启动）")
                    first = not app.store.has_users()
                    app.store.create_user(str(data.get("username", "")), str(data.get("password", "")), admin=first)
                    token = app.store.login(str(data["username"]), str(data["password"]))
                    self.send_json({"ok": True}, cookie=self._cookie(token, 7 * 24 * 3600))
                    return
                if path == "/api/logout":
                    app.store.logout(self._token())
                    self.send_json({"ok": True}, cookie=self._cookie("", 0))
                    return
                user = self._user()
                if user is None:
                    self.send_json({"error": "请先登录"}, 401)
                    return
                result = self.route(path, data, user)
                if result is None:
                    self.send_json({"error": "Not found"}, 404)
                else:
                    self.send_json(result)
            except VisionError as e:
                self.send_json({"error": str(e)}, 400)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
                self.send_json({"error": str(e)}, 400)

        def _local_admin(self, user):
            return (ipaddress.ip_address(self.client_address[0]).is_loopback
                    and urlparse("//" + self.headers.get("Host", "")).hostname in ("localhost", "127.0.0.1", "::1")
                    and app.store.is_admin(user))

        def _view(self, user):
            view = app.view(user)
            if view["arm"] is not None:
                view["arm"]["control"] = self._local_admin(user)
            return view

        def route(self, path, data, user):
            s = app.session
            if path in ("/api/arm/start", "/api/arm/stop", "/api/arm/auto"):
                if app.arm is None:
                    raise ValueError("启动时需要指定 --arm-camera")
                if not self._local_admin(user):
                    raise ValueError("请在连接机械臂的 Mac 上使用管理员账号操作")
                with app.lock:
                    if path == "/api/arm/start":
                        if not app.arm.running:
                            app.arm_reading = None
                        app.arm.start(user)
                    elif path == "/api/arm/stop":
                        app.arm.close()
                        app.arm_reading = None
                    else:
                        enabled = data.get("enabled") is True
                        owner = app.owner if s and s.phase != "terminal" else app.arm.operator
                        if enabled and (not app.vision_enabled or not owner):
                            raise ValueError("请先启用自动识别，并在设置中配置视觉模型")
                        if enabled:
                            problem = app.vision_problem(owner, app.store.settings(owner))
                            if problem:
                                raise ValueError(problem)
                        app.arm.set_auto(enabled)
                        app.vision_force = enabled
                return self._view(user)
            if path == "/api/game/new":
                app.new_game(user, data.get("seats") or [])
                return self._view(user)
            if path == "/api/chatgpt/login":
                if not app.chatgpt or not self._local_admin(user):
                    raise ValueError("请在已启用 ChatGPT 视觉的本机控制台用管理员账号登录")
                return app.chatgpt.sign_in()
            if path == "/api/settings":
                if data.get("vision_provider") == "chatgpt" and (not app.chatgpt or not self._local_admin(user)):
                    raise ValueError("请在已启用 ChatGPT 视觉的本机控制台用管理员账号操作")
                app.store.save_settings(user, data)
                return app.store.public_settings(user)
            if path == "/api/password":
                app.store.change_password(user, str(data.get("old", "")), str(data.get("new", "")))
                return {"ok": True}
            if path == "/api/vision/test":
                if app.store.settings(user)["vision_provider"] == "chatgpt" and not self._local_admin(user):
                    raise ValueError("请在本机用管理员账号试识别")
                return app.test_vision(user)
            if path == "/api/vision/now":
                with app.lock:
                    if app.arm and app.arm.running:
                        with app.arm.lock:
                            app.scan.reset()
                            app.arm.capture_epoch += 1
                            app.arm.revision += 1
                    app.vision_force = True
                return self._view(user)
            if path == "/api/vision/toggle":
                with app.lock:
                    app.vision_enabled = bool(data.get("enabled"))
                    if not app.vision_enabled and app.arm and app.arm.auto:
                        app.arm.set_auto(False)
                return self._view(user)
            if s is None:
                raise ValueError("还没有对局：请先新建对局")
            if path == "/api/game/begin":
                if data.get("market") and data.get("nobles"):
                    s.begin(data["market"], data["nobles"], by="操作员")
                else:
                    s.begin_from_reading()
            elif path == "/api/game/action":
                s.manual_action(data["action"])
            elif path == "/api/game/reveal":
                s.manual_reveal(str(data["card_id"]))
            elif path == "/api/game/confirm-robot":
                s.confirm_robot()
            elif path == "/api/game/choose":
                s.choose_ambiguous(int(data["index"]))
            elif path == "/api/game/undo":
                s.undo()
            elif path == "/api/game/use-my-vision":
                app.owner = user
            else:
                return None
            return self._view(user)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="璀璨宝石线下对局：摄像头 + 视觉模型 + 机器人座位")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument("--data-dir", default="~/.splendor_offline", help="账号、设置与对局记录（私有目录）")
    parser.add_argument("--allow-register", action="store_true", help="允许在登录页注册新账号（默认只有第一个账号可自助创建）")
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument("--camera", default=None, help="本机摄像头：设备编号（如 0）或视频流地址（rtsp://…、http://…）")
    cameras.add_argument("--arm-camera", type=int, help="机械臂 USB 相机编号（在控制台打开）")
    parser.add_argument("--arm-port", help="机械臂串口（单个机械臂连接时可自动查找）")
    parser.add_argument("--arm-rotate", type=int, choices=(0, 90, 180, 270), help="机械臂相机的画面旋转角度")
    parser.add_argument("--arm-poses", help="已校准的取景姿势文件（默认数据目录下 arm_poses.json）")
    parser.add_argument("--chatgpt-vision", action="store_true", help="允许本机管理员用 ChatGPT 账号接入 GPT‑6.1 Terra")
    parser.add_argument("--camera-period", type=float, default=1.0, help="本机摄像头取帧间隔（秒）")
    parser.add_argument("--weights", default=None, help="机器人模型权重（默认 models/splendor_final/weights.pth）")
    parser.add_argument("--config", default=None, help="模型配置（默认 models/splendor_final/config.yml）")
    parser.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    parser.add_argument("--sims", type=int, default=512, help="机器人每步的搜索模拟次数（部署设置 512）")
    parser.add_argument("--certfile", default=None, help="HTTPS 证书（局域网其他设备用浏览器摄像头时需要）")
    parser.add_argument("--keyfile", default=None)
    args = parser.parse_args()

    codex = None
    if args.chatgpt_vision:
        if args.host not in ("127.0.0.1", "localhost"):
            parser.error("--chatgpt-vision 只能和本机监听地址一起使用")
        codex = shutil.which("codex")
        if not codex:
            parser.error("找不到 Codex CLI。请安装或从能运行 codex 的终端启动")
    store = Store(args.data_dir)
    kwargs = {k: v for k, v in (("weights", args.weights), ("config", args.config)) if v}
    provider = ModelProvider(device=args.device, sims=args.sims, **kwargs)
    app = App(store, provider, os.path.join(store.dir, "games"), allow_register=args.allow_register)
    if codex:
        app.chatgpt = ChatGPTVision(codex)
    if args.arm_camera is not None:
        from so_arm.face_track import load_config

        cfg = load_config()
        app.arm = ArmCamera(args.arm_camera, args.arm_port or cfg["port"], cfg["baudrate"],
                            args.arm_rotate if args.arm_rotate is not None else cfg["rotate"],
                            args.arm_poses or os.path.join(store.dir, "arm_poses.json"))
    if args.camera is not None:
        app.camera = LocalCamera(args.camera, app.frames, args.camera_period)
        app.camera.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app, secure=bool(args.certfile)))
    scheme = "http"
    if args.certfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.certfile, args.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"璀璨宝石线下对局：{scheme}://{args.host}:{server.server_port}  (Ctrl+C 退出)", flush=True)
    print(f"数据目录：{store.dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app.chatgpt:
            app.chatgpt.close()
        if app.arm:
            app.arm.close()
            if app.arm.worker:
                app.arm.worker.join(timeout=3)
        server.server_close()


if __name__ == "__main__":
    main()
