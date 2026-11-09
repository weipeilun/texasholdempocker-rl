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
import json
import mimetypes
import os
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
from .robots import POLICY_LABELS, ModelProvider, make_robot
from .session import OfflineSession
from .simulate import vision_json
from .table import card_text, noble_text, normalize_vision
from .vision import SUGGESTED_MODELS, VisionError, list_models, recognize

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
    def _vision_loop(self):
        last_seq, last_call = -1, 0.
        while True:
            time.sleep(0.5)
            try:
                s = self.session
                if s is None or s.phase == "terminal" or self.owner is None:
                    continue
                settings = self.store.settings(self.owner)
                jpeg, frame_time, seq, _ = self.frames.latest()
                now = time.time()
                interval = float(settings.get("interval") or 5)
                if jpeg is None:
                    self._alert("no_frame", "还没有摄像头画面：打开“摄像头”页面开始推流，或用 --camera 启动", "warn")
                    continue
                self._clear("no_frame")
                if now - frame_time > max(15., 3 * interval):
                    self._alert("stale_frame", f"摄像头画面已经 {int(now - frame_time)} 秒没有更新", "error")
                else:
                    self._clear("stale_frame")
                if not (self.vision_enabled or self.vision_force):
                    continue
                if not settings.get("api_key"):
                    self._alert("vision_config", "还没有配置视觉模型的 API Key（设置页面）", "error")
                    continue
                self._clear("vision_config")
                if not self.vision_force and (seq == last_seq or now - last_call < interval):
                    continue
                self.vision_force, self.vision_busy = False, True
                last_seq, last_call = seq, now
                hint = vision_json(s.game) if (s.game is not None and settings.get("hint_expected")) else None
                try:
                    data, meta = recognize(settings, jpeg, hint)
                    meta["frame_time"] = frame_time
                    table = normalize_vision(data)
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
        jpeg, frame_time, _, _ = self.frames.latest()
        if jpeg is None:
            raise ValueError("还没有摄像头画面")
        settings = self.store.settings(user)
        s = self.session
        hint = vision_json(s.game) if (s is not None and s.game is not None and settings.get("hint_expected")) else None
        data, meta = recognize(settings, jpeg, hint)
        table = normalize_vision(data)
        summary = {
            "market": [[card_text(c) for c in row] for row in table.market] if table.market else None,
            "nobles": [noble_text(n) for n in table.nobles] if table.nobles is not None else None,
            "bank": table.bank, "seats": [vars(x) for x in table.seats], "issues": table.issues,
            "visible": table.visible, "confidence": table.confidence,
        }
        return {"raw": data, "summary": summary, "meta": meta}

    def view(self, user):
        s = self.session
        out = {"user": user, "admin": self.store.is_admin(user), "game": None if s is None else s.view(),
               "owner": self.owner, "vision": {"enabled": self.vision_enabled, "busy": self.vision_busy,
                                               "last": self.vision_last},
               "frame": self.frames.info(), "model": self.provider.describe(),
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
                self.send_json(app.view(user))
            elif path == "/api/cards":
                self.send_json(_cards_payload())
            elif path == "/api/frame.jpg":
                jpeg, _, _, _ = app.frames.latest()
                if jpeg is None:
                    self.send_json({"error": "还没有画面"}, 404)
                else:
                    self.send_bytes(jpeg, "image/jpeg")
            elif path == "/api/settings":
                self.send_json(app.store.public_settings(user))
            elif path == "/api/models":
                s = app.store.settings(user)
                try:
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

        def route(self, path, data, user):
            s = app.session
            if path == "/api/game/new":
                app.new_game(user, data.get("seats") or [])
                return app.view(user)
            if path == "/api/settings":
                app.store.save_settings(user, data)
                return app.store.public_settings(user)
            if path == "/api/password":
                app.store.change_password(user, str(data.get("old", "")), str(data.get("new", "")))
                return {"ok": True}
            if path == "/api/vision/test":
                return app.test_vision(user)
            if path == "/api/vision/now":
                app.vision_force = True
                return app.view(user)
            if path == "/api/vision/toggle":
                app.vision_enabled = bool(data.get("enabled"))
                return app.view(user)
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
            return app.view(user)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="璀璨宝石线下对局：摄像头 + 视觉模型 + 机器人座位")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument("--data-dir", default="~/.splendor_offline", help="账号、设置与对局记录（私有目录）")
    parser.add_argument("--allow-register", action="store_true", help="允许在登录页注册新账号（默认只有第一个账号可自助创建）")
    parser.add_argument("--camera", default=None, help="本机摄像头：设备编号（如 0）或视频流地址（rtsp://…、http://…）")
    parser.add_argument("--camera-period", type=float, default=1.0, help="本机摄像头取帧间隔（秒）")
    parser.add_argument("--weights", default=None, help="机器人模型权重（默认 models/splendor_final/weights.pth）")
    parser.add_argument("--config", default=None, help="模型配置（默认 models/splendor_final/config.yml）")
    parser.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    parser.add_argument("--sims", type=int, default=512, help="机器人每步的搜索模拟次数（部署设置 512）")
    parser.add_argument("--certfile", default=None, help="HTTPS 证书（局域网其他设备用浏览器摄像头时需要）")
    parser.add_argument("--keyfile", default=None)
    args = parser.parse_args()

    store = Store(args.data_dir)
    kwargs = {k: v for k, v in (("weights", args.weights), ("config", args.config)) if v}
    provider = ModelProvider(device=args.device, sims=args.sims, **kwargs)
    app = App(store, provider, os.path.join(store.dir, "games"), allow_register=args.allow_register)
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
        server.server_close()


if __name__ == "__main__":
    main()
