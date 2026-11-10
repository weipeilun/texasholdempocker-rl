"""Accounts of the offline console: users with salted PBKDF2 password hashes, login sessions (in memory), and each
user's vision settings (API base URL / relay, API key, model, ...). Files live in a private data directory
(default ~/.splendor_offline, mode 0700; files 0600). The API key never leaves the server: the pages only see a
masked hint."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time

from .vision import DEFAULTS

SESSION_SECONDS = 7 * 24 * 3600
NAME_RE = re.compile(r"^[A-Za-z0-9_.@\-一-鿿]{2,32}$")
EDITABLE = set(DEFAULTS)


def _write_private(path, data):
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _hash(password, salt, rounds=200_000):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), rounds).hex()


class Store:
    def __init__(self, data_dir):
        self.dir = os.path.expanduser(data_dir)
        os.makedirs(os.path.join(self.dir, "settings"), mode=0o700, exist_ok=True)
        os.chmod(self.dir, 0o700)
        self.users_path = os.path.join(self.dir, "users.json")
        self.lock = threading.Lock()
        self.sessions = {}

    # ----- users
    def _users(self):
        if not os.path.exists(self.users_path):
            return {}
        with open(self.users_path, encoding="utf-8") as f:
            return json.load(f)

    def has_users(self):
        return bool(self._users())

    def create_user(self, name, password, admin=False):
        if not NAME_RE.match(name or ""):
            raise ValueError("用户名为 2–32 个字母、数字、汉字或 _ . @ -")
        if len(password or "") < 8:
            raise ValueError("密码至少 8 位")
        with self.lock:
            users = self._users()
            if name in users:
                raise ValueError("用户名已存在")
            salt = secrets.token_hex(16)
            users[name] = {"salt": salt, "hash": _hash(password, salt), "admin": bool(admin), "created": time.time()}
            _write_private(self.users_path, users)

    def verify(self, name, password):
        user = self._users().get(name or "")
        if not user:
            _hash(password or "", "00" * 16)       # same time whether the user exists or not
            return False
        return hmac.compare_digest(user["hash"], _hash(password or "", user["salt"]))

    def change_password(self, name, old, new):
        if not self.verify(name, old):
            raise ValueError("原密码不正确")
        if len(new or "") < 8:
            raise ValueError("密码至少 8 位")
        with self.lock:
            users = self._users()
            salt = secrets.token_hex(16)
            users[name].update(salt=salt, hash=_hash(new, salt))
            _write_private(self.users_path, users)

    def is_admin(self, name):
        return bool(self._users().get(name, {}).get("admin"))

    # ----- sessions
    def login(self, name, password):
        if not self.verify(name, password):
            raise ValueError("用户名或密码不正确")
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.sessions[token] = (name, time.time() + SESSION_SECONDS)
        return token

    def user_of(self, token):
        with self.lock:
            entry = self.sessions.get(token or "")
            if not entry:
                return None
            if entry[1] < time.time():
                self.sessions.pop(token, None)
                return None
            return entry[0]

    def logout(self, token):
        with self.lock:
            self.sessions.pop(token or "", None)

    # ----- settings
    def _settings_path(self, name):
        return os.path.join(self.dir, "settings", hashlib.sha256(name.encode()).hexdigest()[:24] + ".json")

    def settings(self, name):
        path = self._settings_path(name)
        stored = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                stored = json.load(f)
        return dict(DEFAULTS, **stored)

    def save_settings(self, name, updates):
        current = self.settings(name)
        for key, value in updates.items():
            if key not in EDITABLE:
                continue
            if key == "api_key" and not value:
                continue                    # an empty field keeps the stored key (use clear_api_key)
            current[key] = value
        if current["vision_provider"] not in ("api", "chatgpt"):
            raise ValueError("请选择有效的视觉模型接入方式")
        if updates.get("clear_api_key"):
            current["api_key"] = ""
        for key, lo, hi in (("interval", 2, 120), ("image_max_side", 512, 4096), ("jpeg_quality", 50, 100),
                            ("timeout", 10, 300)):
            try:
                current[key] = min(hi, max(lo, int(current[key])))
            except (TypeError, ValueError):
                current[key] = DEFAULTS[key]
        if not isinstance(current.get("layout"), list) or len(current["layout"]) != 4:
            current["layout"] = DEFAULTS["layout"]
        current["base_url"] = (current.get("base_url") or DEFAULTS["base_url"]).strip()
        _write_private(self._settings_path(name), current)
        return current

    def public_settings(self, name):
        s = self.settings(name)
        key = s.pop("api_key", "")
        s["api_key_set"] = bool(key)
        s["api_key_hint"] = (key[:3] + "…" + key[-4:]) if len(key) > 10 else ("已设置" if key else "")
        return s
