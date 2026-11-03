"""Local web UI and JSON API, using only Python's standard library."""

from __future__ import annotations

import argparse
import json
import random
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .bots import choose_action
from .engine import Action, CARDS, NOBLES, new_game
from dataclasses import asdict

WEB = Path(__file__).parent / "web"
SCHEMA = "splendor-public-four-player-immediate-no-pass-v1"


class Session:
    def __init__(self, seed=None):
        self.lock = threading.Lock()
        self.version = 0
        self.policy_rng = random.Random()
        self.reset(seed)

    def reset(self, seed=None):
        seed = secrets.randbelow(2**32) if seed is None else seed
        if type(seed) is not int or abs(seed) > 2**53 - 1:
            raise ValueError("种子必须是安全范围内的整数")
        rng = random.Random(seed)
        game = new_game(rng)
        self.seed, self.rng, self.game, self.history, self.events = seed, rng, game, [], []
        self.version += 1

    def perform(self, action):
        event = {"player": self.game.current_player, "turn": self.game.turn,
                 "action": action.to_dict()}
        self.game.step(action, self.rng)
        self.history.append(action.to_dict())
        self.events.append(event)
        self.version += 1

    def restore(self, record):
        if record.get("schema") != SCHEMA:
            raise ValueError("存档规则版本不匹配")
        actions = record.get("actions")
        if not isinstance(actions, list) or len(actions) > 10000:
            raise ValueError("存档动作列表无效或过长")
        # Validate replay in a temporary session; a bad file never destroys a game.
        candidate = Session(record["seed"])
        for value in actions:
            candidate.perform(Action.from_dict(value))
        candidate.game.assert_valid()
        self.seed, self.rng, self.game, self.history = candidate.seed, candidate.rng, candidate.game, candidate.history
        self.events = candidate.events
        self.version += 1

    def record(self):
        return {"schema": SCHEMA, "seed": self.seed, "actions": self.history[:]}

    def view(self):
        return {"state": self.game.observation(), "seed": self.seed, "version": self.version,
                "actions": [a.to_dict() for a in self.game.legal_actions()],
                "history": self.events[-80:], "history_count": len(self.history),
                "cards": {c: asdict(card) for c, card in CARDS.items()},
                "noble_data": {n: asdict(noble) for n, noble in NOBLES.items()}}


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/state":
                with session.lock:
                    self.send_json(session.view())
            elif self.path == "/api/export":
                with session.lock:
                    self.send_json(session.record())
            elif self.path in ("/", "/app.js", "/style.css"):
                name = "index.html" if self.path == "/" else self.path[1:]
                body = (WEB / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[name])
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            try:
                # Local app: require JSON and reject cross-origin form submissions.
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("只接受 JSON 请求")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000:
                    raise ValueError("请求体大小无效")
                data = json.loads(self.rfile.read(length))
                with session.lock:
                    if data.get("version") != session.version:
                        self.send_json({"error": "局面已更新，请刷新后重试"}, 409)
                        return
                    if self.path == "/api/new":
                        session.reset(data.get("seed"))
                    elif self.path == "/api/action":
                        session.perform(Action.from_dict(data["action"]))
                    elif self.path == "/api/bot":
                        action = choose_action(session.game, session.policy_rng, data.get("policy", "heuristic"))
                        session.perform(action)
                    elif self.path == "/api/undo":
                        record = session.record()
                        record["actions"] = record["actions"][:-1]
                        session.restore(record)
                    elif self.path == "/api/import":
                        session.restore(data["record"])
                    else:
                        self.send_json({"error": "Not found"}, 404)
                        return
                    self.send_json(session.view())
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, fmt, *args):
            pass

    return Handler


def simulate(games, seed, max_turns=2000):
    stats = {"score": 0, "blocked": 0, "truncated": 0, "wins": [0.] * 4, "turns": 0}
    for index in range(games):
        rng, policy_rng = random.Random(seed + index), random.Random(seed + index + 10_000)
        game = new_game(rng)
        while game.phase not in ("terminal", "blocked") and game.turn < max_turns:
            game.step(choose_action(game, policy_rng), rng)
            game.assert_valid()
        stats[game.end_reason or "truncated"] += 1
        stats["turns"] += game.turn
        if game.phase == "terminal":
            stats["wins"] = [v + reward for v, reward in zip(stats["wins"], game.rewards())]
    return stats


def main():
    parser = argparse.ArgumentParser(description="四人璀璨宝石模拟器（公开预留、十枚硬上限）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--simulate", type=int, metavar="GAMES")
    args = parser.parse_args()
    if args.simulate is not None:
        if args.simulate <= 0:
            parser.error("--simulate 必须大于零")
        print(json.dumps(simulate(args.simulate, args.seed or 0), ensure_ascii=False, indent=2))
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Session(args.seed)))
    url = f"http://{args.host}:{server.server_port}"
    print(f"璀璨宝石模拟器：{url}  (Ctrl+C 退出)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
