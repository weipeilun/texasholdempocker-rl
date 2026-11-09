"""End to end over HTTP: the offline console, a fake OpenAI-compatible vision endpoint that "sees" a simulated
physical table, robot seats executed by a simulated proxy. Takes about a minute (readings every 2 s, two readings
per change)."""
import io
import json
import random
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from splendor.bots import choose_action
from splendor.engine import Action, new_game
from splendor.offline.accounts import Store
from splendor.offline.robots import ModelProvider
from splendor.offline.server import App, make_handler
from splendor.offline.simulate import vision_json


class FakeVision:
    """OpenAI-compatible /models and /chat/completions; returns the simulated table as the model's JSON"""

    def __init__(self):
        self.truth = None
        self.calls = 0
        self.last_request = None
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, data, status=200):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer sk-test-1234567890":
                    return self._send({"error": {"message": "bad key"}}, 401)
                self._send({"data": [{"id": "fake-vision"}, {"id": "gpt-4o"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.last_request = body
                fake.calls += 1
                content = body["messages"][1]["content"]
                assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
                if "temperature" in body and body["model"] == "no-temperature":
                    return self._send({"error": {"message": "Unsupported parameter: 'temperature'"}}, 400)
                self._send({"model": body["model"], "usage": {"total_tokens": 1},
                            "choices": [{"message": {"content": "```json\n" + json.dumps(vision_json(fake.truth)) + "\n```"}}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"


def jpeg():
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (20, 80, 40)).save(out, format="JPEG")
    return out.getvalue()


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeVision()
        store = Store(tempfile.mkdtemp())
        self.app = App(store, ModelProvider(device="cpu", sims=16), tempfile.mkdtemp())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.http = requests.Session()

    def post(self, path, data=None, ok=True):
        r = self.http.post(self.base + path, json=data or {})
        if ok:
            self.assertEqual(r.status_code, 200, r.text)
        return r

    def state(self):
        return self.http.get(self.base + "/api/state").json()

    def frame(self):
        r = self.http.post(self.base + "/api/frame", data=jpeg(), headers={"Content-Type": "image/jpeg"})
        self.assertEqual(r.status_code, 200, r.text)

    def wait(self, predicate, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            s = self.state()
            if predicate(s):
                return s
            self.frame()
            time.sleep(0.5)
        self.fail(f"timeout; alerts={s['game'] and s['game']['alerts']} hints={s['game'] and s['game']['hints']}")

    def test_end_to_end(self):
        # pages need a login; the first account is created on the login page
        self.assertEqual(self.http.get(self.base + "/", allow_redirects=False).status_code, 302)
        self.assertEqual(self.http.get(self.base + "/api/state").status_code, 401)
        self.assertTrue(self.http.get(self.base + "/api/me").json()["can_register"])
        self.post("/api/register", {"username": "admin", "password": "secret-pass"})
        self.assertEqual(self.http.get(self.base + "/").status_code, 200)
        other = requests.Session()
        self.assertEqual(other.post(self.base + "/api/register", json={"username": "x", "password": "12345678"}).status_code, 400)
        # cross-site form posts are refused
        self.assertEqual(self.http.post(self.base + "/api/game/undo", data="a=1",
                                        headers={"Content-Type": "application/x-www-form-urlencoded"}).status_code, 400)
        # settings: the key is stored but never returned
        s = self.post("/api/settings", {"base_url": self.fake.url, "api_key": "sk-test-1234567890",
                                        "model": "fake-vision", "interval": 2}).json()
        self.assertNotIn("api_key", s)
        self.assertTrue(s["api_key_set"])
        self.assertIn("fake-vision", self.http.get(self.base + "/api/models").json()["models"])
        # a new game: two humans, two rule-bot robots
        rng, policy = random.Random(3), random.Random(4)
        self.fake.truth = new_game(rng)
        seats = [{"name": "甲", "kind": "human"}, {"name": "机器人乙", "kind": "robot", "policy": "lookahead"},
                 {"name": "丙", "kind": "human"}, {"name": "机器人丁", "kind": "robot", "policy": "heuristic"}]
        self.post("/api/game/new", {"seats": seats})
        self.frame()
        self.post("/api/vision/test")          # one-off reading of the current frame
        self.wait(lambda s: s["game"]["reading"] is not None and "setup" not in {a["key"] for a in s["game"]["alerts"]})
        self.post("/api/game/begin")
        truth = self.fake.truth
        for _ in range(8):
            if truth.phase == "terminal":
                break
            player, turn = truth.current_player, truth.turn
            st = self.state()["game"]
            if seats[player]["kind"] == "robot":
                st = self.wait(lambda s: s["game"]["instruction"] is not None)
                for d in st["game"]["instruction"]["actions"]:
                    truth.step(Action.from_dict(d), None)
                    if truth.phase == "chance":
                        truth.sample_chance(rng)
                        break
                if st["game"]["instruction"].get("needs_confirm"):
                    self.post("/api/game/confirm-robot")
            else:
                while truth.phase != "terminal" and truth.current_player == player and truth.turn == turn:
                    truth.step(choose_action(truth, policy, "heuristic"), rng)
            self.wait(lambda s: s["game"]["state"]["turn"] == truth.turn
                      and s["game"]["state"]["current_player"] == truth.current_player)
            g = self.state()["game"]
            self.assertEqual(g["state"]["bank"], truth.bank)
            self.assertEqual([p["tokens"] for p in g["state"]["players"]], [p.tokens for p in truth.players])
            self.assertFalse([a for a in g["alerts"] if a["level"] == "error"], g["alerts"])
        g = self.state()["game"]
        self.assertTrue(any(h["kind"] == "instruction" for h in g["history"]))
        self.assertTrue(any(h["kind"] == "action" for h in g["history"]))
        self.assertGreater(self.fake.calls, 8)
        self.assertEqual(self.http.get(self.base + "/screen").status_code, 200)
        self.assertEqual(self.http.get(self.base + "/static/index.js").status_code, 200)
        self.assertEqual(self.http.get(self.base + "/static/../server.py").status_code, 404)
        # an illegal move on the table: alert, state unchanged
        before = self.state()["game"]["state"]
        bad = truth.clone()
        for c in range(4):
            bad.players[bad.current_player].tokens[c] += 1
            bad.bank[c] -= 1
        self.fake.truth = bad
        s = self.wait(lambda s: any(a["key"] == "unexplained" for a in s["game"]["alerts"]), timeout=40)
        self.assertEqual(s["game"]["state"]["bank"], before["bank"])
        self.fake.truth = truth
        self.wait(lambda s: not any(a["key"] == "unexplained" for a in s["game"]["alerts"]), timeout=40)
        # an endpoint that rejects temperature: retried without it
        self.post("/api/settings", {"model": "no-temperature"})
        r = self.post("/api/vision/test").json()
        self.assertIn("temperature", r["meta"]["retried_without"])


if __name__ == "__main__":
    unittest.main()
