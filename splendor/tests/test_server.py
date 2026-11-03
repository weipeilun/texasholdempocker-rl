import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from splendor.engine import Action
from splendor.server import SCHEMA, Session, make_handler


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = Session(42)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.session))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.session.reset(42)

    def request(self, method, path, data=None, content_type="application/json"):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request(method, path, body=json.dumps(data) if data is not None else None,
                           headers={"Content-Type": content_type} if data is not None else {})
        response = connection.getresponse()
        body = response.read()
        result = json.loads(body) if response.getheader("Content-Type").startswith("application/json") else body.decode()
        connection.close()
        return response.status, result

    def post(self, path, data=None):
        return self.request("POST", path, {"version": self.session.version, **(data or {})})

    def test_static_ui_and_initial_state(self):
        for path, fragment in [("/", "璀璨宝石"), ("/app.js", "renderDecision"), ("/style.css", ":root")]:
            status, body = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn(fragment, body)
        status, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["actions"]), 27)
        self.assertEqual(len(body["state"]["players"]), 4)
        self.assertNotIn("seed", body["state"])
        self.assertEqual(self.request("GET", "/missing")[0], 404)

    def test_take_reserve_export_import_and_undo(self):
        initial = self.session.game.key()
        status, body = self.post("/api/action", {"action": Action("take", colors=(0, 1, 2)).to_dict()})
        self.assertEqual(status, 200)
        self.assertEqual(body["state"]["current_player"], 1)
        self.assertEqual(body["history"][-1]["player"], 0)
        action = next(a for a in self.session.game.legal_actions() if a.kind == "reserve")
        self.assertEqual(self.post("/api/action", {"action": action.to_dict()})[0], 200)
        self.assertEqual(self.session.game.phase, "action")
        after = self.session.game.key()
        status, record = self.request("GET", "/api/export")
        self.assertEqual(status, 200)
        self.assertEqual(record["schema"], SCHEMA)
        self.assertEqual(self.post("/api/new", {"seed": 999})[0], 200)
        self.assertEqual(self.post("/api/import", {"record": record})[0], 200)
        self.assertEqual(self.session.game.key(), after)
        self.post("/api/undo")
        self.post("/api/undo")
        self.assertEqual(self.session.game.key(), initial)

    def test_stale_write_and_invalid_action_are_rejected_without_mutation(self):
        version = self.session.version
        self.assertEqual(self.post("/api/bot")[0], 200)
        before = self.session.game.key()
        status, _ = self.request("POST", "/api/bot", {"version": version})
        self.assertEqual(status, 409)
        self.assertEqual(self.post("/api/action", {"action": Action("pass").to_dict()})[0], 400)
        self.assertEqual(self.session.game.key(), before)

    def test_bad_import_seed_and_non_json_do_not_destroy_game(self):
        before = self.session.game.key()
        record = self.session.record()
        record["actions"] = [Action("pass").to_dict()]
        self.assertEqual(self.post("/api/import", {"record": record})[0], 400)
        for seed in (True, "bad", 1.2, 2**54):
            self.assertEqual(self.post("/api/new", {"seed": seed})[0], 400)
        self.assertEqual(self.request("POST", "/api/new", {}, "text/plain")[0], 400)
        self.assertEqual(self.session.game.key(), before)


if __name__ == "__main__":
    unittest.main()
