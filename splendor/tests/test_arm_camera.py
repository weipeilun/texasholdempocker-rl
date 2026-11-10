"""Simulate clipped and clear readings and a servo following taught paths."""
from __future__ import annotations

import random
import time
import unittest
from types import SimpleNamespace

from so_arm.table_scan import ARRIVAL_TOLERANCE, MAX_SWING, STEP, TableMotion, auto_poses
from splendor.engine import new_game
from splendor.offline.arm_camera import REGIONS, ArmCamera, assess_table
from splendor.offline.scan_memory import ScanMemory
from splendor.offline.simulate import vision_json
from splendor.offline.table import normalize_vision
from splendor.tests.test_offline import make_session


class FakeArm:
    ids = [1, 2, 3, 4]

    def __init__(self, positions):
        self.positions = positions.copy()
        self.bus = SimpleNamespace(last_error={})
        self.commands = []

    def read_pos(self, sid):
        return self.positions[sid]

    def read_all(self):
        return self.positions.copy()

    def read_limits(self, sid):
        return 100, 4000

    def hold_here(self):
        pass

    def set_motion_profile(self, *args, **kwargs):
        pass

    def set_goals(self, goals):
        for sid, goal in goals.items():
            assert abs(goal - self.positions[sid]) <= STEP
            self.positions[sid] = goal
        self.commands.append(goals)


class ArmCameraTest(unittest.TestCase):
    def setUp(self):
        home = {1: 2000, 2: 2000, 3: 2000, 4: 2000}
        self.servo = FakeArm(home)
        self.camera = ArmCamera(1, None, 1_000_000, 0, "")
        self.camera.motion = TableMotion(self.servo, home, [(1, 2060), (4, 1940)])
        self.camera.trial = (1, home[1])
        self.camera.running = self.camera.auto = True
        self.game = new_game(random.Random(1))
        self.data = vision_json(self.game)

    def frame(self, coverage, complete):
        camera = self.camera
        camera.jpeg, camera.frame_time = b"jpeg", time.time()
        camera.seq += 1
        return camera.accept(camera.revision, coverage, complete, drive=True)

    def move(self):
        camera = self.camera
        for _ in range(100):
            camera._update_motion()
            if not camera.motion.moving:
                break
            self.assertIsNone(camera.sample())
        self.assertFalse(camera.motion.moving)
        self.assertIsNone(camera.sample())
        camera.ready_at = 0

    def test_reframe_and_hold_on_complete_readings(self):
        camera = self.camera
        clear = dict.fromkeys(REGIONS, "clear")
        unreadable = dict(clear, seat2="unreadable")
        revision = camera.revision
        self.frame(unreadable, False)
        self.assertFalse(camera.retry)
        self.frame(unreadable, False)
        self.assertTrue(camera.retry)
        self.assertFalse(camera.accept(revision, clear, True, drive=True))
        self.move()
        self.assertAlmostEqual(self.servo.positions[1], 2060, delta=ARRIVAL_TOLERANCE)
        self.frame(dict(clear, seat2="cropped"), False)
        self.move()
        self.assertAlmostEqual(self.servo.positions[1], 2000, delta=ARRIVAL_TOLERANCE)
        self.assertAlmostEqual(self.servo.positions[4], 1940, delta=ARRIVAL_TOLERANCE)
        self.frame(clear, True)
        self.frame(clear, True)
        self.assertTrue(camera.auto)
        self.assertFalse(camera.searching)
        self.assertIn("已看全", camera.message)

    def test_exhaustion_and_stop(self):
        camera = self.camera
        clipped = dict.fromkeys(REGIONS, "cropped")
        for _ in range(2):
            self.frame(clipped, False)
        camera._update_motion()
        self.assertTrue(camera.motion.moving)
        camera.set_auto(False)
        camera._update_motion()
        self.assertFalse(camera.motion.moving)
        camera.auto = True
        camera.ready_at = 0
        # Best effort returns to the best earlier view after exhausting candidates.
        self.frame(clipped, False)
        self.frame(clipped, False)
        self.move()
        self.frame(dict(clipped, market="clear"), False)
        self.move()
        self.frame(clipped, False)
        self.move()
        self.assertFalse(camera.auto)
        self.assertIn("已保留", camera.message)
        self.assertAlmostEqual(self.servo.positions[1], 2060, delta=ARRIVAL_TOLERANCE)
        self.assertAlmostEqual(self.servo.positions[4], 2000, delta=ARRIVAL_TOLERANCE)

    def test_automatic_candidates_stay_within_limits(self):
        self.servo.positions[4] = 130
        home, views = auto_poses(self.servo)
        self.assertTrue(views)
        self.assertTrue(all(sid in (1, 4) and 100 < goal < 4000 for sid, goal in views))
        self.assertTrue(all(abs(goal - home[sid]) < MAX_SWING for sid, goal in views))
        self.assertFalse(any(sid == 4 and goal < home[4] for sid, goal in views))

    def test_vision_suggestion_selects_only_calibrated_unvisited_pose(self):
        camera = self.camera
        coverage = dict.fromkeys(REGIONS, "unreadable")
        camera.jpeg, camera.frame_time = b"jpeg", time.time()
        self.assertEqual([v["id"] for v in camera.plan({"seat2": ["筹码"]})["available_actions"]],
                         ["view_1", "view_2"])
        camera.accept(camera.revision, coverage, False, drive=True, suggestion="move joint 2 by 90")
        self.assertFalse(camera.retry)
        camera.accept(camera.revision, coverage, False, drive=True, suggestion="view_2")
        self.move()
        self.assertAlmostEqual(self.servo.positions[4], 1940, delta=ARRIVAL_TOLERANCE)
        self.assertNotIn("view_2", [v["id"] for v in camera.plan({"market": ["牌"]})["available_actions"]])

    def test_partial_views_keep_confirmed_fields_but_conflicts_start_new_scan(self):
        memory = ScanMemory()
        memory.prepare(("turn", 1), 10)
        coverage = dict.fromkeys(REGIONS, "clear")
        first = normalize_vision(self.data, strict=True)
        expected_bank = first.bank
        first.market[0][0] = "?"
        first.seats[2].tokens = None
        merged, _ = memory.merge(first, coverage, None, stamp=10)
        self.assertFalse(merged.visible)
        self.assertIsNone(merged.bank)
        second = normalize_vision(self.data, strict=True)
        second.market[0][1:] = ["?"] * 3
        second.market[1:] = [["?"] * 4, ["?"] * 4]
        second.seats[0].tokens = None
        merged, _ = memory.merge(second, dict(coverage, seat1="cropped"), None, stamp=18)
        self.assertTrue(merged.visible)
        self.assertEqual(merged.bank, expected_bank)
        self.assertEqual(merged.seats[0].tokens, [0] * 6)
        third = normalize_vision(self.data, strict=True)
        third.market = None
        third.seats[0].tokens[0] = 1
        merged, _ = memory.merge(third, coverage, None, stamp=24)
        self.assertFalse(merged.visible)
        self.assertTrue(any("已重新采集" in issue for issue in merged.issues))
        memory.prepare(("turn", 2), 30)
        self.assertIn("market", memory.missing())

    def test_unreadable_fields_cannot_advance_or_open_game(self):
        session = make_session(["human"] * 4)
        self.data["coverage"] = dict.fromkeys(REGIONS, "clear")
        _, complete = assess_table(self.data, normalize_vision(self.data))
        self.assertTrue(complete)
        self.data["players"][2]["tokens"] = None
        table = normalize_vision(self.data)
        _, complete = assess_table(self.data, table)
        self.assertFalse(complete)
        session.on_reading(table, {"usable": complete})
        with self.assertRaises(ValueError):
            session.begin_from_reading()
