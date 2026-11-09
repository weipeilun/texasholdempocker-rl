"""Offline (physical table) play: a simulated table and a perfect simulated camera drive the session."""
import random
import unittest

from splendor.bots import choose_action
from splendor.engine import Action, new_game
from splendor.offline.robots import RuleRobot
from splendor.offline.session import OfflineSession
from splendor.offline.simulate import vision_json
from splendor.offline.table import UNKNOWN, normalize_vision, table_of, matches


def read(game):
    return normalize_vision(vision_json(game))


def make_session(kinds, seed=0):
    seats = [{"name": f"P{i + 1}", "kind": k, "policy": "heuristic"} for i, k in enumerate(kinds)]
    robots = {i: RuleRobot("lookahead", seed + i) for i, k in enumerate(kinds) if k == "robot"}
    return OfflineSession(seats, robots, sync_robots=True, stable=2, unexplained=3, clock=lambda: 0.)


def show(session, game, times=2):
    for _ in range(times):
        session.on_reading(read(game))


def play_human_turn(truth, rng, policy_rng, partial_rng=None, session=None):
    """one full turn of the current player on the physical table, by a rule bot; optionally shows half-done states"""
    player, turn = truth.current_player, truth.turn
    while truth.phase not in ("terminal",) and truth.current_player == player and truth.turn == turn:
        a = choose_action(truth, policy_rng, "heuristic")
        truth.step(a, None)
        if truth.phase == "chance":
            if partial_rng is not None and session is not None and partial_rng.random() < .5:
                show(session, truth)          # bought, slot not refilled yet
            truth.sample_chance(rng)
        if partial_rng is not None and session is not None and truth.phase == "discard" and partial_rng.random() < .5:
            show(session, truth)              # took tokens, not returned yet


def execute_instruction(session, truth, rng):
    ins = session.instruction
    assert ins is not None, "robot has no instruction"
    for d in ins["actions"]:
        a = Action.from_dict(d) if isinstance(d, dict) else d
        truth.step(a, None)
        if truth.phase == "chance":
            truth.sample_chance(rng)
            break
    if ins.get("needs_confirm"):
        session.confirm_robot()


class TableTest(unittest.TestCase):
    def test_vision_round_trip(self):
        rng = random.Random(1)
        g = new_game(rng)
        for _ in range(30):
            g.step(choose_action(g, rng, "heuristic"), rng)
        self.assertTrue(matches(table_of(g), read(g)))
        self.assertEqual(read(g).key(), table_of(g).key())

    def test_near_card_and_unknown(self):
        g = new_game(random.Random(2))
        data = vision_json(g)
        data["market"]["1"][0]["cost"]["white"] += 1          # slightly misread cost -> nearest card
        data["market"]["2"][1] = "?"                          # unreadable slot
        t = normalize_vision(data)
        self.assertEqual(t.market[0][0], g.market[0][0])
        self.assertEqual(t.market[1][1], UNKNOWN)
        self.assertTrue(t.issues)


class SessionTest(unittest.TestCase):
    def run_game(self, kinds, seed, partial=False):
        rng, policy_rng, partial_rng = random.Random(seed), random.Random(seed + 1), random.Random(seed + 2)
        truth = new_game(rng)
        s = make_session(kinds, seed)
        show(s, truth)
        self.assertEqual(s.phase, "setup")
        s.begin_from_reading()
        for _ in range(400):
            if truth.phase == "terminal":
                break
            if s.is_robot(truth.current_player):
                execute_instruction(s, truth, rng)
            else:
                play_human_turn(truth, rng, policy_rng, partial_rng if partial else None, s)
            show(s, truth)
            self.assertEqual(s.game.key(), truth.key(), [a["text"] for a in s.alerts.values()] + s.hints())
            self.assertFalse(s.alerts, s.alerts)
        self.assertEqual(truth.phase, "terminal")
        self.assertEqual(s.phase, "terminal")
        self.assertEqual(s.game.winners, truth.winners)
        return s

    def test_all_humans(self):
        for seed in range(4):
            self.run_game(["human"] * 4, 100 + seed)

    def test_humans_and_robots(self):
        for seed in range(4):
            self.run_game(["human", "robot", "human", "robot"], 200 + seed)

    def test_all_robots_with_half_done_states(self):
        for seed in range(3):
            self.run_game(["robot"] * 4, 300 + seed, partial=True)

    def test_half_done_human_turns(self):
        for seed in range(4):
            self.run_game(["human", "human", "robot", "human"], 400 + seed, partial=True)

    def test_illegal_move_alert(self):
        rng = random.Random(5)
        truth = new_game(rng)
        s = make_session(["human"] * 4)
        show(s, truth)
        s.begin_from_reading()
        bad = truth.clone()
        p = bad.players[0]
        for c in range(4):            # four tokens at once: not a legal move
            p.tokens[c] += 1
            bad.bank[c] -= 1
        show(s, bad, times=3)
        self.assertIn("unexplained", s.alerts)
        self.assertTrue(any("最多拿 3 枚" in d for d in s.alerts["unexplained"]["details"]))
        self.assertEqual(s.game.key(), truth.key())
        show(s, truth)                # table restored: alert gone
        self.assertNotIn("unexplained", s.alerts)

    def test_out_of_turn_and_reading_rules(self):
        rng = random.Random(6)
        truth = new_game(rng)
        s = make_session(["human"] * 4)
        show(s, truth)
        s.begin_from_reading()
        bad = truth.clone()
        bad.players[2].tokens[0] += 1   # a token appears from nowhere: totals broken
        show(s, bad)
        self.assertIn("reading_rules", s.alerts)
        bad = truth.clone()
        bad.players[2].tokens[0] += 1
        bad.bank[0] -= 1                # player 3 takes a token while player 1 is to act
        show(s, bad, times=3)
        self.assertIn("unexplained", s.alerts)
        self.assertTrue(any("非当前玩家" in d for d in s.alerts["unexplained"]["details"]))

    def test_camera_not_visible(self):
        truth = new_game(random.Random(7))
        s = make_session(["human"] * 4)
        data = vision_json(truth)
        data["visible"] = False
        data["issues"] = ["右侧玩家区域在画面外"]
        s.on_reading(normalize_vision(data))
        self.assertIn("camera", s.alerts)
        self.assertEqual(s.alerts["camera"]["level"], "error")

    def test_robot_proxy_mistake(self):
        rng = random.Random(8)
        truth = new_game(rng)
        s = make_session(["robot", "human", "human", "human"])
        show(s, truth)
        s.begin_from_reading()
        self.assertIsNotNone(s.instruction)
        instructed = s.instruction["actions"][0]
        wrong = next(a for a in truth.legal_actions() if a.kind == "take" and a != instructed)
        bad = truth.clone()
        bad.step(wrong, rng)
        show(s, bad)
        self.assertIn("mismatch", s.alerts)
        self.assertEqual(s.game.key(), truth.key())

    def test_undo_and_manual(self):
        rng = random.Random(9)
        truth = new_game(rng)
        s = make_session(["human"] * 4)
        show(s, truth)
        s.begin_from_reading()
        a = next(x["action"] for x in s.legal_view() if x["action"]["kind"] == "take")
        s.manual_action(a)
        self.assertEqual(s.game.current_player, 1)
        s.undo()
        self.assertEqual(s.game.key(), truth.key())
        buy_or_reserve = next(x["action"] for x in s.legal_view() if x["action"]["kind"] == "reserve")
        s.manual_action(buy_or_reserve)
        self.assertEqual(s.game.phase, "chance")
        card = s.view()["reveal_pool"][0]["id"]
        s.manual_reveal(card)
        self.assertEqual(s.game.current_player, 1)


if __name__ == "__main__":
    unittest.main()
