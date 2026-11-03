import random
import unittest
from collections import Counter

from splendor.bots import choose_action
from splendor.engine import Action, CARDS, NOBLES, new_game
from splendor.server import Session


def move_card(game, card_id, destination):
    """Build scenarios without duplicating or deleting any cards."""
    for row in game.market:
        if card_id in row:
            row[row.index(card_id)] = None
    for pool in game.remaining:
        if card_id in pool:
            pool.remove(card_id)
    for p in game.players:
        for zone in (p.purchased, p.reserved):
            if card_id in zone:
                zone.remove(card_id)
    destination.append(card_id)


def set_tokens(game, player, tokens):
    for color, (old, new) in enumerate(zip(game.players[player].tokens, tokens)):
        game.bank[color] += old - new
    game.players[player].tokens = list(tokens)


def give_bonuses(game, counts, player=0):
    for color, count in enumerate(counts):
        for card in [c for c in CARDS.values() if c.bonus == color and c.points == 0][:count]:
            move_card(game, card.id, game.players[player].purchased)


def give_points(game, points, player):
    for value in points:
        card_id = next(c.id for c in CARDS.values() if c.points == value
                       and all(c.id not in p.purchased for p in game.players))
        move_card(game, card_id, game.players[player].purchased)


class RulesTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)
        self.game = new_game(self.rng)

    def test_inventory_and_exact_opening_actions(self):
        g = self.game
        self.assertEqual(Counter(c.level for c in CARDS.values()), {1: 40, 2: 30, 3: 20})
        self.assertEqual(len(NOBLES), 10)
        self.assertEqual([len(pool) for pool in g.remaining], [36, 26, 16])
        self.assertEqual(g.bank, [7, 7, 7, 7, 7, 5])
        self.assertEqual(Counter(a.kind for a in g.legal_actions()), {"take": 15, "reserve": 12})
        g.assert_valid()

    def test_photograph_verified_card_cost_regression(self):
        # The common bouk CSV has a wrong white self-cost for this card.
        self.assertEqual(CARDS["blue-L1-03"].cost, (1, 0, 0, 0, 2))
        self.assertEqual(CARDS["black-L1-05"].cost, (0, 0, 1, 3, 1))

    def test_hard_cap_eight_nine_ten_and_atomic_rejection(self):
        # Legacy rule: discard_over_limit=False.
        g = new_game(self.rng, discard_over_limit=False)
        set_tokens(g, 0, [2, 2, 2, 2, 0, 0])
        actions = g.legal_actions()
        self.assertFalse(any(a.kind == "take" and len(set(a.colors)) == 3 for a in actions))
        self.assertTrue(any(a == Action("take", colors=(4, 4)) for a in actions))
        before = g.key()
        with self.assertRaises(ValueError):
            g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(g.key(), before)
        set_tokens(g, 0, [2, 2, 2, 2, 1, 0])
        self.assertFalse(any(a.kind == "take" for a in g.legal_actions()))
        self.assertTrue(any(a.kind == "reserve" for a in g.legal_actions()))
        set_tokens(g, 0, [2, 2, 2, 2, 2, 0])
        self.assertFalse(any(a.kind in ("take", "reserve") for a in g.legal_actions()))
        g.assert_valid()

    def test_scarce_bank_not_capacity_allows_fewer_colors(self):
        g = self.game
        # Only green/red remain; take both rather than voluntarily choosing one.
        g.bank[:5] = [0, 0, 2, 1, 0]
        takes = [a for a in g.legal_actions() if a.kind == "take"]
        self.assertEqual(takes, [Action("take", colors=(2, 3))])
        set_tokens(g, 0, [2, 2, 2, 2, 1, 0])
        self.assertFalse(any(a.kind == "take" for a in g.legal_actions()))

    def test_double_requires_four_in_bank(self):
        g = self.game
        set_tokens(g, 0, [4, 0, 0, 0, 0, 0])
        self.assertNotIn(Action("take", colors=(0, 0)), g.legal_actions())
        set_tokens(g, 0, [3, 0, 0, 0, 0, 0])
        self.assertIn(Action("take", colors=(0, 0)), g.legal_actions())

    def test_reserve_is_public_draws_gold_and_refills_only_on_demand(self):
        g = self.game
        action = next(a for a in g.legal_actions() if a.kind == "reserve")
        before = g.key()
        child = g.apply(action)
        self.assertEqual(g.key(), before)
        self.assertEqual(child.phase, "chance")
        self.assertEqual(child.current_player, 0)
        self.assertIn(action.card_id, child.observation()["players"][0]["reserved"])
        self.assertEqual(child.bank[5], 4)
        self.assertEqual(child.players[0].tokens[5], 1)
        outcomes = child.chance_outcomes()
        self.assertEqual(len(outcomes), 36)
        self.assertAlmostEqual(sum(p for _, p in outcomes), 1)
        self.assertEqual(child.legal_actions(), [])
        child.resolve_chance(outcomes[-1][0])
        self.assertEqual(child.current_player, 1)
        self.assertEqual(len(child.remaining[0]), 35)
        child.assert_valid()

    def test_each_chance_outcome_has_its_own_successor_and_clone(self):
        action = next(a for a in self.game.legal_actions() if a.kind == "reserve")
        chance = self.game.apply(action)
        keys = set()
        for card_id, _ in chance.chance_outcomes():
            child = chance.clone()
            child.resolve_chance(card_id)
            child.assert_valid()
            keys.add(child.key())
        self.assertEqual(len(keys), 36)
        self.assertEqual(chance.phase, "chance")

    def test_clone_has_no_rng_or_order_to_peek_and_branches_can_diverge(self):
        g = self.game
        action = next(a for a in g.legal_actions() if a.kind == "reserve")
        branches = {g.apply(action, random.Random(seed)).key() for seed in range(20)}
        self.assertGreater(len(branches), 1)
        self.assertFalse(any("rng" in name or "deck" in name for name in vars(g)))
        reversed_pool = g.clone()
        for pool in reversed_pool.remaining:
            pool.reverse()
        self.assertEqual(g.key(), reversed_pool.key())
        self.assertEqual(g.observation(), reversed_pool.observation())

    def test_full_hand_can_reserve_if_bank_has_no_gold(self):
        g = self.game
        set_tokens(g, 0, [2, 2, 2, 2, 2, 0])
        set_tokens(g, 1, [0, 0, 0, 0, 0, 5])
        action = next(a for a in g.legal_actions() if a.kind == "reserve")
        g.step(action, self.rng)
        self.assertEqual(sum(g.players[0].tokens), 10)
        g.assert_valid()

    def test_reserve_limit_and_no_blind_reserve(self):
        g = self.game
        for i in range(3):
            move_card(g, f"white-L1-0{i+1}", g.players[0].reserved)
        self.assertFalse(any(a.kind == "reserve" for a in g.legal_actions()))
        with self.assertRaises(ValueError):
            g.step(Action("reserve", card_id="deck-1"), self.rng)
        g.assert_valid()

    def test_gold_payments_include_voluntary_substitution_and_return_to_bank(self):
        g = self.game
        card_id = "white-L1-01"  # blue x3
        move_card(g, card_id, g.players[0].reserved)
        set_tokens(g, 0, [0, 3, 0, 0, 0, 2])
        payments = g.payments(card_id)
        self.assertEqual({(p[1], p[5]) for p in payments}, {(3, 0), (2, 1), (1, 2)})
        bank_before = g.bank[:]
        pools_before = [len(pool) for pool in g.remaining]
        g.step(Action("buy", card_id=card_id, payment=(0, 1, 0, 0, 0, 2)), self.rng)
        self.assertEqual(g.players[0].tokens, [0, 2, 0, 0, 0, 0])
        self.assertEqual(g.bank[1], bank_before[1] + 1)
        self.assertEqual(g.bank[5], bank_before[5] + 2)
        self.assertNotIn(card_id, g.players[0].reserved)
        self.assertIn(card_id, g.players[0].purchased)
        self.assertEqual([len(pool) for pool in g.remaining], pools_before)
        g.assert_valid()

    def test_discounts_allow_free_purchase_and_are_not_consumed(self):
        g = self.game
        give_bonuses(g, [0, 3, 0, 0, 0])
        move_card(g, "white-L1-01", g.players[0].reserved)
        self.assertEqual(g.payments("white-L1-01"), [(0, 0, 0, 0, 0, 0)])
        g.step(Action("buy", card_id="white-L1-01", payment=(0,)*6), self.rng)
        self.assertEqual(g.players[0].bonuses, [1, 3, 0, 0, 0])
        g.assert_valid()

    def test_multiple_nobles_require_choice_and_exactly_one_award(self):
        g = self.game
        g.nobles = ["N1", "N2", "N3", "N4", "N5"]
        give_bonuses(g, [4, 4, 4, 0, 0])
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(g.phase, "noble")
        self.assertEqual(g.turn, 0)
        self.assertEqual({a.noble_id for a in g.legal_actions()}, {"N1", "N2"})
        g.step(Action("noble", noble_id="N2"), self.rng)
        self.assertEqual(g.players[0].nobles, ["N2"])
        self.assertIn("N1", g.nobles)
        self.assertEqual(g.turn, 1)
        g.assert_valid()

    def test_single_noble_automatic_tokens_cannot_attract(self):
        g = self.game
        g.nobles = ["N1", "N2", "N3", "N4", "N5"]
        set_tokens(g, 0, [4, 4, 0, 0, 0, 0])
        self.assertEqual(g.eligible_nobles(), [])
        set_tokens(g, 0, [0]*6)
        give_bonuses(g, [4, 4, 0, 0, 0])
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(g.players[0].nobles, ["N1"])
        self.assertEqual(g.current_player, 1)
        g.assert_valid()

    def test_exhausted_deck_does_not_create_empty_chance_node(self):
        g = self.game
        for card_id in list(g.remaining[0]):
            move_card(g, card_id, g.players[1].purchased)
        card_id = next(c for c in g.market[0] if c)
        g.step(Action("reserve", card_id=card_id))
        self.assertEqual(g.phase, "action")
        self.assertEqual(g.current_player, 1)
        self.assertIn(None, g.market[0])
        self.assertEqual(g.chance_outcomes(), [])
        g.assert_valid()

    def test_fifteen_points_ends_immediately_without_other_players_acting(self):
        g = self.game
        give_points(g, [5, 5, 5], 0)
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(g.phase, "terminal")
        self.assertEqual(g.turn, 1)
        self.assertEqual(g.current_player, 0)
        self.assertTrue(all(not any(p.tokens) for p in g.players[1:]))
        self.assertEqual(g.winners, [0])
        self.assertEqual(g.rewards(), [1, 0, 0, 0])
        self.assertEqual(g.legal_actions(), [])
        g.assert_valid()

    def test_purchase_crossing_fifteen_does_not_allow_next_player(self):
        g = self.game
        give_points(g, [5, 5, 4], 0)
        card_id = "white-L1-02"
        move_card(g, card_id, g.players[0].reserved)
        set_tokens(g, 0, [0, 0, 4, 0, 0, 0])
        action = Action("buy", card_id=card_id, payment=g.payments(card_id)[0])
        g.step(action, self.rng)
        self.assertEqual(g.phase, "terminal")
        self.assertEqual(g.players[0].points, 15)
        self.assertEqual(g.current_player, 0)
        self.assertEqual(g.turn, 1)
        g.assert_valid()

    def test_noble_can_trigger_immediate_end(self):
        g = self.game
        g.nobles = ["N1", "N2", "N3", "N4", "N5"]
        give_bonuses(g, [4, 4, 0, 0, 0])
        give_points(g, [5, 4, 3], 0)
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(g.phase, "terminal")
        self.assertEqual(g.players[0].points, 15)
        self.assertEqual(g.current_player, 0)
        self.assertEqual(g.turn, 1)
        g.assert_valid()

    def test_fewer_purchased_cards_breaks_score_tie(self):
        g = self.game
        give_points(g, [5, 5, 5], 0)
        give_points(g, [3, 3, 3, 3, 3], 1)
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(g.winners, [0])

    def test_shared_winners_split_four_player_utility(self):
        g = self.game
        give_points(g, [5, 4, 3, 3], 0)
        give_points(g, [5, 4, 3, 3], 1)
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(g.winners, [0, 1])
        self.assertEqual(g.rewards(), [.5, .5, 0, 0])

    def test_no_pass_action_and_blocked_player_stops_without_scoring(self):
        # Legacy hard cap: with the discard rule this player could still take tokens.
        g = new_game(self.rng, discard_over_limit=False)
        self.assertNotIn(Action("pass"), g.legal_actions())
        # Player 2 has 10 blue/green/black tokens, three costly reserves and
        # cannot buy any of the exposed high-cost cards. Player 1 still acts.
        set_tokens(g, 1, [0, 4, 0, 0, 6, 0])
        for card_id in ("white-L3-01", "blue-L3-01", "red-L3-01"):
            move_card(g, card_id, g.players[1].reserved)
        # Put all other market cards in player 4's ownership for this fixture.
        for row in list(g.market):
            for card_id in list(row):
                if card_id:
                    move_card(g, card_id, g.players[3].purchased)
        g.step(Action("take", colors=(0, 2, 3)), self.rng)
        self.assertEqual(g.phase, "blocked")
        self.assertEqual(g.current_player, 1)
        self.assertEqual(g.end_reason, "blocked")
        self.assertEqual(g.legal_actions(), [])
        self.assertEqual(g.winners, [])
        with self.assertRaises(ValueError):
            g.step(Action("pass"), self.rng)
        with self.assertRaises(ValueError):
            g.rewards()
        g.assert_valid()

    def test_seed_replay_including_random_draws_and_undo(self):
        session = Session(101)
        policy_rng = random.Random(77)
        for _ in range(50):
            session.perform(choose_action(session.game, policy_rng))
        replay = Session(999)
        replay.restore(session.record())
        self.assertEqual(session.game.key(), replay.game.key())
        self.assertEqual(session.rng.getstate(), replay.rng.getstate())
        record = session.record()
        record["actions"] = record["actions"][:-1]
        replay.restore(record)
        replay.perform(Action.from_dict(session.history[-1]))
        self.assertEqual(session.game.key(), replay.game.key())
        before = replay.game.key()
        with self.assertRaises(ValueError):
            replay.restore({"schema": "bad"})
        self.assertEqual(replay.game.key(), before)

    def test_random_and_heuristic_games_preserve_all_invariants(self):
        for seed in range(60):
            rng = random.Random(seed)
            policy_rng = random.Random(seed + 1000)
            game = new_game(rng, discard_over_limit=seed < 30)
            policy = "random" if seed % 2 else "heuristic"
            for _ in range(1200):
                if game.phase in ("terminal", "blocked"):
                    break
                game.step(choose_action(game, policy_rng, policy), rng)
                game.assert_valid()
            self.assertIn(game.phase, ("terminal", "blocked"), f"seed {seed}, {policy}")
            if game.phase == "terminal":
                self.assertAlmostEqual(sum(game.rewards()), 1)
            else:
                self.assertEqual(game.winners, [])



class DiscardRuleTest(unittest.TestCase):
    """Official rule (default): exceed 10, then return tokens one at a time before the turn ends."""

    def setUp(self):
        self.rng = random.Random(7)
        self.game = new_game(self.rng)
        self.assertTrue(self.game.discard_over_limit)

    def test_take_above_ten_enters_discard_and_turn_waits(self):
        g = self.game
        set_tokens(g, 0, [2, 2, 2, 2, 1, 0])
        self.assertIn(Action("take", colors=(0, 1, 2)), g.legal_actions())
        self.assertIn(Action("take", colors=(3, 3)), g.legal_actions())
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(sum(g.players[0].tokens), 12)
        self.assertEqual((g.phase, g.current_player, g.turn), ("discard", 0, 0))
        self.assertTrue(all(a.kind == "discard" for a in g.legal_actions()))
        self.assertEqual({a.colors[0] for a in g.legal_actions()}, {0, 1, 2, 3, 4})
        g.assert_valid()
        g.step(Action("discard", colors=(4,)), self.rng)
        self.assertEqual((g.phase, g.current_player, g.turn), ("discard", 0, 0))
        g.step(Action("discard", colors=(0,)), self.rng)
        self.assertEqual(g.players[0].tokens, [2, 3, 3, 2, 0, 0])
        self.assertEqual((g.phase, g.current_player, g.turn), ("action", 1, 1))
        g.assert_valid()

    def test_can_return_tokens_just_taken_and_gold(self):
        g = self.game
        set_tokens(g, 0, [2, 2, 2, 2, 1, 1])
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        actions = g.legal_actions()
        self.assertEqual({a.colors[0] for a in actions}, {0, 1, 2, 3, 4, 5})
        self.assertNotIn(Action("take", colors=(0, 1, 3)), actions)
        bank_gold = g.bank[5]
        g.step(Action("discard", colors=(5,)), self.rng)
        self.assertEqual(g.bank[5], bank_gold + 1)
        g.step(Action("discard", colors=(0,)), self.rng)
        g.step(Action("discard", colors=(1,)), self.rng)
        self.assertEqual(g.players[0].tokens, [2, 2, 3, 2, 1, 0])
        g.assert_valid()

    def test_discard_actions_only_in_discard_phase(self):
        g = self.game
        set_tokens(g, 0, [2, 2, 2, 0, 0, 0])
        with self.assertRaises(ValueError):
            g.step(Action("discard", colors=(0,)), self.rng)

    def test_reserve_with_gold_at_ten_refills_then_discards(self):
        g = self.game
        set_tokens(g, 0, [2, 2, 2, 2, 2, 0])
        action = next(a for a in g.legal_actions() if a.kind == "reserve")
        chance = g.apply(action)
        self.assertEqual(chance.phase, "chance")
        self.assertEqual(sum(chance.players[0].tokens), 11)
        chance.assert_valid()
        chance.resolve_chance(chance.chance_outcomes()[0][0])
        self.assertEqual((chance.phase, chance.current_player), ("discard", 0))
        chance.step(Action("discard", colors=(5,)), self.rng)
        self.assertEqual((chance.phase, chance.current_player, chance.turn), ("action", 1, 1))
        chance.assert_valid()

    def test_nobles_and_fifteen_points_are_settled_after_returning(self):
        g = self.game
        g.nobles = ["N1", "N2", "N3", "N4", "N5"]
        give_bonuses(g, [4, 4, 0, 0, 0])
        give_points(g, [5, 4, 3], 0)
        set_tokens(g, 0, [2, 2, 2, 2, 1, 0])
        g.step(Action("take", colors=(0, 1, 2)), self.rng)
        self.assertEqual(g.phase, "discard")
        self.assertEqual(g.players[0].nobles, [])
        for color in (0, 1):
            g.step(Action("discard", colors=(color,)), self.rng)
        self.assertEqual(g.players[0].nobles, ["N1"])
        self.assertEqual((g.phase, g.current_player, g.turn), ("terminal", 0, 1))
        g.assert_valid()

    def test_discard_rule_blocks_far_less_than_hard_cap(self):
        blocked = {True: 0, False: 0}
        for rule in blocked:
            for seed in range(60):
                rng, policy_rng = random.Random(seed), random.Random(seed + 1)
                game = new_game(rng, discard_over_limit=rule)
                while game.phase not in ("terminal", "blocked") and game.turn < 600:
                    game.step(choose_action(game, policy_rng, "random"), rng)
                blocked[rule] += game.phase == "blocked"
        # Random play: about 55% blocked with the hard cap, about 13% with the discard rule.
        self.assertLess(blocked[True] * 3, blocked[False])
        # What is left needs an empty colour bank, three reserves and no affordable card.


if __name__ == "__main__":
    unittest.main()
