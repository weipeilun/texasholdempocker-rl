"""Rule bots (splendor/bots.py): legal moves, complete games, the obvious tactical rules of 'lookahead'."""
import random
import unittest

from splendor import bots
from splendor.engine import CARDS, Player, new_game


def play(policies, seed, max_turns=600):
    rng = random.Random(seed)
    game = new_game(random.Random(seed + 1))
    bot_rngs = [random.Random(seed * 10 + s) for s in range(4)]
    while game.phase != 'terminal' and game.turn < max_turns:
        action = bots.choose_action(game, bot_rngs[game.current_player], policies[game.current_player])
        game.step(action, rng)      # validated: raises on an illegal action
    return game


class BotTest(unittest.TestCase):
    def test_every_policy_plays_legal_complete_games(self):
        for seed in range(3):
            game = play(['lookahead', 'heuristic', 'random', 'lookahead'], seed)
            self.assertEqual(game.phase, 'terminal')
            self.assertIn(game.end_reason, ('score', 'draw'))

    def test_unknown_policy(self):
        with self.assertRaises(ValueError):
            bots.choose_action(new_game(random.Random(0)), random.Random(0), 'nobody')

    def test_lookahead_wins_when_it_can(self):
        game = new_game(random.Random(3))
        card_id = next(c for c in game.market[2] if c and CARDS[c].points >= 3)
        card = CARDS[card_id]
        me = game.players[0]
        # 15 - card points from cards of the first level, and exactly the tokens the card costs
        filler = [c for c in CARDS if CARDS[c].points == 1 and c not in sum(game.market, [])]
        me.purchased = filler[:15 - card.points]          # 15 - card.points points: the card wins
        need = [max(0, c - b) for c, b in zip(card.cost, me.bonuses)]
        me.tokens = need + [0]
        if sum(me.tokens) > 10 or me.points + card.points < 15:
            self.skipTest('constructed position not reachable with this deal')
        game.bank = [max(0, 7 - t) for t in me.tokens[:5]] + [5]
        action = bots.lookahead_action(game, random.Random(0))
        game.step(action, random.Random(0))
        self.assertEqual((game.phase, game.winners), ('terminal', [0]))

    def test_lookahead_blocks_an_opponent_win(self):
        game = new_game(random.Random(5))
        card_id = next(c for row in game.market for c in row if c and CARDS[c].points >= 2)
        card = CARDS[card_id]
        opponent = game.players[1]
        opponent.purchased = [c for c in CARDS if CARDS[c].points == 1 and c not in sum(game.market, [])][:15 - card.points]
        opponent.tokens = [max(0, c - b) for c, b in zip(card.cost, opponent.bonuses)] + [0]
        if sum(opponent.tokens) > 10:
            self.skipTest('constructed position not reachable with this deal')
        self.assertIn(card_id, bots._winning_cards(game, 1))
        game.players[0] = Player()
        action = bots.lookahead_action(game, random.Random(0))
        self.assertEqual(action.kind, 'reserve')
        self.assertIn(action.card_id, bots._winning_cards(game, 1))


if __name__ == '__main__':
    unittest.main()
