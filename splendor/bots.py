"""Small heuristic baseline for exercising the simulator, not a trained/search AI."""

import random

from .engine import CARDS, NOBLES


def choose_action(game, rng=None, policy="heuristic"):
    rng = rng if rng is not None else random.Random()
    actions = game.legal_actions()
    if not actions:
        raise ValueError("当前阶段没有玩家动作")
    if policy == "random":
        return rng.choice(actions)
    if policy != "heuristic":
        raise ValueError("未知策略")
    p = game.players[game.current_player]
    bonuses = p.bonuses
    targets = [CARDS[c] for row in game.market for c in row if c] + [CARDS[c] for c in p.reserved]

    def distance(card, tokens):
        return max(0, sum(max(0, c - b - t) for c, b, t in zip(card.cost, bonuses, tokens)) - tokens[5])

    def hand_value(tokens):
        return sum(sorted(((1 + card.points * .4) / (1 + distance(card, tokens)) for card in targets),
                          reverse=True)[:3])

    def score(action):
        if action.kind == "noble":
            return 100
        if action.kind == "discard":
            # Return the token whose loss hurts the progress towards the best targets least.
            tokens = p.tokens[:]
            tokens[action.colors[0]] -= 1
            return hand_value(tokens)
        if action.kind == "buy":
            card = CARDS[action.card_id]
            after = bonuses[:]
            after[card.bonus] += 1
            noble_gain = sum(all(b >= c for b, c in zip(after, NOBLES[n].cost)) for n in game.nobles)
            progress = sum(bonuses[card.bonus] < NOBLES[n].cost[card.bonus] for n in game.nobles)
            win = p.points + card.points + 3 * bool(noble_gain) >= 15
            return (100 if win else 0) + 8 + card.points * 3 + bool(noble_gain) * 9 + progress * .4 - sum(action.payment) * .3 - action.payment[5] * .2
        if action.kind == "take":
            tokens = p.tokens[:]
            for color in action.colors:
                tokens[color] += 1
            gains = [((1 + card.points * .4) / (1 + distance(card, tokens))
                      - (1 + card.points * .4) / (1 + distance(card, p.tokens)))
                     for card in targets]
            return 2 + max(gains, default=0) * 3 + sum(sorted(gains, reverse=True)[:3])
        if action.kind == "reserve":
            card = CARDS[action.card_id]
            return 1 + (card.points * .5 + 1) / (1 + distance(card, p.tokens)) + bool(game.bank[5]) * .8
        return -100

    # Random tie-breaks are independent from the environment RNG.
    return max(actions, key=lambda action: score(action) + rng.random() * .08)
