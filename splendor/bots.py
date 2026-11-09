"""Rule-based Splendor players: baselines and benchmark opponents for the trained agent (not search AIs).

policy 'random'     uniformly random legal action
policy 'heuristic'  greedy one-step scoring of each action (progress towards the best three targets)
policy 'lookahead'  stronger rule player: plays every legal action on a copy of the state and scores the
                    position it leads to (points, engine, nobles, how many turns the best cards still need),
                    wins at once when it can, and reserves a card an opponent would win with on his next
                    turn. Used as the "average player" proxy of the benchmark (tools/splendor_benchmark.py).
policy 'lookahead_alt1', 'lookahead_alt2'  the lookahead player with fixed, deterministically perturbed weights
                    (each weight times exp(U(-0.25, 0.25)), seeds 8 and 10, picked among seeds 1-10 as the two
                    closest to lookahead's strength: 0.245 / 0.25 against 3 lookahead, 200 games): the same
                    algorithm in a different style. Held-out benchmark opponents (auto_research R007): never a
                    teacher, never in self-play.
"""

import math
import random

from .engine import CARDS, NOBLES

POLICIES = ('random', 'heuristic', 'lookahead', 'lookahead_alt1', 'lookahead_alt2')


def choose_action(game, rng=None, policy="heuristic"):
    rng = rng if rng is not None else random.Random()
    if policy == "lookahead":
        return lookahead_action(game, rng)
    if policy in ALT_WEIGHTS:
        return lookahead_action(game, rng, ALT_WEIGHTS[policy])
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


# ---------------------------------------------------------------------------- lookahead player
WIN_VALUE = 1e6


def _turns_to_buy(cost, bonuses, tokens, bank):
    """rough number of take-token turns before the card is affordable (0 = affordable now); None = the bank
    cannot supply it"""
    short_total, short_max = 0, 0
    for k in range(5):
        short = cost[k] - bonuses[k] - tokens[k]
        if short > 0:
            if short > bank[k] + tokens[5] + 1:
                return None
            short_total += short
            short_max = max(short_max, short)
    short_total -= tokens[5]
    if short_total <= 0:
        return 0
    # three different colours per turn, or two of one colour
    return max(math.ceil(short_total / 3), math.ceil(short_max / 2))


def _noble_points_now(game, bonuses):
    return 3 if any(all(b >= c for b, c in zip(bonuses, NOBLES[n].cost)) for n in game.nobles) else 0


def evaluate_position(game, seat, weights=None):
    """heuristic value of the position for player `seat` (larger is better); WIN_VALUE when he has won"""
    w = DEFAULT_WEIGHTS if weights is None else weights
    p = game.players[seat]
    bonuses = p.bonuses
    points = p.points
    if game.phase in ("chance", "discard", "noble") and game.current_player == seat:
        points += _noble_points_now(game, bonuses)   # a noble earned by this turn's purchase
    if points >= 15:
        return WIN_VALUE + points
    tokens = p.tokens
    leader = max(q.points for q in game.players)
    progress = min(1., leader / 15.)                   # 0 early, 1 at the end of the game
    market = [c for row in game.market for c in row if c is not None]

    # what one more bonus of each colour is worth: discounts on the visible cards and noble progress
    demand = [0.] * 5
    for card_id in market + p.reserved:
        card = CARDS[card_id]
        weight = (1. + card.points) * (1.5 if card_id in p.reserved else 1.)
        for k in range(5):
            if card.cost[k] > bonuses[k]:
                demand[k] += weight * w['demand']
    noble_term = 0.
    for n in game.nobles:
        cost = NOBLES[n].cost
        missing = sum(max(0, c - b) for c, b in zip(cost, bonuses))
        noble_term += w['noble'] / (1. + missing) ** 1.6
        if missing <= 4:
            for k in range(5):
                if cost[k] > bonuses[k]:
                    demand[k] += w['noble_demand']
    engine_weight = w['engine'] * (1. - progress) + w['engine_late']
    engine = sum(engine_weight * min(b, 6) for b in bonuses)

    # how fast the best cards can be bought
    best = list()
    for card_id in market + p.reserved:
        card = CARDS[card_id]
        turns = _turns_to_buy(card.cost, bonuses, tokens, game.bank)
        if turns is None:
            continue
        gain = card.points * (1. + progress) + min(1.5, demand[card.bonus]) * (1. - .7 * progress)
        if points + card.points >= 15:
            gain += w['finisher']
        best.append(gain / (1. + turns) ** w['turn_exp'])
    best.sort(reverse=True)
    potential = w['potential'] * sum(f * v for f, v in zip((1., .5, .25), best))

    held = sum(tokens)
    token_term = w['token'] * min(held, 10) + w['gold'] * tokens[5] - .4 * max(0, held - 10)
    cards = w['card'] * (1. - progress) * len(p.purchased)
    return w['points'] * points * (1. + progress) + engine + cards + potential + noble_term + token_term


# tuned by hill climbing on the win share of one lookahead player against three heuristic players (160 games
# per candidate, seats rotated); on 200 other games: 0.805 (equal strength 0.25), 0.99 against random players,
# about 25 rounds per game among lookahead players
DEFAULT_WEIGHTS = {'demand': .046, 'noble': 2.68, 'noble_demand': .25, 'engine': 1.29, 'engine_late': .15,
                   'finisher': 6., 'potential': .525, 'token': .19, 'gold': .1, 'deny': 8., 'points': 1.,
                   'turn_exp': 1., 'card': 0.}


def _perturbed_weights(seed, spread=.25):
    rng = random.Random(seed)
    return {k: v * math.exp(rng.uniform(-spread, spread)) for k, v in sorted(DEFAULT_WEIGHTS.items())}


ALT_WEIGHTS = {'lookahead_alt1': _perturbed_weights(8), 'lookahead_alt2': _perturbed_weights(10)}


def _winning_cards(game, seat):
    """market cards player `seat` could buy on his next turn to reach 15 points"""
    p = game.players[seat]
    bonuses = p.bonuses
    result = list()
    for row in game.market:
        for card_id in row:
            if card_id is None:
                continue
            card = CARDS[card_id]
            short = sum(max(0, c - b - t) for c, b, t in zip(card.cost, bonuses, p.tokens))
            if short > p.tokens[5]:
                continue
            after = bonuses[:]
            after[card.bonus] += 1
            if p.points + card.points + _noble_points_now(game, after) >= 15:
                result.append(card_id)
    return result


def lookahead_action(game, rng, weights=None):
    w = DEFAULT_WEIGHTS if weights is None else weights
    actions = game.legal_actions(all_payments=False)
    if not actions:
        raise ValueError("当前阶段没有玩家动作")
    seat = game.current_player
    threats = set()
    if game.phase == "action":
        for other in range(4):
            if other != seat:
                threats.update(_winning_cards(game, other))
    best, best_score = None, -math.inf
    for action in actions:
        after = game.apply(action)            # stops at a replacement draw: the slot stays empty
        score = evaluate_position(after, seat, w)
        if action.kind == "reserve" and action.card_id in threats and score < WIN_VALUE:
            score += w['deny']                  # take the card an opponent would win with
        score += rng.random() * .01
        if score > best_score:
            best, best_score = action, score
    return best
