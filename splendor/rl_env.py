"""Splendor environment adapter for the AlphaGoZero-style training pipeline.

The rules engine (splendor/engine.py) stays dependency free. This module adds what the
RL pipeline needs on top of it, mirroring the role env/env.py plays for poker:

* a fixed-size discrete action space (``N_ACTIONS``) plus a legal-action mask,
* an integer observation, encoded relative to the acting player, that can be fed to an
  embedding table (one vocab range per field group, same idea as rl/models.EnvEmbedding),
* terminal rewards for all four seats,
* ``new_random()``: a copy of the state with a fresh RNG for the hidden information
  (future deck order), the splendor equivalent of resampling hidden cards in poker MCTS.

Action space (67 actions):
    0-9    take three different colors (itertools.combinations(range(5), 3))
    10-19  take two different colors   (only legal when the bank has exactly two colors left)
    20-24  take one color              (only legal when the bank has one color left)
    25-29  take two of the same color
    30-41  reserve a market card,      index = level * 4 + slot
    42-53  buy a market card,          index = level * 4 + slot
    54-56  buy own reserved card,      index = position in player.reserved
    57-66  choose noble N1..N10        (only in the noble phase)

Buying always uses the payment that spends the least gold (the engine sorts payments by
gold first, and the least-gold payment is unique). Voluntary gold substitution is
therefore not part of the learned policy.
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from .engine import CARDS, COLORS, NOBLES, Action, new_game

N_PLAYERS = 4
NUM_COLORS = len(COLORS)

TAKE3_COMBOS = list(combinations(range(NUM_COLORS), 3))
TAKE2_COMBOS = list(combinations(range(NUM_COLORS), 2))

OFFSET_TAKE3 = 0
OFFSET_TAKE2 = OFFSET_TAKE3 + len(TAKE3_COMBOS)      # 10
OFFSET_TAKE1 = OFFSET_TAKE2 + len(TAKE2_COMBOS)      # 20
OFFSET_TAKE_DOUBLE = OFFSET_TAKE1 + NUM_COLORS        # 25
OFFSET_RESERVE = OFFSET_TAKE_DOUBLE + NUM_COLORS      # 30
OFFSET_BUY_MARKET = OFFSET_RESERVE + 12               # 42
OFFSET_BUY_RESERVED = OFFSET_BUY_MARKET + 12          # 54
OFFSET_NOBLE = OFFSET_BUY_RESERVED + 3                # 57
NOBLE_IDS = sorted(NOBLES, key=lambda n: int(n[1:]))
N_ACTIONS = OFFSET_NOBLE + len(NOBLE_IDS)             # 67

ACTION_NAMES = (
    [f'take3-{"".join("WBGRK"[c] for c in cs)}' for cs in TAKE3_COMBOS]
    + [f'take2-{"".join("WBGRK"[c] for c in cs)}' for cs in TAKE2_COMBOS]
    + [f'take1-{"WBGRK"[c]}' for c in range(NUM_COLORS)]
    + [f'double-{"WBGRK"[c]}' for c in range(NUM_COLORS)]
    + [f'reserve-L{i // 4 + 1}S{i % 4 + 1}' for i in range(12)]
    + [f'buy-L{i // 4 + 1}S{i % 4 + 1}' for i in range(12)]
    + [f'buy-reserved{i + 1}' for i in range(3)]
    + [f'noble-{n}' for n in NOBLE_IDS]
)
assert len(ACTION_NAMES) == N_ACTIONS

CARD_IDS = sorted(CARDS)
CARD_INDEX = {card_id: i for i, card_id in enumerate(CARD_IDS)}
EMPTY_CARD = len(CARD_IDS)                 # 90
NOBLE_INDEX = {noble_id: i for i, noble_id in enumerate(NOBLE_IDS)}
EMPTY_NOBLE = len(NOBLE_IDS)               # 10

# ---------------------------------------------------------------------------
# Observation layout.
# Every field belongs to a vocab group; fields of the same group share one embedding
# range (like the (num_fields, num_dims) groups of rl/models.EnvEmbedding) and are told
# apart by the factorised position embedding (FIELD_LAYOUT below).
# ---------------------------------------------------------------------------
MAX_TURN_BUCKET = 127
CARD_FIELD_GROUPS = ['card_id'] + [f'cost_{c}' for c in COLORS] + ['card_points', 'card_bonus']

VOCAB_SIZES = {
    'phase': 2,
    'turn': MAX_TURN_BUCKET + 1,
    'noble_id': EMPTY_NOBLE + 1,
    'card_id': EMPTY_CARD + 1,
    'card_points': 7,          # 0..5, 6 = empty slot
    'card_bonus': NUM_COLORS + 1,  # 5 = empty slot
    'deck_remaining': 41,
    'player_points': 31,
    'player_nobles': 6,
    'player_purchased': 32,
    'player_reserved_num': 4,
}
for _c in COLORS + ('gold',):
    VOCAB_SIZES[f'bank_{_c}'] = 8
    VOCAB_SIZES[f'tokens_{_c}'] = 11
for _c in COLORS:
    VOCAB_SIZES[f'cost_{_c}'] = 9    # 0..7, 8 = empty slot
    VOCAB_SIZES[f'bonus_{_c}'] = 16  # capped at 15


# Factorised field position: every field is (attr, owner, slot).
#   attr:  what the field measures    -> its vocab group (card_id, cost_blue, tokens_gold, ...)
#   owner: who the field belongs to   -> global / market / player0..3 (relative to the acting player)
#   slot:  which entity of that owner -> market card L1S1..L3S4, reserved card 1..3, deck level, noble 1..5, none
# The model embeds the three parts separately and adds them (rl/splendor_model.py), so e.g. the
# 8 tokens of one card share a slot vector and all fields of "opponent k" share an owner vector.
OWNERS = ['global', 'market'] + [f'player{k}' for k in range(N_PLAYERS)]
SLOTS = (['none']
         + [f'L{level + 1}S{slot + 1}' for level in range(3) for slot in range(4)]
         + [f'reserved{i + 1}' for i in range(3)]
         + [f'deck_L{level + 1}' for level in range(3)]
         + [f'noble{i + 1}' for i in range(5)])


def _build_field_layout():
    """[(attr = vocab group, owner, slot)] in exactly the order encode_observation writes the fields."""
    layout = [('phase', 'global', 'none'), ('turn', 'global', 'none')]
    layout += [(f'bank_{c}', 'global', 'none') for c in COLORS + ('gold',)]
    layout += [('deck_remaining', 'global', f'deck_L{level + 1}') for level in range(3)]
    layout += [('noble_id', 'global', f'noble{i + 1}') for i in range(5)]
    for level in range(3):
        for slot in range(4):
            layout += [(group, 'market', f'L{level + 1}S{slot + 1}') for group in CARD_FIELD_GROUPS]
    for k in range(N_PLAYERS):
        owner = f'player{k}'
        layout += [(f'tokens_{c}', owner, 'none') for c in COLORS + ('gold',)]
        layout += [(f'bonus_{c}', owner, 'none') for c in COLORS]
        layout += [(group, owner, 'none') for group in ('player_points', 'player_nobles', 'player_purchased', 'player_reserved_num')]
        for i in range(3):
            layout += [(group, owner, f'reserved{i + 1}') for group in CARD_FIELD_GROUPS]
    return layout


FIELD_LAYOUT = _build_field_layout()
assert len(set(FIELD_LAYOUT)) == len(FIELD_LAYOUT), 'every field must have a unique (attr, owner, slot)'
FIELD_GROUPS = [attr for attr, _, _ in FIELD_LAYOUT]
ATTRS = sorted(set(FIELD_GROUPS))
FIELD_ATTR_IDX = [ATTRS.index(attr) for attr, _, _ in FIELD_LAYOUT]
FIELD_OWNER_IDX = [OWNERS.index(owner) for _, owner, _ in FIELD_LAYOUT]
FIELD_SLOT_IDX = [SLOTS.index(slot) for _, _, slot in FIELD_LAYOUT]
N_OBSERVATION = len(FIELD_GROUPS)   # 268


def _card_tokens(card_id):
    if card_id is None:
        return [EMPTY_CARD] + [8] * NUM_COLORS + [6, NUM_COLORS]
    card = CARDS[card_id]
    return [CARD_INDEX[card_id]] + [min(v, 7) for v in card.cost] + [card.points, card.bonus]


def encode_observation(game):
    """Integer observation from the acting player's point of view (seat 0 = acting player)."""
    me = game.current_player
    obs = [0 if game.phase == 'action' else 1, min(game.turn // 2, MAX_TURN_BUCKET)]
    obs += [min(v, 7) for v in game.bank]
    obs += [min(len(pool), 40) for pool in game.remaining]
    nobles = [NOBLE_INDEX[n] for n in game.nobles] + [EMPTY_NOBLE] * (5 - len(game.nobles))
    obs += nobles
    for row in game.market:
        for card_id in row:
            obs += _card_tokens(card_id)
    for k in range(N_PLAYERS):
        p = game.players[(me + k) % N_PLAYERS]
        obs += list(p.tokens)
        obs += [min(b, 15) for b in p.bonuses]
        obs += [min(p.points, 30), len(p.nobles), min(len(p.purchased), 31), len(p.reserved)]
        reserved = p.reserved + [None] * (3 - len(p.reserved))
        for card_id in reserved:
            obs += _card_tokens(card_id)
    assert len(obs) == N_OBSERVATION
    return np.asarray(obs, dtype=np.int32)


def action_to_index(game, action):
    if action.kind == 'take':
        cs = tuple(action.colors)
        if len(cs) == 3:
            return OFFSET_TAKE3 + TAKE3_COMBOS.index(cs)
        if len(cs) == 2 and cs[0] == cs[1]:
            return OFFSET_TAKE_DOUBLE + cs[0]
        if len(cs) == 2:
            return OFFSET_TAKE2 + TAKE2_COMBOS.index(cs)
        return OFFSET_TAKE1 + cs[0]
    if action.kind == 'noble':
        return OFFSET_NOBLE + NOBLE_INDEX[action.noble_id]
    for level, row in enumerate(game.market):
        for slot, card_id in enumerate(row):
            if card_id == action.card_id:
                base = OFFSET_RESERVE if action.kind == 'reserve' else OFFSET_BUY_MARKET
                return base + level * 4 + slot
    assert action.kind == 'buy', action
    return OFFSET_BUY_RESERVED + game.players[game.current_player].reserved.index(action.card_id)


class SplendorRLEnv:
    """Thin stateful wrapper of splendor.engine.Game used by MCTS and self-play."""

    n_actions = N_ACTIONS
    n_observation = N_OBSERVATION
    n_players = N_PLAYERS

    def __init__(self, max_turns=400, seed=None):
        self.max_turns = max_turns
        self.rng = random.Random(seed)
        self.game = None
        self.truncated = False
        self._legal_cache = None

    # ----- lifecycle -----
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.game = new_game(self.rng)
        self.truncated = False
        self._legal_cache = None
        return self.observation()

    def new_random(self, seed=None):
        """Same public state, independent RNG for future (hidden) deck draws."""
        env = SplendorRLEnv.__new__(SplendorRLEnv)
        env.max_turns = self.max_turns
        env.rng = random.Random(seed)
        env.game = self.game.clone()
        env.truncated = self.truncated
        env._legal_cache = self._legal_cache
        return env

    # ----- state queries -----
    @property
    def acting_player(self):
        return self.game.current_player

    @property
    def terminated(self):
        return self.truncated or self.game.phase in ('terminal', 'blocked')

    @property
    def end_reason(self):
        if self.truncated:
            return 'truncated'
        return self.game.end_reason

    def key(self):
        return self.game.key()

    def observation(self):
        return encode_observation(self.game)

    def legal_actions(self):
        """dict: action index -> engine Action (least-gold payment for purchases)."""
        if self._legal_cache is None:
            result = dict()
            if not self.terminated:
                for action in self.game.legal_actions():
                    idx = action_to_index(self.game, action)
                    # engine sorts payments by gold first: keep the first (least gold) one
                    if idx not in result:
                        result[idx] = action
            self._legal_cache = result
        return self._legal_cache

    def legal_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[list(self.legal_actions().keys())] = True
        return mask

    # ----- transition -----
    def step(self, action_idx, sample_chance=True):
        """Apply a player action.

        sample_chance=True: a replacement draw is sampled immediately with self.rng (real games).
        sample_chance=False: stop at the chance node (``at_chance_node``) so the caller (MCTS)
        picks the drawn card itself with ``chance_outcomes()`` / ``resolve_chance()``.
        """
        legal = self.legal_actions()
        if action_idx not in legal:
            raise ValueError(f'illegal action {action_idx} ({ACTION_NAMES[action_idx]})')
        self.game.step(legal[action_idx], self.rng if sample_chance else None)
        self._after_transition()
        return self.terminated

    @property
    def at_chance_node(self):
        return self.game.phase == 'chance'

    def chance_outcomes(self):
        """[(card_id, probability)] of the pending replacement draw."""
        return self.game.chance_outcomes()

    def resolve_chance(self, card_id):
        self.game.resolve_chance(card_id)
        self._after_transition()
        return self.terminated

    def _after_transition(self):
        self._legal_cache = None
        if not self.terminated and not self.at_chance_node and self.game.turn >= self.max_turns:
            self.truncated = True

    def final_rewards(self):
        """Absolute-seat rewards. Win share for normal ends, all zero for blocked/truncated."""
        if self.game.phase == 'terminal':
            return np.asarray(self.game.rewards(), dtype=np.float32)
        return np.zeros(N_PLAYERS, dtype=np.float32)


def to_relative(values, seat):
    """Absolute seat order -> order relative to `seat` (index 0 = seat)."""
    return np.roll(np.asarray(values), -seat)


def to_absolute(values, seat):
    """Order relative to `seat` -> absolute seat order."""
    return np.roll(np.asarray(values), seat)
