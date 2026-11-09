"""Splendor environment adapter for the AlphaGoZero-style training pipeline.

The rules engine (splendor/engine.py) stays dependency free. This module adds what the
RL pipeline needs on top of it, mirroring the role env/env.py plays for poker:

* a fixed-size discrete action space (``N_ACTIONS``) plus a legal-action mask,
* an integer observation, encoded relative to the acting player, that can be fed to an
  embedding table (one vocab range per field group, same idea as rl/models.EnvEmbedding),
* terminal rewards for all four seats,
* the exact symmetries of the encoding (``SYMMETRY_GROUPS``) for data augmentation,
* ``new_random()``: a copy of the state with a fresh RNG for the hidden information
  (future deck order), the splendor equivalent of resampling hidden cards in poker MCTS.

Action space (68 actions):
    0-9    take three different colors (itertools.combinations(range(5), 3))
    10-19  take two different colors   (only legal when the bank has exactly two colors left)
    20-24  take one color              (only legal when the bank has one color left)
    25-29  take two of the same color
    30-41  reserve a market card,      index = level * 4 + slot
    42-53  buy a market card,          index = level * 4 + slot
    54-56  buy own reserved card,      index = position in player.reserved
    57-61  take the noble in noble slot 1..5 (only in the noble phase; slot = position in game.nobles)
    62-67  return one token W/B/G/R/K/gold (only in the discard phase: official rule, tokens
           above 10 are returned one at a time before nobles and the end of the turn)

Buying always uses the payment that spends the least gold (the engine sorts payments by
gold first, and the least-gold payment is unique). Voluntary gold substitution is
therefore not part of the learned policy.

Observation: cards and nobles are described by what they are (level, cost, points, bonus /
noble requirement), never by an id. Every card has a unique (cost, points, bonus), so no
information is lost, but nothing invites the network to learn one embedding per card either.
Market slots, noble slots and reserved slots are interchangeable: permuting the slots of one
group (and the slot-indexed actions with them) gives an equivalent state. SYMMETRY_GROUPS lists
these groups; training batches and MCTS evaluations apply random permutations of them.
"""

from __future__ import annotations

import hashlib
import random
from itertools import combinations

import numpy as np

from .engine import CARDS, COLORS, NOBLES, RULES_VERSION, Action, new_game

N_PLAYERS = 4
NUM_COLORS = len(COLORS)
NUM_NOBLE_SLOTS = 5

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
OFFSET_DISCARD = OFFSET_NOBLE + NUM_NOBLE_SLOTS       # 62
N_ACTIONS = OFFSET_DISCARD + NUM_COLORS + 1           # 68

ACTION_NAMES = (
    [f'take3-{"".join("WBGRK"[c] for c in cs)}' for cs in TAKE3_COMBOS]
    + [f'take2-{"".join("WBGRK"[c] for c in cs)}' for cs in TAKE2_COMBOS]
    + [f'take1-{"WBGRK"[c]}' for c in range(NUM_COLORS)]
    + [f'double-{"WBGRK"[c]}' for c in range(NUM_COLORS)]
    + [f'reserve-L{i // 4 + 1}S{i % 4 + 1}' for i in range(12)]
    + [f'buy-L{i // 4 + 1}S{i % 4 + 1}' for i in range(12)]
    + [f'buy-reserved{i + 1}' for i in range(3)]
    + [f'noble-slot{i + 1}' for i in range(NUM_NOBLE_SLOTS)]
    + [f'discard-{"WBGRKY"[c]}' for c in range(NUM_COLORS + 1)]
)
assert len(ACTION_NAMES) == N_ACTIONS

_TAKE3_INDEX = {cs: OFFSET_TAKE3 + i for i, cs in enumerate(TAKE3_COMBOS)}
_TAKE2_INDEX = {cs: OFFSET_TAKE2 + i for i, cs in enumerate(TAKE2_COMBOS)}

# ---------------------------------------------------------------------------
# Observation layout.
# Every field belongs to a vocab group; fields of the same group share one embedding
# range (like the (num_fields, num_dims) groups of rl/models.EnvEmbedding) and are told
# apart by the factorised position embedding (FIELD_LAYOUT below).
# ---------------------------------------------------------------------------
MAX_ROUND_BUCKET = 63
# Observation version 2 (auto_research R005): derived fields the rule bots' evaluation is built from, which the
# network otherwise has to compute from the raw cost / bonus / token fields:
#   card_short_owner  gems the card's buyer still lacks after bonuses, tokens and gold (market card: the acting
#                     player; reserved card of player k: player k), 0..8, 9 = 9 or more, 10 = empty slot
#   card_short_rival  the same, smallest over the acting player's opponents (market cards; 10 for reserved cards)
#   noble_missing_owner / noble_missing_rival  bonuses the acting player / the closest opponent still lacks for
#                     the noble, 0..8, 9 = 9 or more, 10 = empty slot
CARD_FIELD_GROUPS = (['card_level'] + [f'cost_{c}' for c in COLORS] + ['card_points', 'card_bonus']
                     + ['card_short_owner', 'card_short_rival'])
NOBLE_FIELD_GROUPS = [f'noble_req_{c}' for c in COLORS] + ['noble_missing_owner', 'noble_missing_rival']
DERIVED_FIELD_GROUPS = ('card_short_owner', 'card_short_rival', 'noble_missing_owner', 'noble_missing_rival')
DERIVED_EMPTY = 10

VOCAB_SIZES = {
    'phase': 3,                # action / noble / discard
    'discard_needed': 4,       # tokens the acting player still has to return (0..3)
    'round': MAX_ROUND_BUCKET + 1,   # game.turn // 4
    'seat': N_PLAYERS,         # absolute seat of the acting player (0 = starting player)
    'card_level': 4,           # 0..2, 3 = empty slot
    'card_points': 7,          # 0..5, 6 = empty slot
    'card_bonus': NUM_COLORS + 1,  # 5 = empty slot
    'deck_remaining': 41,
    'player_points': 31,
    'player_nobles': 6,
    'player_purchased': 32,
    'player_reserved_num': 4,
}
for _g in DERIVED_FIELD_GROUPS:
    VOCAB_SIZES[_g] = DERIVED_EMPTY + 1
for _c in COLORS + ('gold',):
    VOCAB_SIZES[f'bank_{_c}'] = 8
    VOCAB_SIZES[f'tokens_{_c}'] = 11
for _c in COLORS:
    VOCAB_SIZES[f'cost_{_c}'] = 9    # 0..7, 8 = empty slot
    VOCAB_SIZES[f'bonus_{_c}'] = 16  # capped at 15
    VOCAB_SIZES[f'noble_req_{_c}'] = 6   # 0..4, 5 = empty slot


# Factorised field position: every field is (attr, owner, slot).
#   attr:  what the field measures    -> its vocab group (card_level, cost_blue, tokens_gold, ...)
#   owner: who the field belongs to   -> global / market / player0..3 (relative to the acting player)
#   slot:  which entity of that owner -> market card L1S1..L3S4, reserved card 1..3, deck level, noble 1..5, none
# The model embeds the three parts separately and adds them (rl/splendor_model.py), so e.g. the
# 8 tokens of one card share a slot vector and all fields of "opponent k" share an owner vector.
OWNERS = ['global', 'market'] + [f'player{k}' for k in range(N_PLAYERS)]
SLOTS = (['none']
         + [f'L{level + 1}S{slot + 1}' for level in range(3) for slot in range(4)]
         + [f'reserved{i + 1}' for i in range(3)]
         + [f'deck_L{level + 1}' for level in range(3)]
         + [f'noble{i + 1}' for i in range(NUM_NOBLE_SLOTS)])


def _build_field_layout():
    """[(attr = vocab group, owner, slot)] in exactly the order encode_observation writes the fields."""
    layout = [('phase', 'global', 'none'), ('discard_needed', 'global', 'none'), ('round', 'global', 'none'),
              ('seat', 'global', 'none')]
    layout += [(f'bank_{c}', 'global', 'none') for c in COLORS + ('gold',)]
    layout += [('deck_remaining', 'global', f'deck_L{level + 1}') for level in range(3)]
    for i in range(NUM_NOBLE_SLOTS):
        layout += [(group, 'global', f'noble{i + 1}') for group in NOBLE_FIELD_GROUPS]
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
N_OBSERVATION = len(FIELD_GROUPS)   # 290 in observation version 1, 348 in version 2
assert all(v <= 256 for v in VOCAB_SIZES.values()), 'observation values must fit in one byte'


def _field_index(attr, owner, slot):
    return FIELD_LAYOUT.index((attr, owner, slot))


# ---------------------------------------------------------------------------
# Symmetries: groups of interchangeable slots. A symmetry picks one permutation per group;
# slot j of the transformed state holds what slot perm[j] of the original state held, for the
# observation fields and for the slot-indexed actions alike.
#   (obs_positions [n_slots, fields per slot], action_positions [n_slots, actions per slot] or None)
# ---------------------------------------------------------------------------
def _build_symmetry_groups():
    groups = []
    for level in range(3):
        obs = [[_field_index(g, 'market', f'L{level + 1}S{s + 1}') for g in CARD_FIELD_GROUPS] for s in range(4)]
        act = [[OFFSET_RESERVE + level * 4 + s, OFFSET_BUY_MARKET + level * 4 + s] for s in range(4)]
        groups.append((np.asarray(obs), np.asarray(act)))
    obs = [[_field_index(g, 'global', f'noble{i + 1}') for g in NOBLE_FIELD_GROUPS] for i in range(NUM_NOBLE_SLOTS)]
    act = [[OFFSET_NOBLE + i] for i in range(NUM_NOBLE_SLOTS)]
    groups.append((np.asarray(obs), np.asarray(act)))
    for k in range(N_PLAYERS):
        obs = [[_field_index(g, f'player{k}', f'reserved{i + 1}') for g in CARD_FIELD_GROUPS] for i in range(3)]
        act = [[OFFSET_BUY_RESERVED + i] for i in range(3)] if k == 0 else None   # only the acting player can buy
        groups.append((np.asarray(obs), None if act is None else np.asarray(act)))
    return groups


SYMMETRY_GROUPS = _build_symmetry_groups()


# ---------------------------------------------------------------------------
# Versions. Training data and checkpoints record them (rl/splendor_data.py) and are refused when they differ.
#   RULES_VERSION (splendor/engine.py): what a state, a legal action or a game result is
#   OBSERVATION_VERSION: what the encoded fields mean;  ACTION_VERSION: what each action index means
# Bump the version when the meaning changes. The layout hashes catch layout changes made without a bump.
# ---------------------------------------------------------------------------
OBSERVATION_VERSION = 2   # R005: derived shortfall / noble-distance fields (DERIVED_FIELD_GROUPS)
ACTION_VERSION = 1


def _short_hash(value):
    return hashlib.sha1(repr(value).encode('utf-8')).hexdigest()[:12]


def _plain(value):
    """nested lists / tuples / numpy arrays -> plain Python lists of ints and strings (repr is the same for every
    numpy version: numpy 2 prints np.int64(3) where numpy 1 prints 3)"""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


OBSERVATION_LAYOUT_HASH = _short_hash(_plain([FIELD_LAYOUT, sorted(VOCAB_SIZES.items()),
                                              [[o, a] for o, a in SYMMETRY_GROUPS]]))
ACTION_LAYOUT_HASH = _short_hash(_plain(ACTION_NAMES))


def version_info(discard_over_limit=True):
    """what training data and checkpoints must agree on (discard_over_limit is a rule variant)"""
    return {'rules': RULES_VERSION, 'observation': OBSERVATION_VERSION, 'action': ACTION_VERSION,
            'observation_layout': OBSERVATION_LAYOUT_HASH, 'action_layout': ACTION_LAYOUT_HASH,
            'discard_over_limit': bool(discard_over_limit)}


def symmetry_indices(perms):
    """perms: one permutation (sequence of slot indices) per SYMMETRY_GROUPS entry ->
    (obs_index [N_OBSERVATION], action_index [N_ACTIONS]): transformed = original[index] for the
    observation, and for anything indexed by action (policy target, legal mask)."""
    obs_index = np.arange(N_OBSERVATION)
    action_index = np.arange(N_ACTIONS)
    for (obs_pos, act_pos), perm in zip(SYMMETRY_GROUPS, perms):
        perm = np.asarray(perm)
        obs_index[obs_pos.reshape(-1)] = obs_pos[perm].reshape(-1)
        if act_pos is not None:
            action_index[act_pos.reshape(-1)] = act_pos[perm].reshape(-1)
    return obs_index, action_index


def random_symmetry_indices(rng):
    """rng: numpy Generator -> symmetry_indices of a uniformly random symmetry"""
    return symmetry_indices([rng.permutation(len(obs_pos)) for obs_pos, _ in SYMMETRY_GROUPS])


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def _card_tokens(card_id):
    """the static fields of a card (without the derived ones)"""
    if card_id is None:
        return [3] + [8] * NUM_COLORS + [6, NUM_COLORS]
    card = CARDS[card_id]
    return [card.level - 1] + [min(v, 7) for v in card.cost] + [card.points, card.bonus]


_CARD_TOKENS = {card_id: _card_tokens(card_id) for card_id in CARDS}
_EMPTY_CARD_TOKENS = _card_tokens(None) + [DERIVED_EMPTY, DERIVED_EMPTY]
_NOBLE_TOKENS = {noble_id: list(noble.cost) for noble_id, noble in NOBLES.items()}
_EMPTY_NOBLE_TOKENS = [5] * NUM_COLORS + [DERIVED_EMPTY, DERIVED_EMPTY]
_CARD_COST = {card_id: tuple(CARDS[card_id].cost) for card_id in CARDS}
_NOBLE_COST = {noble_id: tuple(noble.cost) for noble_id, noble in NOBLES.items()}


def card_shortfall(cost, bonuses, tokens):
    """gems a player still lacks for a card after his bonuses, coloured tokens and gold (0 = affordable now)"""
    short = 0
    for i in range(NUM_COLORS):
        c = cost[i] - bonuses[i] - tokens[i]
        if c > 0:
            short += c
    short -= tokens[NUM_COLORS]
    return short if short > 0 else 0


def noble_missing(cost, bonuses):
    """bonuses a player still lacks for a noble"""
    missing = 0
    for i in range(NUM_COLORS):
        c = cost[i] - bonuses[i]
        if c > 0:
            missing += c
    return missing


def _clip9(v):
    return v if v < 9 else 9
_NONE3 = [None, None, None]
PHASE_INDEX = {'action': 0, 'noble': 1, 'discard': 2}


def encode_observation_list(game):
    """Integer observation (list of ints, each < 256) from the acting player's point of view
    (player0 = acting player). Field order: FIELD_LAYOUT."""
    me = game.current_player
    players = game.players
    obs = [PHASE_INDEX[game.phase],
           min(max(sum(players[me].tokens) - 10, 0), 3) if game.phase == 'discard' else 0,
           min(game.turn // 4, MAX_ROUND_BUCKET), me]
    obs += [v if v < 7 else 7 for v in game.bank]
    obs += [len(pool) if len(pool) < 40 else 40 for pool in game.remaining]
    seats = [players[(me + k) % N_PLAYERS] for k in range(N_PLAYERS)]   # relative order, seats[0] = acting player
    bonuses = [p.bonuses for p in seats]
    tokens = [p.tokens for p in seats]
    for noble_id in game.nobles:
        cost = _NOBLE_COST[noble_id]
        obs += _NOBLE_TOKENS[noble_id]
        obs += [_clip9(noble_missing(cost, bonuses[0])),
                _clip9(min(noble_missing(cost, bonuses[k]) for k in range(1, N_PLAYERS)))]
    for _ in range(NUM_NOBLE_SLOTS - len(game.nobles)):
        obs += _EMPTY_NOBLE_TOKENS
    for row in game.market:
        for card_id in row:
            if card_id is None:
                obs += _EMPTY_CARD_TOKENS
            else:
                cost = _CARD_COST[card_id]
                obs += _CARD_TOKENS[card_id]
                obs += [_clip9(card_shortfall(cost, bonuses[0], tokens[0])),
                        _clip9(min(card_shortfall(cost, bonuses[k], tokens[k]) for k in range(1, N_PLAYERS)))]
    for k in range(N_PLAYERS):
        p = seats[k]
        obs += p.tokens
        obs += [b if b < 15 else 15 for b in bonuses[k]]
        points = p.points
        obs += [points if points < 30 else 30, len(p.nobles), min(len(p.purchased), 31), len(p.reserved)]
        for card_id in (p.reserved + _NONE3)[:3]:
            if card_id is None:
                obs += _EMPTY_CARD_TOKENS
            else:
                obs += _CARD_TOKENS[card_id]
                obs += [_clip9(card_shortfall(_CARD_COST[card_id], bonuses[k], tokens[k])), DERIVED_EMPTY]
    return obs


def encode_observation(game):
    obs = encode_observation_list(game)
    assert len(obs) == N_OBSERVATION
    return np.asarray(obs, dtype=np.int32)


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
def legal_action_indices(game):
    """Indices of the legal actions: the set game.legal_actions(all_payments=False) maps to with action_to_index."""
    phase = game.phase
    if phase == 'noble':
        bonuses = game.players[game.current_player].bonuses
        return [OFFSET_NOBLE + j for j, n in enumerate(game.nobles)
                if all(b >= c for b, c in zip(bonuses, NOBLES[n].cost))]
    if phase == 'discard':
        tokens = game.players[game.current_player].tokens
        return [OFFSET_DISCARD + c for c in range(6) if tokens[c] > 0]
    if phase != 'action':
        return []
    p = game.players[game.current_player]
    bank = game.bank
    tokens = p.tokens
    held = sum(tokens)
    no_cap = game.discard_over_limit
    result = []
    available = tuple(c for c in range(5) if bank[c] > 0)
    count = len(available) if len(available) < 3 else 3
    if count and (no_cap or held + count <= 10):
        if count == 3:
            result += [_TAKE3_INDEX[cs] for cs in combinations(available, 3)]
        elif count == 2:
            result.append(_TAKE2_INDEX[available])
        else:
            result.append(OFFSET_TAKE1 + available[0])
    if no_cap or held + 2 <= 10:
        result += [OFFSET_TAKE_DOUBLE + c for c in range(5) if bank[c] >= 4]
    market = game.market
    if len(p.reserved) < 3 and (no_cap or held + (bank[5] > 0) <= 10):
        result += [OFFSET_RESERVE + i for i in range(12) if market[i >> 2][i & 3] is not None]
    bonuses = p.bonuses
    gold = tokens[5]
    for i in range(12):
        card_id = market[i >> 2][i & 3]
        if card_id is not None and _affordable(CARDS[card_id].cost, bonuses, tokens, gold):
            result.append(OFFSET_BUY_MARKET + i)
    for i, card_id in enumerate(p.reserved):
        if _affordable(CARDS[card_id].cost, bonuses, tokens, gold):
            result.append(OFFSET_BUY_RESERVED + i)
    return result


def _affordable(cost, bonuses, tokens, gold):
    """game.least_gold_payment(card) is not None, without building the payment"""
    need = 0
    for i in range(5):
        c = cost[i] - bonuses[i] - tokens[i]
        if c > 0:
            need += c
    return need <= gold


def index_to_action(game, idx):
    """engine Action of legal action index idx in this state (purchases use the least-gold payment)"""
    if idx < OFFSET_TAKE2:
        return Action('take', colors=TAKE3_COMBOS[idx - OFFSET_TAKE3])
    if idx < OFFSET_TAKE1:
        return Action('take', colors=TAKE2_COMBOS[idx - OFFSET_TAKE2])
    if idx < OFFSET_TAKE_DOUBLE:
        return Action('take', colors=(idx - OFFSET_TAKE1,))
    if idx < OFFSET_RESERVE:
        c = idx - OFFSET_TAKE_DOUBLE
        return Action('take', colors=(c, c))
    if idx < OFFSET_BUY_MARKET:
        i = idx - OFFSET_RESERVE
        return Action('reserve', card_id=game.market[i >> 2][i & 3])
    if idx < OFFSET_BUY_RESERVED:
        i = idx - OFFSET_BUY_MARKET
        card_id = game.market[i >> 2][i & 3]
        return Action('buy', card_id=card_id, payment=game.least_gold_payment(card_id))
    if idx < OFFSET_NOBLE:
        card_id = game.players[game.current_player].reserved[idx - OFFSET_BUY_RESERVED]
        return Action('buy', card_id=card_id, payment=game.least_gold_payment(card_id))
    if idx < OFFSET_DISCARD:
        return Action('noble', noble_id=game.nobles[idx - OFFSET_NOBLE])
    return Action('discard', colors=(idx - OFFSET_DISCARD,))


def action_to_index(game, action):
    if action.kind == 'take':
        cs = tuple(action.colors)
        if len(cs) == 3:
            return _TAKE3_INDEX[cs]
        if len(cs) == 2 and cs[0] == cs[1]:
            return OFFSET_TAKE_DOUBLE + cs[0]
        if len(cs) == 2:
            return _TAKE2_INDEX[cs]
        return OFFSET_TAKE1 + cs[0]
    if action.kind == 'noble':
        return OFFSET_NOBLE + game.nobles.index(action.noble_id)
    if action.kind == 'discard':
        return OFFSET_DISCARD + action.colors[0]
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

    __slots__ = ('max_turns', 'discard_over_limit', 'rng', 'game', 'truncated', '_legal', '_mask')

    def __init__(self, max_turns=400, seed=None, discard_over_limit=True):
        self.max_turns = max_turns
        self.discard_over_limit = discard_over_limit
        self.rng = random.Random(seed)
        self.game = None
        self.truncated = False
        self._legal = None    # cached legal action indices of the current state
        self._mask = None     # cached legal mask as bytes

    # ----- lifecycle -----
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.game = new_game(self.rng, discard_over_limit=self.discard_over_limit)
        self.truncated = False
        self._legal = self._mask = None
        return self.observation()

    def _copy(self, rng):
        env = SplendorRLEnv.__new__(SplendorRLEnv)
        env.max_turns = self.max_turns
        env.discard_over_limit = self.discard_over_limit
        env.rng = rng
        env.game = self.game.clone()
        env.truncated = self.truncated
        env._legal = self._legal
        env._mask = self._mask
        return env

    def copy_for_search(self):
        """Private copy of the state for one MCTS simulation. The search resolves card draws itself
        (step(..., sample_chance=False) + resolve_chance), so the copy shares this env's RNG
        instead of seeding a new one every simulation."""
        return self._copy(self.rng)

    def new_random(self, seed=None):
        """Same public state, independent RNG for future (hidden) deck draws."""
        return self._copy(random.Random(seed))

    # ----- state queries -----
    @property
    def acting_player(self):
        return self.game.current_player

    @property
    def terminated(self):
        """the game is over: ended by the rules (score / draw) or truncated at max_turns"""
        return self.truncated or self.game.phase == 'terminal'

    @property
    def end_reason(self):
        if self.truncated:
            return 'truncated'
        return self.game.end_reason

    def key(self):
        """Transposition key for the search: the engine's Markov key, plus what the action indices and the
        observation depend on beyond it (the acting player's reserved order: buy-reserved actions are
        indexed by position; the noble order: noble actions are indexed by slot) and the turn (round field
        of the observation, truncation)."""
        g = self.game
        return g.key(), tuple(g.players[g.current_player].reserved), tuple(g.nobles), g.turn

    def observation(self):
        return encode_observation(self.game)

    def observation_bytes(self):
        return bytes(encode_observation_list(self.game))

    def legal_indices(self):
        """legal actions of the state; a truncated state keeps its legal actions (the search evaluates it with
        the network, which needs the mask), only a state ended by the rules has none"""
        if self._legal is None:
            self._legal = [] if self.game.phase == 'terminal' else legal_action_indices(self.game)
        return self._legal

    def legal_mask_bytes(self):
        if self._mask is None:
            mask = bytearray(N_ACTIONS)
            for i in self.legal_indices():
                mask[i] = 1
            self._mask = bytes(mask)
        return self._mask

    def legal_mask(self):
        return np.frombuffer(self.legal_mask_bytes(), dtype=np.uint8).astype(bool)

    def legal_actions(self):
        """dict: action index -> engine Action (least-gold payment for purchases)."""
        return {i: index_to_action(self.game, i) for i in self.legal_indices()}

    # ----- transition -----
    def step(self, action_idx, sample_chance=True, check=True):
        """Apply a player action.

        sample_chance=True: a replacement draw is sampled immediately with self.rng (real games).
        sample_chance=False: stop at the chance node (``at_chance_node``) so the caller (MCTS)
        picks the drawn card itself with ``chance_pool()`` / ``resolve_chance()``.
        check=False: the caller guarantees the action is legal (MCTS takes it from the node's legal mask
        of this very state), so the legal actions are not recomputed.
        """
        if check and not (0 <= action_idx < N_ACTIONS and self.legal_mask_bytes()[action_idx]):
            raise ValueError(f'illegal action {action_idx}')
        # the action is legal in this exact state: skip the engine's re-check
        self.game.step(index_to_action(self.game, action_idx), self.rng if sample_chance else None, validate=False)
        self._after_transition()
        return self.terminated

    @property
    def at_chance_node(self):
        return self.game.phase == 'chance'

    def chance_outcomes(self):
        """[(card_id, probability)] of the pending replacement draw."""
        return self.game.chance_outcomes()

    def chance_pool(self):
        """the cards the pending replacement draw picks from uniformly (same order as chance_outcomes)"""
        return self.game.remaining[self.game.pending_refill[0]]

    def resolve_chance(self, card_id):
        self.game.resolve_chance(card_id)
        self._after_transition()
        return self.terminated

    def _after_transition(self):
        self._legal = self._mask = None
        if not self.terminated and not self.at_chance_node and self.game.turn >= self.max_turns:
            self.truncated = True

    def final_rewards(self):
        """Absolute-seat rewards: win shares of a game ended by the rules (a draw: 0.25 each); all zero for a
        truncated game, which has no result (see value_target)."""
        if self.game.phase == 'terminal':
            return np.asarray(self.game.rewards(), dtype=np.float32)
        return np.zeros(N_PLAYERS, dtype=np.float32)

    def value_target(self):
        """(absolute-seat value target, value weight): a truncated game has no result, so its positions get
        no value supervision (weight 0; their policy targets are still good)"""
        return self.final_rewards(), (1. if self.game.phase == 'terminal' else 0.)


def to_relative(values, seat):
    """Absolute seat order -> order relative to `seat` (index 0 = seat)."""
    return np.roll(np.asarray(values), -seat)


def to_absolute(values, seat):
    """Order relative to `seat` -> absolute seat order."""
    return np.roll(np.asarray(values), seat)
