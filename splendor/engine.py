"""Dependency-free rules engine. Randomness belongs to the caller, never the state.

Token limit (Game.discard_over_limit):
    True  (default, official rule): a player may go above 10 tokens by taking or reserving and must
          then return tokens one at a time ("discard" phase, any colour incl. gold and the ones just
          taken) until holding 10, before nobles are checked and the turn ends.
    False (legacy house rule): any action that would exceed 10 tokens is illegal; no returning.

Use apply(action, rng) for a sampled transition, or apply(action) to stop at a
chance node and enumerate chance_outcomes()/resolve_chance(card_id). Search
branches must supply independent RNGs; cloning a state cannot reveal a deck order.
"""

from __future__ import annotations

import copy
import csv
import random
from dataclasses import asdict, dataclass, field
from itertools import combinations, product
from pathlib import Path

COLORS = ("white", "blue", "green", "red", "black")
TOKEN_NAMES = COLORS + ("gold",)
LABELS = ("白", "蓝", "绿", "红", "黑", "金")


@dataclass(frozen=True)
class Card:
    id: str
    level: int
    bonus: int
    points: int
    cost: tuple[int, ...]


@dataclass(frozen=True)
class Noble:
    id: str
    cost: tuple[int, ...]
    points: int = 3


def _load_cards():
    with (Path(__file__).parent / "data" / "cards.csv").open() as source:
        return {
            row["id"]: Card(row["id"], int(row["id"].split("-L")[1][0]),
                            COLORS.index(row["id"].split("-")[0]),
                            int(row["prestige_points"]),
                            tuple(int(row[c]) for c in COLORS))
            for row in csv.DictReader(source)
        }


CARDS = _load_cards()
# Five adjacent pairs and five adjacent triples on the W-B-G-R-K color cycle.
NOBLES = {
    f"N{i + 1}": Noble(f"N{i + 1}", tuple(values))
    for i, values in enumerate([
        (4, 4, 0, 0, 0), (0, 4, 4, 0, 0), (0, 0, 4, 4, 0),
        (0, 0, 0, 4, 4), (4, 0, 0, 0, 4),
        (3, 3, 3, 0, 0), (0, 3, 3, 3, 0), (0, 0, 3, 3, 3),
        (3, 0, 0, 3, 3), (3, 3, 0, 0, 3),
    ])
}


@dataclass(frozen=True)
class Action:
    kind: str
    colors: tuple[int, ...] = ()
    card_id: str = ""
    payment: tuple[int, ...] = ()
    noble_id: str = ""

    @classmethod
    def from_dict(cls, value):
        return cls(value["kind"], tuple(value.get("colors", ())),
                   value.get("card_id", ""), tuple(value.get("payment", ())),
                   value.get("noble_id", ""))

    def to_dict(self):
        return asdict(self)


@dataclass
class Player:
    tokens: list[int] = field(default_factory=lambda: [0] * 6)
    purchased: list[str] = field(default_factory=list)
    reserved: list[str] = field(default_factory=list)
    nobles: list[str] = field(default_factory=list)

    @property
    def bonuses(self):
        return [sum(CARDS[c].bonus == color for c in self.purchased) for color in range(5)]

    @property
    def points(self):
        return sum(CARDS[c].points for c in self.purchased) + 3 * len(self.nobles)


@dataclass
class Game:
    players: list[Player] = field(default_factory=lambda: [Player() for _ in range(4)])
    bank: list[int] = field(default_factory=lambda: [7, 7, 7, 7, 7, 5])
    market: list[list[str | None]] = field(default_factory=lambda: [[None] * 4 for _ in range(3)])
    remaining: list[list[str]] = field(default_factory=lambda: [[], [], []])
    nobles: list[str] = field(default_factory=list)
    current_player: int = 0
    turn: int = 0
    phase: str = "action"
    pending_refill: tuple[int, int] | None = None
    end_reason: str | None = None
    winners: list[int] = field(default_factory=list)
    discard_over_limit: bool = True

    def clone(self):
        """Independent copy (explicit field copy; several times faster than copy.deepcopy)."""
        return Game(
            players=[Player(p.tokens[:], p.purchased[:], p.reserved[:], p.nobles[:]) for p in self.players],
            bank=self.bank[:], market=[row[:] for row in self.market],
            remaining=[pool[:] for pool in self.remaining], nobles=self.nobles[:],
            current_player=self.current_player, turn=self.turn, phase=self.phase,
            pending_refill=self.pending_refill, end_reason=self.end_reason,
            winners=self.winners[:], discard_over_limit=self.discard_over_limit)

    def key(self):
        """Canonical Markov key for transpositions; no turn history or RNG state."""
        return (
            tuple(self.bank), tuple(tuple(row) for row in self.market),
            tuple(tuple(sorted(pool)) for pool in self.remaining), tuple(sorted(self.nobles)),
            tuple((tuple(p.tokens), tuple(sorted(p.purchased)), tuple(sorted(p.reserved)),
                   tuple(sorted(p.nobles))) for p in self.players),
            self.current_player, self.phase, self.pending_refill,
            self.end_reason, tuple(self.winners),
        )

    def payments(self, card_id, bonuses=None):
        """All distinct legal payments, including voluntary gold substitution."""
        p = self.players[self.current_player]
        bonuses = p.bonuses if bonuses is None else bonuses
        cost = [max(0, c - b) for c, b in zip(CARDS[card_id].cost, bonuses)]
        minimum_gold = sum(max(0, c - t) for c, t in zip(cost, p.tokens))
        if minimum_gold > p.tokens[5]:
            return []
        # Colored payment cannot exceed cost, inventory, or leave too much for gold.
        ranges = [range(max(0, c - p.tokens[5]), min(c, p.tokens[i]) + 1)
                  for i, c in enumerate(cost)]
        result = []
        for colored in product(*ranges):
            gold = sum(cost) - sum(colored)
            if gold <= p.tokens[5]:
                result.append(tuple(colored) + (gold,))
        return sorted(result, key=lambda payment: (payment[5], payment))

    def eligible_nobles(self):
        bonuses = self.players[self.current_player].bonuses
        return [n for n in self.nobles if all(b >= c for b, c in zip(bonuses, NOBLES[n].cost))]

    def legal_actions(self, all_payments=True):
        """all_payments=False lists only the least-gold payment of each purchase (cheaper; used by RL)."""
        if self.phase in ("chance", "terminal", "blocked"):
            return []
        if self.phase == "noble":
            return [Action("noble", noble_id=n) for n in self.eligible_nobles()]
        p = self.players[self.current_player]
        if self.phase == "discard":
            return [Action("discard", colors=(c,)) for c in range(6) if p.tokens[c] > 0]
        held = sum(p.tokens)

        def fits(extra):
            return self.discard_over_limit or held + extra <= 10

        result = []
        available = [c for c in range(5) if self.bank[c] > 0]
        # Fewer than three only when fewer than three colors remain in the bank.
        count = min(3, len(available))
        if count and fits(count):
            result.extend(Action("take", colors=cs) for cs in combinations(available, count))
        if fits(2):
            result.extend(Action("take", colors=(c, c)) for c in range(5) if self.bank[c] >= 4)
        market_ids = [c for row in self.market for c in row if c is not None]
        if len(p.reserved) < 3 and fits(int(self.bank[5] > 0)):
            result.extend(Action("reserve", card_id=c) for c in market_ids)
        bonuses = p.bonuses
        for card_id in market_ids + p.reserved:
            if all_payments:
                result.extend(Action("buy", card_id=card_id, payment=payment)
                              for payment in self.payments(card_id, bonuses))
            else:
                payment = self.least_gold_payment(card_id, bonuses)
                if payment is not None:
                    result.append(Action("buy", card_id=card_id, payment=payment))
        return result

    def least_gold_payment(self, card_id, bonuses=None):
        """payments(card_id)[0] without enumerating the others; None if unaffordable."""
        p = self.players[self.current_player]
        bonuses = p.bonuses if bonuses is None else bonuses
        cost = [max(0, c - b) for c, b in zip(CARDS[card_id].cost, bonuses)]
        colored = [min(c, t) for c, t in zip(cost, p.tokens)]
        gold = sum(cost) - sum(colored)
        if gold > p.tokens[5]:
            return None
        return tuple(colored) + (gold,)

    def chance_outcomes(self):
        if self.phase != "chance":
            return []
        level, _ = self.pending_refill
        pool = self.remaining[level]
        return [(card_id, 1 / len(pool)) for card_id in pool]

    def resolve_chance(self, card_id):
        if self.phase != "chance":
            raise ValueError("当前不是补牌节点")
        level, slot = self.pending_refill
        if card_id not in self.remaining[level]:
            raise ValueError("补牌不在该层剩余牌池中")
        self.remaining[level].remove(card_id)
        self.market[level][slot] = card_id
        self.pending_refill = None
        self._after_action()

    def sample_chance(self, rng):
        if self.phase != "chance":
            raise ValueError("当前不是补牌节点")
        level, _ = self.pending_refill
        self.resolve_chance(rng.choice(self.remaining[level]))

    def apply(self, action, rng=None):
        """Pure transition; rng=None leaves replacement draws as explicit chance nodes."""
        result = self.clone()
        result.step(action, rng)
        return result

    def step(self, action, rng=None, validate=True):
        """Validated in-place transition. Invalid actions leave this state unchanged.

        validate=False skips the legality check; only for actions taken from legal_actions() of this
        exact state (search/self-play hot path).
        """
        if validate and action not in self.legal_actions():
            raise ValueError("非法动作：库存、支付、阶段或持币上限不满足")
        p = self.players[self.current_player]
        if action.kind == "discard":
            color = action.colors[0]
            p.tokens[color] -= 1
            self.bank[color] += 1
            self._after_action()
            return
        if action.kind == "noble":
            p.nobles.append(action.noble_id)
            self.nobles.remove(action.noble_id)
            self._finish_turn()
            return
        if action.kind == "take":
            for c in action.colors:
                p.tokens[c] += 1
                self.bank[c] -= 1
        elif action.kind in ("reserve", "buy"):
            card_id = action.card_id
            if action.kind == "reserve":
                p.reserved.append(card_id)
                if self.bank[5]:
                    p.tokens[5] += 1
                    self.bank[5] -= 1
            else:
                for i, amount in enumerate(action.payment):
                    p.tokens[i] -= amount
                    self.bank[i] += amount
                p.purchased.append(card_id)
            location = next(((level, slot) for level, row in enumerate(self.market)
                             for slot, value in enumerate(row) if value == card_id), None)
            if location is None:
                p.reserved.remove(card_id)
            else:
                level, slot = location
                self.market[level][slot] = None
                if self.remaining[level]:
                    self.pending_refill = location
                    self.phase = "chance"
                    if rng is not None:
                        self.sample_chance(rng)
                    return
        self._after_action()

    def _after_action(self):
        if self.discard_over_limit and sum(self.players[self.current_player].tokens) > 10:
            # Return tokens one at a time before nobles and the end of the turn.
            self.phase = "discard"
            return
        eligible = self.eligible_nobles()
        if len(eligible) == 1:
            noble = eligible[0]
            self.players[self.current_player].nobles.append(noble)
            self.nobles.remove(noble)
        elif len(eligible) > 1:
            self.phase = "noble"
            return
        self._finish_turn()

    def _finish_turn(self):
        self.turn += 1
        # House rule: end immediately after this player's action/noble settlement.
        if self.players[self.current_player].points >= 15:
            self.end_reason = "score"
            best = max((p.points, -len(p.purchased)) for p in self.players)
            self.winners = [i for i, p in enumerate(self.players)
                            if (p.points, -len(p.purchased)) == best]
            self.phase = "terminal"
        else:
            self.current_player = (self.current_player + 1) % 4
            self.phase = "action"
            if not self.legal_actions(all_payments=False):
                # No invented pass or winner. Preserve the blocked state.
                self.phase = "blocked"
                self.end_reason = "blocked"

    def rewards(self):
        """Four-player terminal utility, shared wins split; no alternating negation."""
        if self.phase != "terminal":
            raise ValueError("游戏尚未结束")
        return [1 / len(self.winners) if i in self.winners else 0.0 for i in range(4)]

    def assert_valid(self):
        assert len(self.players) == 4 and 0 <= self.current_player < 4
        assert self.phase in ("action", "chance", "noble", "discard", "terminal", "blocked")
        assert self.phase != "discard" or self.discard_over_limit
        assert len(self.bank) == 6 and all(v >= 0 for v in self.bank)
        for i, p in enumerate(self.players):
            assert len(p.tokens) == 6 and all(v >= 0 for v in p.tokens)
            # Above 10 only for the current player between taking/reserving and returning tokens.
            over_limit_allowed = (self.discard_over_limit and i == self.current_player
                                  and self.phase in ("discard", "chance"))
            assert sum(p.tokens) <= (13 if over_limit_allowed else 10) and len(p.reserved) <= 3
        for c, total in enumerate((7, 7, 7, 7, 7, 5)):
            assert self.bank[c] + sum(p.tokens[c] for p in self.players) == total
        seen = [c for row in self.market for c in row if c is not None]
        seen += [c for pool in self.remaining for c in pool]
        seen += [c for p in self.players for c in p.purchased + p.reserved]
        assert len(seen) == 90 and set(seen) == set(CARDS)
        for level, row in enumerate(self.market):
            assert len(row) == 4
            assert all(c is None or CARDS[c].level == level + 1 for c in row)
            assert all(CARDS[c].level == level + 1 for c in self.remaining[level])
        nobles = self.nobles + [n for p in self.players for n in p.nobles]
        assert len(nobles) == len(set(nobles)) == 5 and set(nobles) <= set(NOBLES)
        assert (self.phase == "chance") == (self.pending_refill is not None)
        if self.phase == "chance":
            level, slot = self.pending_refill
            assert self.market[level][slot] is None and self.remaining[level]

    def observation(self):
        """All reserves public; remaining cards are an unordered, inferable pool."""
        return {
            "players": [dict(tokens=p.tokens[:], bonuses=p.bonuses, points=p.points,
                             purchased=p.purchased[:], reserved=p.reserved[:], nobles=p.nobles[:])
                        for p in self.players],
            "bank": self.bank[:], "market": copy.deepcopy(self.market),
            "remaining": [sorted(pool) for pool in self.remaining],
            "nobles": self.nobles[:], "current_player": self.current_player,
            "turn": self.turn, "phase": self.phase, "pending_refill": self.pending_refill,
            "end_reason": self.end_reason, "winners": self.winners[:],
            "discard_over_limit": self.discard_over_limit,
        }


def new_game(rng=None, discard_over_limit=True):
    """discard_over_limit=True: official "return tokens down to 10" rule; False: legacy hard cap."""
    rng = rng if rng is not None else random.Random()
    game = Game(discard_over_limit=discard_over_limit)
    game.remaining = [[c.id for c in CARDS.values() if c.level == level] for level in (1, 2, 3)]
    # Sample only the twelve visible initial cards, without shuffling future decks.
    for level in range(3):
        for slot in range(4):
            card_id = rng.choice(game.remaining[level])
            game.remaining[level].remove(card_id)
            game.market[level][slot] = card_id
    game.nobles = rng.sample(list(NOBLES), 5)
    game.assert_valid()
    return game
