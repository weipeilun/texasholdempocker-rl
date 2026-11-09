"""A simulated camera for tests and demos: the vision-model JSON a perfect reader would return for an engine state."""

from __future__ import annotations

from ..engine import CARDS, COLORS, NOBLES
from .table import TOKEN_KEYS


def card_desc(card_id):
    c = CARDS[card_id]
    return {"level": c.level, "color": COLORS[c.bonus], "points": c.points, "cost": dict(zip(COLORS, c.cost))}


def noble_desc(noble_id):
    return {"cost": dict(zip(COLORS, NOBLES[noble_id].cost))}


def vision_json(game):
    return {
        "visible": True, "issues": [], "confidence": 1.0,
        "bank": dict(zip(TOKEN_KEYS, game.bank)),
        "nobles": [noble_desc(n) for n in game.nobles],
        "market": {str(level + 1): [None if c is None else card_desc(c) for c in row]
                   for level, row in enumerate(game.market)},
        "players": [{"seat": i + 1, "tokens": dict(zip(TOKEN_KEYS, p.tokens)), "cards": dict(zip(COLORS, p.bonuses)),
                     "reserved": [card_desc(c) for c in p.reserved], "nobles": [noble_desc(n) for n in p.nobles]}
                    for i, p in enumerate(game.players)],
    }
