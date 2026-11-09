"""What a camera can see of a physical table, and how it relates to the engine state.

TableState is the observable part of a game: the bank, the 3x4 market, the nobles on the table, and per seat the
tokens, the purchased cards counted by bonus colour, the (public) reserved cards and the nobles. A vision model
produces one (normalize_vision); the engine state gives the expected one (table_of). Fields the vision model could
not read are None (a whole field) or UNKNOWN (one market slot): they match anything.

explain(game, observed) finds the legal ways the current player's turn can lead from the engine state to the
observed table: an action, the card turned up to refill the market (read from the table), returned tokens (from
the token difference) and a chosen noble. A turn may also be observed half done (tokens still to return, a market
slot not refilled yet, a noble not chosen yet).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..engine import Action, CARDS, COLORS, NOBLES

UNKNOWN = "?"
TOKEN_KEYS = COLORS + ("gold",)
CN = ("白", "蓝", "绿", "红", "黑", "金")
TOTALS = (7, 7, 7, 7, 7, 5)

COLOR_ALIASES = {
    "white": 0, "diamond": 0, "w": 0, "白": 0, "白色": 0, "钻石": 0,
    "blue": 1, "sapphire": 1, "u": 1, "b": 1, "蓝": 1, "蓝色": 1, "蓝宝石": 1,
    "green": 2, "emerald": 2, "g": 2, "绿": 2, "绿色": 2, "祖母绿": 2,
    "red": 3, "ruby": 3, "r": 3, "红": 3, "红色": 3, "红宝石": 3,
    "black": 4, "onyx": 4, "k": 4, "黑": 4, "黑色": 4, "玛瑙": 4, "brown": 4,
    "gold": 5, "joker": 5, "yellow": 5, "金": 5, "金色": 5, "黄金": 5,
}
CARD_BY_FEATURES = {(c.level, c.bonus, c.points, c.cost): cid for cid, c in CARDS.items()}
NOBLE_BY_COST = {n.cost: nid for nid, n in NOBLES.items()}


# ---------------------------------------------------------------------------------------------- text
def tokens_text(counts, empty="无"):
    parts = [f"{CN[c]}×{n}" for c, n in enumerate(counts) if n]
    return "、".join(parts) if parts else empty


def cost_text(cost):
    return " ".join(f"{CN[c]}{n}" for c, n in enumerate(cost) if n) or "免费"


def card_text(card_id):
    if card_id is None:
        return "空位"
    if card_id == UNKNOWN:
        return "未识别"
    c = CARDS[card_id]
    return f"{c.level}级{CN[c.bonus]}牌·{c.points}分（{cost_text(c.cost)}）"


def noble_text(noble_id):
    return f"贵族（{cost_text(NOBLES[noble_id].cost)}）"


def slot_text(level, slot):
    return f"{level + 1}级第{slot + 1}张"


# ---------------------------------------------------------------------------------------------- state
@dataclass
class SeatState:
    tokens: list | None = None      # 6 counts, gold last
    bonuses: list | None = None     # 5 counts: purchased cards by bonus colour
    reserved: list | None = None    # card ids (public reserves), sorted
    nobles: list | None = None      # noble ids, sorted


@dataclass
class TableState:
    bank: list | None = None
    market: list | None = None      # 3 rows x 4 slots: card id, None (empty slot) or UNKNOWN
    nobles: list | None = None      # noble ids on the table, sorted
    seats: list = field(default_factory=lambda: [SeatState() for _ in range(4)])
    visible: bool = True
    issues: list = field(default_factory=list)
    confidence: float | None = None

    def key(self):
        """canonical form of the readable content (what must stay stable between two readings)"""
        return json.dumps([self.bank, self.market, self.nobles,
                           [[s.tokens, s.bonuses, s.reserved, s.nobles] for s in self.seats]])

    def to_dict(self):
        return {"bank": self.bank, "market": self.market, "nobles": self.nobles,
                "seats": [vars(s) for s in self.seats], "visible": self.visible, "issues": self.issues,
                "confidence": self.confidence}


def table_of(game):
    """the table the engine state looks like (during a refill the slot is empty, as on the real table)"""
    return TableState(
        bank=game.bank[:], market=[row[:] for row in game.market], nobles=sorted(game.nobles),
        seats=[SeatState(p.tokens[:], p.bonuses, sorted(p.reserved), sorted(p.nobles)) for p in game.players])


def _field_diff(label, expected, observed, fmt=str):
    if observed is None or expected == observed:
        return []
    return [f"{label}：应为 {fmt(expected)}，识别为 {fmt(observed)}"]


def diff(expected, observed, names=None):
    """human-readable differences (Chinese), only for the fields the observation could read"""
    names = names or [f"玩家{i + 1}" for i in range(4)]
    out = []
    if observed.bank is not None:
        for c in range(6):
            if expected.bank[c] != observed.bank[c]:
                out.append(f"银行{CN[c]}：应为 {expected.bank[c]}，识别为 {observed.bank[c]}")
    if observed.market is not None:
        for level in range(3):
            for slot in range(4):
                e, o = expected.market[level][slot], observed.market[level][slot]
                if o != UNKNOWN and e != o:
                    out.append(f"{slot_text(level, slot)}：应为 {card_text(e)}，识别为 {card_text(o)}")
    out += _field_diff("桌面贵族", expected.nobles, observed.nobles, lambda v: "、".join(map(noble_text, v)) or "无")
    for i, (e, o) in enumerate(zip(expected.seats, observed.seats)):
        if o.tokens is not None:
            for c in range(6):
                if e.tokens[c] != o.tokens[c]:
                    out.append(f"{names[i]} 筹码{CN[c]}：应为 {e.tokens[c]}，识别为 {o.tokens[c]}")
        if o.bonuses is not None:
            for c in range(5):
                if e.bonuses[c] != o.bonuses[c]:
                    out.append(f"{names[i]} {CN[c]}色发展卡：应为 {e.bonuses[c]} 张，识别为 {o.bonuses[c]} 张")
        out += _field_diff(f"{names[i]} 预留牌", e.reserved, o.reserved,
                           lambda v: "、".join(card_text(c) for c in v) or "无")
        out += _field_diff(f"{names[i]} 贵族", e.nobles, o.nobles, lambda v: "、".join(map(noble_text, v)) or "无")
    return out


def matches(expected, observed):
    if observed.bank is not None and observed.bank != expected.bank:
        return False
    if observed.market is not None:
        for level in range(3):
            for slot in range(4):
                o = observed.market[level][slot]
                if o != UNKNOWN and o != expected.market[level][slot]:
                    return False
    if observed.nobles is not None and observed.nobles != expected.nobles:
        return False
    for e, o in zip(expected.seats, observed.seats):
        for name in ("tokens", "bonuses", "reserved", "nobles"):
            value = getattr(o, name)
            if value is not None and value != getattr(e, name):
                return False
    return True


def validate(observed, current_player=None):
    """rule violations visible in one reading alone (wrong totals, a card twice, ...): a misreading or a bad table"""
    problems = []
    if observed.bank is not None and all(s.tokens is not None for s in observed.seats):
        for c in range(6):
            total = observed.bank[c] + sum(s.tokens[c] for s in observed.seats)
            if total != TOTALS[c]:
                problems.append(f"{CN[c]}筹码总数为 {total}，应为 {TOTALS[c]}（银行加四位玩家）")
    if observed.bank is not None and any(v < 0 for v in observed.bank):
        problems.append("银行筹码数为负")
    seen = []
    if observed.market is not None:
        seen += [c for row in observed.market for c in row if c not in (None, UNKNOWN)]
    for i, s in enumerate(observed.seats):
        if s.reserved is not None:
            seen += s.reserved
            if len(s.reserved) > 3:
                problems.append(f"玩家{i + 1} 预留了 {len(s.reserved)} 张牌（最多 3 张）")
        if s.tokens is not None and sum(s.tokens) > 10 and i != current_player:
            problems.append(f"玩家{i + 1} 持有 {sum(s.tokens)} 枚筹码（回合结束时最多 10 枚）")
    duplicated = sorted({c for c in seen if seen.count(c) > 1})
    for c in duplicated:
        problems.append(f"同一张牌出现了两次：{card_text(c)}")
    nobles = list(observed.nobles or []) + [n for s in observed.seats for n in (s.nobles or [])]
    if len(nobles) != len(set(nobles)):
        problems.append("同一位贵族出现了两次")
    if observed.nobles is not None and all(s.nobles is not None for s in observed.seats) and len(set(nobles)) != 5:
        problems.append(f"贵族总数为 {len(set(nobles))}，应为 5")
    return problems


# ---------------------------------------------------------------------------------------------- vision -> state
def _color(value):
    if value is None:
        return None
    return COLOR_ALIASES.get(str(value).strip().lower())


def _counts(value, n):
    """{"white": 2, ...} (any colour aliases) or a list -> list of n ints; None if unreadable"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) < n or not all(isinstance(v, (int, float)) for v in value[:n]):
            return None
        return [max(0, int(v)) for v in value[:n]]
    if not isinstance(value, dict):
        return None
    out = [0] * n
    for k, v in value.items():
        c = _color(k)
        if c is None or c >= n:
            continue
        try:
            out[c] += max(0, int(v))
        except (TypeError, ValueError):
            return None
    return out


def match_card(desc, level=None, exclude=()):
    """(card id or UNKNOWN or None, note). desc: {"color", "points", "cost": {...}, "level"?}; None = empty slot"""
    if desc is None or desc == "empty" or (isinstance(desc, dict) and desc.get("empty")):
        return None, None
    if not isinstance(desc, dict):
        return UNKNOWN, None
    level = int(desc.get("level") or level or 0)
    bonus = _color(desc.get("color") or desc.get("bonus"))
    cost = _counts(desc.get("cost"), 5)
    try:
        points = int(desc.get("points") or 0)
    except (TypeError, ValueError):
        points = 0
    if bonus is None or cost is None or level not in (1, 2, 3):
        return UNKNOWN, None
    exact = CARD_BY_FEATURES.get((level, bonus, points, tuple(cost)))
    if exact and exact not in exclude:
        return exact, None
    best, best_d = None, None
    for cid, c in CARDS.items():
        if c.level != level or c.bonus != bonus or cid in exclude:
            continue
        d = sum(abs(a - b) for a, b in zip(c.cost, cost)) + 2 * abs(c.points - points)
        if best_d is None or d < best_d:
            best, best_d = cid, d
    if best is None or best_d > 3:
        return UNKNOWN, f"读到的牌（{level}级{CN[bonus]}·{points}分·{cost_text(cost)}）不在牌库中"
    return best, f"读到的牌（{level}级{CN[bonus]}·{points}分·{cost_text(cost)}）按最接近的 {card_text(best)} 处理"


def match_noble(desc):
    cost = _counts(desc.get("cost", desc) if isinstance(desc, dict) else desc, 5)
    if cost is None:
        return None, "贵族要求无法识别"
    exact = NOBLE_BY_COST.get(tuple(cost))
    if exact:
        return exact, None
    best = min(NOBLES, key=lambda n: sum(abs(a - b) for a, b in zip(NOBLES[n].cost, cost)))
    return best, f"贵族要求（{cost_text(cost)}）按最接近的 {noble_text(best)} 处理"


def normalize_vision(data):
    """vision model JSON (see vision.SCHEMA_TEXT) -> TableState; unreadable parts become None / UNKNOWN"""
    if not isinstance(data, dict):
        raise ValueError("识别结果不是 JSON 对象")
    t = TableState(seats=[SeatState() for _ in range(4)])
    t.visible = bool(data.get("visible", True))
    t.issues = [str(x) for x in (data.get("issues") or []) if x]
    try:
        t.confidence = None if data.get("confidence") is None else float(data["confidence"])
    except (TypeError, ValueError):
        t.confidence = None
    t.bank = _counts(data.get("bank"), 6)
    market = data.get("market")
    if isinstance(market, dict):
        rows = [market.get(str(level)) or market.get(level) or market.get(f"level{level}") for level in (1, 2, 3)]
    elif isinstance(market, list) and len(market) == 3:
        rows = market
    else:
        rows = None
    if rows is not None and all(isinstance(r, list) for r in rows):
        t.market = []
        for level, row in enumerate(rows):
            cells = list(row[:4]) + [UNKNOWN] * (4 - len(row[:4]))
            out = []
            for cell in cells:
                if cell == UNKNOWN:
                    out.append(UNKNOWN)
                    continue
                cid, note = match_card(cell, level + 1)
                if note:
                    t.issues.append(note)
                out.append(cid)
            t.market.append(out)
    if isinstance(data.get("nobles"), list):
        ids = []
        for desc in data["nobles"]:
            nid, note = match_noble(desc)
            if note:
                t.issues.append(note)
            if nid:
                ids.append(nid)
        t.nobles = sorted(ids)
    players = data.get("players")
    if isinstance(players, list):
        for k, p in enumerate(players[:4]):
            if not isinstance(p, dict):
                continue
            seat = p.get("seat", k + 1)
            try:
                i = int(seat) - 1
            except (TypeError, ValueError):
                i = k
            if not 0 <= i < 4:
                continue
            s = t.seats[i]
            s.tokens = _counts(p.get("tokens"), 6)
            s.bonuses = _counts(p.get("cards") or p.get("bonuses"), 5)
            if isinstance(p.get("reserved"), list):
                ids = []
                for desc in p["reserved"]:
                    cid, note = match_card(desc)
                    if note:
                        t.issues.append(f"玩家{i + 1} 预留牌：{note}")
                    if cid not in (None, UNKNOWN):
                        ids.append(cid)
                    elif cid == UNKNOWN:
                        ids = None
                        break
                s.reserved = None if ids is None else sorted(ids)
            if isinstance(p.get("nobles"), list):
                ids = []
                for desc in p["nobles"]:
                    nid, note = match_noble(desc)
                    if note:
                        t.issues.append(f"玩家{i + 1}：{note}")
                    if nid:
                        ids.append(nid)
                s.nobles = sorted(ids)
    return t


# ---------------------------------------------------------------------------------------------- explaining a change
@dataclass
class Candidate:
    actions: list           # the player's actions (engine Action), in order
    reveals: list           # (level, slot, card id) turned up from the decks
    game: object            # the engine state after them
    status: str             # complete | await_refill | await_discard | await_noble
    note: str = ""

    def key(self):
        return (self.status, self.game.key())


def explain(game, observed, max_candidates=64):
    """the legal ways the current player's turn leads to the observed table (see module doc)"""
    if game.phase == "terminal":
        return []
    player, turn = game.current_player, game.turn
    start_nobles = list(game.players[player].nobles)
    out = []

    def finished(g):
        return g.phase == "terminal" or g.current_player != player or g.turn != turn

    def record(g, actions, reveals, status):
        expected = table_of(g)
        if matches(expected, observed):
            out.append(Candidate(actions, reveals, g, status))
            return
        # a noble the engine awarded automatically may still lie on the table
        new = [n for n in g.players[player].nobles if n not in start_nobles]
        if status == "complete" and new:
            unmoved = table_of(g)
            unmoved.nobles = sorted(unmoved.nobles + new)
            unmoved.seats[player].nobles = sorted(start_nobles)
            if matches(unmoved, observed):
                out.append(Candidate(actions, reveals, g, status, note="noble_unmoved"))

    def rec(g, actions, reveals, depth):
        if len(out) >= max_candidates or depth > 12:
            return
        if finished(g):
            record(g, actions, reveals, "complete")
            return
        if g.phase == "chance":
            level, slot = g.pending_refill
            seen = None if observed.market is None else observed.market[level][slot]
            if observed.market is None or seen == UNKNOWN:
                return
            if seen is None:
                record(g, actions, reveals, "await_refill")
                return
            if seen not in g.remaining[level]:
                return
            g2 = g.clone()
            g2.resolve_chance(seen)
            rec(g2, actions, reveals + [(level, slot, seen)], depth + 1)
            return
        if g.phase == "discard":
            have, seen = g.players[player].tokens, observed.seats[player].tokens
            if seen is None:
                return
            d = [a - b for a, b in zip(have, seen)]
            if any(x < 0 for x in d) or sum(d) > sum(have) - 10:
                return
            if sum(d) == 0:
                record(g, actions, reveals, "await_discard")
                return
            g2, acts = g.clone(), []
            for c in range(6):
                for _ in range(d[c]):
                    a = Action("discard", colors=(c,))
                    g2.step(a)
                    acts.append(a)
            rec(g2, actions + acts, reveals, depth + 1)
            return
        if g.phase == "noble":
            seen = observed.seats[player].nobles
            got = [] if seen is None else [n for n in seen if n not in g.players[player].nobles]
            choices = [n for n in g.eligible_nobles() if n in got]
            if not choices:
                record(g, actions, reveals, "await_noble")
                return
            for n in choices:
                a = Action("noble", noble_id=n)
                rec(g.apply(a), actions + [a], reveals, depth + 1)
            return
        if actions:     # the action phase only starts a turn
            return
        for a in g.legal_actions(all_payments=True):
            rec(g.apply(a), [a], reveals, depth + 1)

    rec(game, [], [], 0)
    unique, seen_keys = [], set()
    for cand in out:
        if cand.key() not in seen_keys:
            seen_keys.add(cand.key())
            unique.append(cand)
    return unique


# ---------------------------------------------------------------------------------------------- describing actions
def action_text(game, action, names=None):
    """Chinese instruction / history text of an action of the current player in `game` (the state before it)"""
    names = names or [f"玩家{i + 1}" for i in range(4)]
    if action.kind == "take":
        counts = [0] * 6
        for c in action.colors:
            counts[c] += 1
        return f"从银行拿取 {tokens_text(counts)}"
    if action.kind == "discard":
        return f"退回 1 枚{CN[action.colors[0]]}筹码到银行"
    if action.kind == "noble":
        return f"选择{noble_text(action.noble_id)}，放到自己面前"
    where = next((slot_text(level, slot) for level, row in enumerate(game.market)
                  for slot, c in enumerate(row) if c == action.card_id), None)
    if action.kind == "reserve":
        gold = "，并从银行拿 1 枚黄金" if game.bank[5] else "（黄金已空，不拿黄金）"
        return f"预留桌面{where}：{card_text(action.card_id)}，正面朝上放在自己面前{gold}"
    if action.kind == "buy":
        source = f"桌面{where}" if where else "自己的预留牌"
        pay = tokens_text(action.payment, empty="不需要筹码")
        return f"购买{source}：{card_text(action.card_id)}；支付 {pay}（放回银行），牌放到自己面前"
    return str(action.to_dict())
