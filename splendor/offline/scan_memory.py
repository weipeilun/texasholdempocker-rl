"""Merge readable fields across views of one stationary physical table."""
from __future__ import annotations

from .table import TOTALS, UNKNOWN, SeatState, TableState, validate

SEAT_FIELDS = ("tokens", "bonuses", "reserved", "nobles")
SCAN_SECONDS = 300  # long enough for one measured ten-view sweep


class ScanMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scope = None
        self.started = 0.
        self.values = {}

    def prepare(self, scope, stamp):
        if scope != self.scope or stamp - self.started > SCAN_SECONDS:
            self.reset()
            self.scope, self.started = scope, stamp

    def missing(self):
        fields = {}
        market = [f"{level + 1}级第{slot + 1}张" for level in range(3) for slot in range(4)
                  if ("market", level, slot) not in self.values]
        if ("nobles",) not in self.values:
            market.append("桌面贵族")
        if market:
            fields["market"] = market
        for seat in range(4):
            names = {"tokens": "筹码", "bonuses": "已购牌", "reserved": "预留牌", "nobles": "贵族"}
            needs = [names[field] for field in SEAT_FIELDS if ("seat", seat, field) not in self.values]
            if needs:
                fields[f"seat{seat + 1}"] = needs
        return fields

    def merge(self, observed, coverage, current_player, *, stamp):
        found = {}
        for level, row in enumerate(observed.market or []):
            for slot, card in enumerate(row):
                if card != UNKNOWN and (card is not None or coverage["market"] in ("clear", "unreadable")):
                    found["market", level, slot] = card
        if observed.nobles is not None and coverage["market"] in ("clear", "unreadable"):
            found["nobles",] = observed.nobles
        for i, seat in enumerate(observed.seats):
            if coverage[f"seat{i + 1}"] not in ("clear", "unreadable"):
                continue
            for field in SEAT_FIELDS:
                value = getattr(seat, field)
                if value is not None:
                    found["seat", i, field] = value

        # A newly visible conflict starts a new snapshot. An obscured value never erases a good read.
        conflict = any(key in self.values and value != self.values[key] for key, value in found.items())
        if conflict:
            self.started, self.values = stamp, {}
        self.values.update(found)
        table = self._table()
        problems = validate(table, current_player=current_player)
        if problems and self.values != found:
            self.started, self.values = stamp, found
            table = self._table()
            problems = validate(table, current_player=current_player)
            conflict = True

        missing = self.missing()
        if conflict:
            table.issues.append("复读到与先前不同的桌面信息，已重新采集")
        table.issues.extend(problems)
        table.issues.extend(f"{'公共区' if region == 'market' else '玩家' + region[4:]}待复读：{'、'.join(fields)}"
                            for region, fields in missing.items())
        if not found and not missing:
            table.issues.append("当前画面未复核到任何已确认字段")
        table.visible = bool(found) and not missing and not problems and not conflict and table.bank is not None
        # Keep concrete single-frame feedback while some regions still need a view.
        if missing:
            table.issues.extend(observed.issues[:4])
        merged_coverage = {region: state if region in missing else "retained" if state != "clear" else "clear"
                           for region, state in coverage.items()}
        return table, merged_coverage

    def _table(self):
        values = self.values
        seats = [SeatState(**{field: values.get(("seat", i, field)) for field in SEAT_FIELDS}) for i in range(4)]
        bank = None
        if all(seat.tokens is not None for seat in seats):
            totals = [total - sum(seat.tokens[i] for seat in seats) for i, total in enumerate(TOTALS)]
            if min(totals) >= 0:
                bank = totals
        return TableState(
            bank=bank, nobles=values.get(("nobles",)), seats=seats,
            market=[[values.get(("market", level, slot), UNKNOWN) for slot in range(4)] for level in range(3)],
        )
