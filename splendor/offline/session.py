"""One physical game: the engine state, what the camera last read, the robots' instructions to their human proxies,
the history, hints and alerts shown on the web pages.

Flow: setup (the 12 market cards and 5 nobles are read and confirmed) -> play. In play, every reading of the table
is compared with the engine state. A change is accepted once two readings in a row agree and it is explained by
the current player's legal turn (table.explain); for a robot seat it must also follow the robot's instruction. An
unexplained change that stays on the table becomes an alert (illegal move, misreading, a hidden part of the table).
The operator can always step in by hand: apply a legal action, name the card turned up, confirm a robot move, undo.
"""

from __future__ import annotations

import json
import os
import threading
import time

from ..engine import Action, CARDS, Game, NOBLES
from .table import (CN, UNKNOWN, Candidate, action_text, card_text, diff, explain, matches, noble_text,
                    slot_text, table_of, tokens_text, validate)

STABLE_READINGS = 2         # identical readings before a change of the table is accepted
UNEXPLAINED_READINGS = 3    # identical unexplained readings before an alert is raised


class OfflineSession:
    def __init__(self, seats, robots, log_dir=None, stable=STABLE_READINGS, unexplained=UNEXPLAINED_READINGS,
                 sync_robots=False, clock=time.time):
        """seats: 4 dicts {"name", "kind": "human" | "robot", "policy"}; robots: seat index -> robot (choose(game))"""
        if len(seats) != 4:
            raise ValueError("需要 4 个座位")
        self.lock = threading.RLock()
        self.seats = [dict(s) for s in seats]
        self.names = [s.get("name") or f"玩家{i + 1}" for i, s in enumerate(seats)]
        self.robots = robots
        self.log_dir, self.stable, self.unexplained = log_dir, stable, unexplained
        self.sync_robots, self.clock = sync_robots, clock
        self.phase = "setup"
        self.game = None
        self.history = []
        self.alerts = {}
        self.instruction = None
        self.thinking = False
        self.robot_error = None
        self.snapshots = []
        self.ambiguous = []
        self.progress = None        # what the last reading shows of a turn in progress
        self.reading, self.reading_time, self.reading_meta = None, None, None
        self._last_key, self._repeat = None, 0
        self.version = 0
        self.started = clock()
        self.log_path = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = os.path.join(log_dir, time.strftime("game-%Y%m%d-%H%M%S.json", time.localtime(self.started)))

    # ------------------------------------------------------------------------------------------ bookkeeping
    def _bump(self):
        self.version += 1

    def _event(self, kind, text, player=None, by=None, action=None):
        entry = {"t": self.clock(), "turn": None if self.game is None else self.game.turn, "kind": kind, "text": text}
        if player is not None:
            entry["player"], entry["name"] = player, self.names[player]
        if by:
            entry["by"] = by
        if action is not None:
            entry["action"] = action.to_dict()
        self.history.append(entry)

    def _alert(self, key, text, level="error", details=None):
        old = self.alerts.get(key)
        since = old["since"] if old and old["text"] == text else self.clock()
        self.alerts[key] = {"key": key, "level": level, "text": text, "details": details or [], "since": since}

    def _clear(self, *keys):
        for k in keys:
            self.alerts.pop(k, None)

    def _save(self):
        if not self.log_path:
            return
        record = {"seats": self.seats, "started": self.started, "phase": self.phase,
                  "initial": getattr(self, "initial", None), "history": self.history,
                  "result": None if self.game is None or self.game.phase != "terminal" else
                  {"end_reason": self.game.end_reason, "winners": self.game.winners,
                   "points": [p.points for p in self.game.players]}}
        tmp = self.log_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.log_path)

    def is_robot(self, player):
        return self.seats[player]["kind"] == "robot"

    # ------------------------------------------------------------------------------------------ setup
    def begin(self, market, nobles, by="operator"):
        """start the game from the 12 market cards (3 rows x 4, card ids) and the 5 nobles on the table"""
        with self.lock:
            if self.phase != "setup":
                raise ValueError("对局已经开始")
            if len(market) != 3 or any(len(row) != 4 for row in market):
                raise ValueError("桌面应为 3 行 × 4 张")
            ids = [c for row in market for c in row]
            if any(c not in CARDS for c in ids):
                raise ValueError("桌面上有未识别或不存在的牌")
            if len(set(ids)) != 12:
                raise ValueError("桌面上有重复的牌")
            for level, row in enumerate(market):
                if any(CARDS[c].level != level + 1 for c in row):
                    raise ValueError(f"第 {level + 1} 行有不是 {level + 1} 级的牌")
            if len(nobles) != 5 or len(set(nobles)) != 5 or any(n not in NOBLES for n in nobles):
                raise ValueError("需要 5 位不同的贵族")
            g = Game(discard_over_limit=True)
            g.market = [list(row) for row in market]
            g.remaining = [[c for c, card in CARDS.items() if card.level == level + 1 and c not in ids]
                           for level in range(3)]
            g.nobles = list(nobles)
            g.assert_valid()
            if not g._has_legal_action():
                raise ValueError("开局没有合法动作")
            self.game = g
            self.initial = {"market": [list(r) for r in market], "nobles": list(nobles)}
            self.phase = "play"
            self._clear("setup")
            self._event("system", f"开局（{by}确认）：{self.names[0]} 先手。规则：只能预留桌面公开牌；有人到 15 分立即结束。")
            self.snapshots = [(g.clone(), len(self.history))]
            self._last_key, self._repeat = None, 0
            self._bump()
            self._save()
            self._maybe_start_robot()

    def begin_from_reading(self):
        with self.lock:
            r = self.reading
            if r is None or r.market is None or r.nobles is None or not self.reading_meta.get("usable", True):
                raise ValueError("还没有完整识别出桌面的牌和贵族")
            if any(c in (None, UNKNOWN) for row in r.market for c in row):
                raise ValueError("桌面上还有未识别的牌，请手动选择")
            self.begin(r.market, r.nobles, by="识别结果，操作员")

    # ------------------------------------------------------------------------------------------ readings
    def on_reading(self, t, meta=None):
        """a new reading of the table (TableState) from the vision model"""
        with self.lock:
            self.reading, self.reading_time, self.reading_meta = t, self.clock(), meta or {}
            usable = self.reading_meta.get("usable", True)
            key = t.key() if usable else None
            self._repeat = (self._repeat + 1 if key == self._last_key else 1) if usable else 0
            self._last_key = key
            if not t.visible or t.issues:
                text = "摄像头画面不完整或有看不清的地方" if not t.visible else "识别提示"
                self._alert("camera", text, "error" if not t.visible else "warn", t.issues)
            else:
                self._clear("camera")
            self._bump()
            if self.phase == "setup":
                self._setup_reading(t)
                return
            if self.phase != "play" or not usable:
                return
            problems = validate(t, current_player=self.game.current_player)
            if problems:
                self._alert("reading_rules", "识别出的牌局不符合游戏规则（多半是识别错误或桌面摆放有误）", "error", problems)
                return
            self._clear("reading_rules")
            expected = table_of(self.game)
            if matches(expected, t):
                self._clear("unexplained", "mismatch", "ambiguous")
                self.ambiguous, self.progress = [], None
                return
            if self._repeat < self.stable:
                self.progress = "检测到桌面变化，等待下一次识别确认"
                return
            candidates = [c for c in explain(self.game, t) if c.actions or c.reveals]
            if self.is_robot(self.game.current_player) and self.thinking:
                self.progress = f"{self.names[self.game.current_player]} 正在计算动作，桌面变化稍后再核对"
                return
            if self.is_robot(self.game.current_player) and self.instruction is not None:
                self._robot_reading(t, candidates, expected)
            else:
                self._human_reading(t, candidates, expected)

    def _setup_reading(self, t):
        problems = []
        if t.market is None:
            problems.append("没有读出桌面的发展卡")
        else:
            for level, row in enumerate(t.market):
                for slot, c in enumerate(row):
                    if c in (None, UNKNOWN):
                        problems.append(f"{slot_text(level, slot)} 未识别")
        if t.nobles is None or len(t.nobles) != 5:
            problems.append(f"贵族应为 5 位，识别出 {0 if t.nobles is None else len(t.nobles)} 位")
        ids = [c for row in (t.market or []) for c in row if c not in (None, UNKNOWN)]
        if len(ids) != len(set(ids)):
            problems.append("识别出重复的牌")
        if problems:
            self._alert("setup", "开局摆放尚未完整识别，可手动修改后确认", "warn", problems)
        else:
            self._clear("setup")

    def _human_reading(self, t, candidates, expected):
        if not candidates:
            self._unexplained(t, expected)
            return
        keys = {c.game.key() for c in candidates}
        if len(keys) > 1:
            self.ambiguous = candidates
            self._alert("ambiguous", "识别到的变化有多种合法解释，请在页面上选择实际发生的动作", "warn",
                        [self.candidate_text(c) for c in candidates])
            return
        self._accept(candidates[0], by="识别")

    def _robot_reading(self, t, candidates, expected):
        good = [c for c in map(self._as_instructed, candidates) if c is not None]
        if good:
            best = max(good, key=lambda c: (len(c.actions), c.status == "complete"))
            self._accept(best, by="代理执行（识别确认）")
            return
        if candidates:
            self._alert("mismatch", f"代理执行的动作与 {self.names[self.game.current_player]} 的指令不符，请按大屏幕指令恢复后重做",
                        "error", [f"识别为：{self.candidate_text(c)}" for c in candidates[:3]] +
                        [f"指令为：{s}" for s in self.instruction["steps"]])
            return
        self._unexplained(t, expected)

    def _as_instructed(self, cand):
        """the candidate rewritten as a prefix of the robot's instruction if it reaches the same state (returning
        a blue token after taking blue, red and black is the same table as taking red and black and returning
        nothing more), else None"""
        target = cand.game.key()
        g, reveals = self.game.clone(), list(cand.reveals)
        if g.phase == "chance" and reveals:
            level, slot, card = reveals[0]
            g.resolve_chance(card)
            reveals = reveals[1:]
        used = len(cand.reveals) - len(reveals)
        instructed = self.instruction["actions"]
        for k in range(len(instructed) + 1):
            if k:
                g.step(instructed[k - 1])
                if g.phase == "chance" and reveals:
                    g.resolve_chance(reveals[0][2])
                    reveals = reveals[1:]
            if g.key() == target and not reveals and (k or used):
                return Candidate(instructed[:k], cand.reveals, cand.game, cand.status, cand.note)
            if g.phase == "chance":
                break
        return None

    def _unexplained(self, t, expected):
        if self._repeat < self.unexplained:
            self.progress = "桌面变化暂时无法解释，可能是动作还没做完"
            return
        g = self.game
        player = g.current_player
        details = diff(expected, t, self.names)
        hints = []
        for i, (e, o) in enumerate(zip(expected.seats, t.seats)):
            changed = any(getattr(o, f) is not None and getattr(o, f) != getattr(e, f)
                          for f in ("tokens", "bonuses", "reserved", "nobles"))
            if changed and i != player:
                hints.append(f"非当前玩家 {self.names[i]} 的区域发生了变化（现在轮到 {self.names[player]}）")
        o = t.seats[player]
        if o.reserved is not None and len(o.reserved) > len(g.players[player].reserved) and t.market is not None:
            if [c for row in t.market for c in row] == [c for row in g.market for c in row]:
                hints.append("疑似从牌堆暗扣：本规则只允许预留桌面上公开的牌")
        if o.tokens is not None:
            gained = [b - a for a, b in zip(g.players[player].tokens, o.tokens)]
            for c in range(5):
                if gained[c] == 2 and g.bank[c] < 4:
                    hints.append(f"银行{CN[c]}筹码不足 4 枚时不能拿两枚同色")
            if sum(max(0, x) for x in gained[:5]) > 3:
                hints.append("一次最多拿 3 枚筹码")
        self._alert("unexplained", "桌面变化无法用当前玩家的合法动作解释：可能是不允许的动作、识别错误或摆放有误。"
                    "请核对后把桌面恢复，或在页面上手动录入动作", "error", hints + details)

    # ------------------------------------------------------------------------------------------ applying a turn
    def candidate_text(self, cand):
        g = self.game.clone()
        parts = []
        actions, reveals = list(cand.actions), list(cand.reveals)
        if g.phase == "chance" and reveals:
            level, slot, card = reveals.pop(0)
            parts.append(f"翻出 {card_text(card)} 补到{slot_text(level, slot)}")
            g.resolve_chance(card)
        for a in actions:
            parts.append(action_text(g, a, self.names))
            g.step(a)
            if g.phase == "chance" and reveals:
                level, slot, card = reveals.pop(0)
                parts.append(f"翻出 {card_text(card)} 补到{slot_text(level, slot)}")
                g.resolve_chance(card)
        return "；".join(parts)

    def _accept(self, cand, by):
        g = self.game.clone()
        player = g.current_player
        actions, reveals = list(cand.actions), list(cand.reveals)

        def reveal():
            level, slot, card = reveals.pop(0)
            g.resolve_chance(card)
            self._event("reveal", f"牌堆翻出 {card_text(card)}，补到{slot_text(level, slot)}")

        def awarded(before):
            for n in g.players[player].nobles:
                if n not in before:
                    self._event("noble", f"{self.names[player]} 获得{noble_text(n)}（+3 分）", player)

        if g.phase == "chance" and reveals:
            before = list(g.players[player].nobles)
            reveal()
            awarded(before)
        for a in actions:
            before = list(g.players[player].nobles)
            text = action_text(g, a, self.names)
            g.step(a)
            self._event("action", f"{self.names[player]}：{text}", player, by, a)
            if g.phase == "chance" and reveals:
                reveal()
            if a.kind != "noble":
                awarded(before)
        if g.key() != cand.game.key():
            raise RuntimeError("内部错误：复现的局面与识别解释不一致")
        self.game = g
        if self.instruction is not None and cand.actions != self.instruction["actions"][:len(cand.actions)]:
            self.instruction = None     # the operator played something else for the robot
        if self.instruction is not None:
            done = len(cand.actions)
            self.instruction["actions"] = self.instruction["actions"][done:]
            self.instruction["steps"] = self.instruction["steps"][done:]
            if not self.instruction["actions"] and g.phase != "chance":
                self.instruction = None
        if cand.note == "noble_unmoved":
            self._alert("noble_unmoved", "请把刚获得的贵族板块移到玩家面前", "warn")
        else:
            self._clear("noble_unmoved")
        self._clear("unexplained", "mismatch", "ambiguous")
        self.ambiguous, self.progress = [], None
        self._after_change()

    def _after_change(self):
        g = self.game
        self.snapshots.append((g.clone(), len(self.history)))
        if g.phase == "terminal":
            self.phase = "terminal"
            self.instruction = None
            points = "，".join(f"{self.names[i]} {p.points} 分" for i, p in enumerate(g.players))
            if g.end_reason == "draw":
                self._event("end", f"所有玩家都无法行动，平局。{points}")
            else:
                winners = "、".join(self.names[i] for i in g.winners)
                self._event("end", f"对局结束：{winners} 获胜。{points}")
        self._bump()
        self._save()
        self._maybe_start_robot()

    # ------------------------------------------------------------------------------------------ robots
    def _maybe_start_robot(self):
        g = self.game
        if (self.phase != "play" or g is None or g.phase not in ("action", "discard", "noble")
                or not self.is_robot(g.current_player) or self.instruction is not None or self.thinking):
            return
        self.thinking, self.robot_error = True, None
        snapshot = g.clone()
        if self.sync_robots:
            self._plan(snapshot)
        else:
            threading.Thread(target=self._plan, args=(snapshot,), daemon=True).start()

    def _plan(self, snapshot):
        player, turn = snapshot.current_player, snapshot.turn
        try:
            robot = self.robots[player]
            g, pairs, notes = snapshot.clone(), [], []
            while g.phase in ("action", "discard", "noble") and g.current_player == player and g.turn == turn:
                a = robot.choose(g)
                before = list(g.players[player].nobles)
                pairs.append((a, action_text(g, a, self.names)))
                g = g.apply(a)
                if a.kind != "noble":
                    for n in g.players[player].nobles:
                        if n not in before:
                            notes.append(f"（自动）获得{noble_text(n)}，把贵族板块放到面前")
            # returned tokens: only the multiset matters on the table (table.explain lists them by colour)
            ordered, run = [], []
            for a, text in pairs + [(None, None)]:
                if a is not None and a.kind == "discard":
                    run.append((a, text))
                    continue
                ordered += sorted(run, key=lambda x: x[0].colors)
                run = []
                if a is not None:
                    ordered.append((a, text))
            actions = [a for a, _ in ordered]
            steps = [t for _, t in ordered] + notes
            refill = None
            # e.g. take three tokens and return the same three: the camera cannot tell it from waiting
            invisible = g.phase != "chance" and table_of(g).key() == table_of(snapshot).key()
            if g.phase == "chance":
                level, slot = g.pending_refill
                refill = f"从 {level + 1} 级牌堆翻开最上面一张，补到{slot_text(level, slot)}"
            error = None
        except Exception as e:     # noqa: BLE001  (shown on the page; the operator can play the move by hand)
            actions, steps, refill, invisible, error = [], [], None, False, f"{type(e).__name__}: {e}"
        with self.lock:
            self.thinking = False
            if self.game is None or self.game.key() != snapshot.key() or self.phase != "play":
                self._bump()
                self._maybe_start_robot()
                return
            if error:
                self.robot_error = error
                self._alert("robot", f"{self.names[player]} 计算动作失败，可在页面上手动录入动作", "error", [error])
                self._bump()
                return
            self._clear("robot")
            # the engine awards a single eligible noble itself: those lines are text only, not actions
            texts = [s for s in steps if not s.startswith("（自动）")]
            self.instruction = {"player": player, "name": self.names[player], "actions": actions, "steps": texts,
                                "notes": [s for s in steps if s.startswith("（自动）")], "refill": refill,
                                "needs_confirm": invisible, "created": self.clock()}
            if invisible:
                self.instruction["notes"].append("这一步执行后桌面看起来没有变化，执行完请在控制台点击“确认已执行”")
            self._event("instruction", f"{self.names[player]} 的指令：" + "；".join(texts + ([refill] if refill else [])),
                        player, "机器人")
            self._bump()

    def instruction_view(self):
        ins = self.instruction
        g = self.game
        if ins is None:
            if g is not None and self.phase == "play" and g.phase == "chance" and self.is_robot(g.current_player):
                level, slot = g.pending_refill
                return {"player": g.current_player, "name": self.names[g.current_player], "steps": [],
                        "refill": f"从 {level + 1} 级牌堆翻开最上面一张，补到{slot_text(level, slot)}", "notes": []}
            return None
        out = {k: v for k, v in ins.items() if k != "actions"}
        if g is not None and g.phase == "chance":
            level, slot = g.pending_refill
            out["refill"] = f"从 {level + 1} 级牌堆翻开最上面一张，补到{slot_text(level, slot)}"
        out["actions"] = [a.to_dict() for a in ins["actions"]]
        return out

    # ------------------------------------------------------------------------------------------ operator
    def manual_action(self, action_dict, by="手动录入"):
        with self.lock:
            if self.phase != "play":
                raise ValueError("对局未在进行中")
            a = Action.from_dict(action_dict)
            g = self.game.clone()
            if a not in g.legal_actions():
                raise ValueError("不是当前玩家的合法动作")
            g.step(a)
            cand = Candidate([a], [], g, "complete" if g.phase in ("action", "terminal") else "partial")
            self._accept(cand, by=by)

    def manual_reveal(self, card_id):
        with self.lock:
            g = self.game
            if self.phase != "play" or g.phase != "chance":
                raise ValueError("现在不需要补牌")
            level, slot = g.pending_refill
            if card_id not in g.remaining[level]:
                raise ValueError("这张牌不在该级剩余牌堆中")
            g2 = g.clone()
            g2.resolve_chance(card_id)
            self._accept(Candidate([], [(level, slot, card_id)], g2, "complete"), by="手动录入")

    def confirm_robot(self):
        with self.lock:
            if self.instruction is None:
                raise ValueError("没有待执行的机器人指令")
            done = 0
            g2 = self.game.clone()
            for a in self.instruction["actions"]:
                g2.step(a)
                done += 1
                if g2.phase == "chance":
                    break
            self._accept(Candidate(self.instruction["actions"][:done], [], g2, "partial"), by="操作员确认执行")

    def choose_ambiguous(self, index):
        with self.lock:
            if not 0 <= index < len(self.ambiguous):
                raise ValueError("选项无效")
            self._accept(self.ambiguous[index], by="识别（操作员选择）")

    def undo(self):
        with self.lock:
            if self.phase == "setup" or len(self.snapshots) < 2:
                raise ValueError("没有可以撤销的步骤")
            self.snapshots.pop()
            g, n = self.snapshots[-1]
            self.game = g.clone()
            self.history = self.history[:n]
            self.phase = "play"
            self.instruction, self.thinking = None, False
            self.ambiguous, self.progress = [], None
            self._clear("unexplained", "mismatch", "ambiguous", "robot")
            self._event("system", "撤销了上一步：请把实体桌面也恢复到这个局面")
            self.snapshots[-1] = (g, len(self.history))
            self._bump()
            self._save()
            self._maybe_start_robot()

    # ------------------------------------------------------------------------------------------ view
    def hints(self):
        g = self.game
        if self.phase == "setup":
            return ["请摆好开局：三行各 4 张发展卡、5 位贵族、银行五色各 7 枚和黄金 5 枚。识别完成后在页面上确认开局。"]
        if g is None:
            return []
        out = []
        if g.phase == "terminal":
            return ["对局已结束"]
        p = g.current_player
        who = f"{self.names[p]}（{'机器人：请人类代理按大屏幕指令执行' if self.is_robot(p) else '人类玩家'}）"
        if g.phase == "action":
            out.append(f"轮到 {who} 行动")
        elif g.phase == "chance":
            level, slot = g.pending_refill
            out.append(f"请从 {level + 1} 级牌堆翻开最上面一张，补到{slot_text(level, slot)}")
        elif g.phase == "discard":
            n = sum(g.players[p].tokens)
            out.append(f"{self.names[p]} 持有 {n} 枚筹码，必须退回 {n - 10} 枚（任意颜色，包括黄金）")
        elif g.phase == "noble":
            out.append(f"{self.names[p]} 同时满足多位贵族，选择一位：" + "、".join(map(noble_text, g.eligible_nobles())))
        if self.thinking:
            out.append(f"{self.names[p]} 正在计算动作…")
        if self.progress:
            out.append(self.progress)
        if g.phase == "action":
            if len(g.players[p].reserved) >= 3:
                out.append(f"{self.names[p]} 已有 3 张预留牌，不能再预留")
            if not g.bank[5]:
                out.append("黄金已空：预留不再获得黄金")
            low = [CN[c] for c in range(5) if g.bank[c] < 4]
            if low:
                out.append(f"银行 {'、'.join(low)} 少于 4 枚：这些颜色不能拿两枚同色")
        leaders = [(pl.points, i) for i, pl in enumerate(g.players) if pl.points >= 12]
        for pts, i in sorted(leaders, reverse=True):
            out.append(f"{self.names[i]} 已有 {pts} 分：到 15 分立即结束，不补完本轮")
        return out

    def legal_view(self):
        g = self.game
        if self.phase != "play" or g is None or g.phase in ("chance", "terminal"):
            return []
        return [{"action": a.to_dict(), "text": action_text(g, a, self.names)} for a in g.legal_actions()]

    def view(self):
        with self.lock:
            g = self.game
            reveal_pool = []
            if g is not None and g.phase == "chance":
                level, _ = g.pending_refill
                reveal_pool = [{"id": c, "text": card_text(c)} for c in sorted(g.remaining[level])]
            reading = None
            if self.reading is not None:
                reading = self.reading.to_dict()
                reading["time"] = self.reading_time
                reading["meta"] = self.reading_meta
                reading["repeat"] = self._repeat
            return {
                "phase": self.phase, "version": self.version, "seats": self.seats, "names": self.names,
                "state": None if g is None else g.observation(),
                "current": None if g is None else g.current_player,
                "hints": self.hints(), "alerts": sorted(self.alerts.values(), key=lambda a: a["since"]),
                "instruction": self.instruction_view(), "thinking": self.thinking,
                "history": self.history[-300:], "legal": self.legal_view(), "reveal_pool": reveal_pool,
                "ambiguous": [self.candidate_text(c) for c in self.ambiguous],
                "reading": reading, "log": self.log_path,
            }
