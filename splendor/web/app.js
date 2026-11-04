"use strict";
const $ = (id) => document.getElementById(id);
const names = ["白", "蓝", "绿", "红", "黑", "金"];
const colors = ["white", "blue", "green", "red", "black", "gold"];
let view = null,
  selection = [],
  selectedCard = null,
  running = false,
  busy = false,
  timer = null;
const gem = (c) => `var(--${colors[c]})`;
const reqs = (values) =>
  values
    .map((n, c) =>
      n
        ? `<span class="req" style="--gem:${gem(c)}"><i class="dot"></i>${n}</span>`
        : "",
    )
    .join("");
const paymentText = (payment) =>
  payment
    .map((n, c) => (n ? `${names[c]}${n}` : ""))
    .filter(Boolean)
    .join(" · ") || "免费";
function actionText(a) {
  if (a.kind === "take")
    return "拿取 " + a.colors.map((c) => names[c]).join("、");
  if (a.kind === "reserve") return "公开预留 " + a.card_id;
  if (a.kind === "buy") return `购买 ${a.card_id}（${paymentText(a.payment)}）`;
  if (a.kind === "noble") return "选择贵族 " + a.noble_id;
  if (a.kind === "discard") return "退回 " + names[a.colors[0]];
  return "未知动作";
}
function stop() {
  running = false;
  clearTimeout(timer);
  $("auto").textContent = "连续模拟";
}
function error(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}
async function post(path, data = {}) {
  if (busy) return false;
  busy = true;
  $("error").hidden = true;
  syncControls();
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...data, version: view.version }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "请求失败");
    view = result;
    selection = [];
    selectedCard = null;
    busy = false;
    render();
    return true;
  } catch (e) {
    stop();
    error(e.message);
    return false;
  } finally {
    busy = false;
    syncControls();
  }
}
function syncControls() {
  const ended = !view || view.state.phase === "terminal";
  for (const id of ["new", "undo", "step", "import", "export"])
    $(id).disabled =
      busy ||
      (id === "step" && ended) ||
      (id === "undo" && (!view || !view.history_count));
  $("auto").disabled = (busy && !running) || ended;
  $("take").disabled = busy || !matchingTake();
}
function matchingTake() {
  if (!view) return null;
  const wanted = [...selection].sort((a, b) => a - b).join(",");
  return view.actions.find(
    (a) => a.kind === "take" && a.colors.join(",") === wanted,
  );
}
function selectToken(color) {
  if (
    busy ||
    running ||
    view.state.phase !== "action" ||
    color === 5 ||
    !view.state.bank[color]
  )
    return;
  const matches = selection.filter((c) => c === color).length;
  if (matches === 2) selection = selection.filter((c) => c !== color);
  else if (matches === 1) {
    if (
      selection.length === 1 &&
      view.actions.some(
        (a) =>
          a.kind === "take" &&
          a.colors.length === 2 &&
          a.colors[0] === color &&
          a.colors[1] === color,
      )
    )
      selection = [color, color];
    else selection = selection.filter((c) => c !== color);
  } else {
    if (selection.length === 2 && selection[0] === selection[1]) selection = [];
    if (selection.length < 3) selection.push(color);
  }
  renderBank();
  syncControls();
}
function cardHTML(id, reserved = false) {
  if (!id) return '<div class="empty-card">牌堆已空</div>';
  const c = view.cards[id];
  const buys = view.actions.filter((a) => a.kind === "buy" && a.card_id === id);
  const reserve = view.actions.some(
    (a) => a.kind === "reserve" && a.card_id === id,
  );
  const actionable = !busy && !running;
  return `<article class="card" data-card="${id}" style="--gem:${gem(c.bonus)}"><div class="card-top"><span class="points">${c.points}<small>分</small></span><span class="bonus">◆ ${names[c.bonus]} +1</span></div><div class="card-id">${id}</div><div class="costs">${reqs(c.cost)}</div><div class="card-actions"><button data-buy="${id}" ${!buys.length || !actionable ? "disabled" : ""}>购买${buys.length > 1 ? " ▾" : ""}</button>${!reserved ? `<button data-reserve="${id}" ${!reserve || !actionable ? "disabled" : ""}>预留</button>` : ""}</div></article>`;
}
function renderBank() {
  const s = view.state;
  $("bank").innerHTML = s.bank
    .map(
      (n, c) =>
        `<div class="bank-item"><button class="bank-token ${selection.includes(c) ? "selected" : ""}" data-color="${c}" aria-label="${names[c]}宝石，库存${n}" style="--gem:${gem(c)}" ${!n || c === 5 || s.phase !== "action" || running ? "disabled" : ""}>${n}</button><span>${names[c]}${selection.includes(c) ? " ×" + selection.filter((i) => i === c).length : ""}</span></div>`,
    )
    .join("");
  document
    .querySelectorAll("[data-color]")
    .forEach(
      (el) => (el.onclick = () => selectToken(Number(el.dataset.color))),
    );
  $("selection").textContent = selection.length
    ? `已选 ${selection.map((c) => names[c]).join("、")}${matchingTake() ? "" : "（当前不能这样拿取）"}`
    : "选择三种不同颜色，或连点同色选择两枚";
  const held = s.players[s.current_player].tokens.reduce((a, b) => a + b, 0);
  $("token-hint").textContent =
    s.phase === "discard"
      ? `玩家 ${s.current_player + 1} 持币 ${held}/10 · 还需退回 ${held - 10} 枚`
      : s.discard_over_limit
        ? `玩家 ${s.current_player + 1} 持币 ${held}/10 · 超过 10 枚须在回合结束前退回`
        : `玩家 ${s.current_player + 1} 持币 ${held}/10 · 剩余容量 ${10 - held}`;
}
function renderDecision() {
  const s = view.state,
    box = $("decision");
  if (s.phase === "terminal") {
    box.innerHTML = `<h3>${s.end_reason === "draw" ? "四人连续跳过 · 平局" : "胜者：" + s.winners.map((i) => "玩家 " + (i + 1)).join("、")}</h3><p>共 ${s.turn} 手（含 ${s.passes || 0} 次无合法动作的跳过）。可导出对局复盘，或用相同种子重新开始。</p>`;
    return;
  }
  if (s.phase === "discard") {
    const tokens = s.players[s.current_player].tokens;
    const held = tokens.reduce((a, b) => a + b, 0);
    box.innerHTML =
      `<h3>退回筹码</h3><p>持币 ${held} 枚，超过上限，还需退回 ${held - 10} 枚。可退回任意颜色，包括刚拿的和黄金；退完后结算贵族并结束回合。</p><div class="payment-list">` +
      view.actions
        .map(
          (a, i) =>
            `<button class="payment-option" data-action="${i}"><span class="chip" style="--gem:${gem(a.colors[0])}"><i class="dot"></i>${tokens[a.colors[0]]}</span> 退回 1 枚${names[a.colors[0]]}</button>`,
        )
        .join("") +
      "</div>";
  } else if (s.phase === "noble") {
    box.innerHTML =
      "<h3>选择一位贵族</h3><p>本回合尚未结束；必须选择一位满足条件的贵族。</p>" +
      view.actions
        .map(
          (a, i) =>
            `<button class="payment-option" data-action="${i}">${a.noble_id} · 3 分 ${reqs(view.noble_data[a.noble_id].cost)}</button>`,
        )
        .join("");
  } else if (selectedCard) {
    const choices = view.actions
      .map((a, i) => ({ a, i }))
      .filter((x) => x.a.kind === "buy" && x.a.card_id === selectedCard);
    box.innerHTML = `<h3>选择支付方式</h3><p>${selectedCard} · 折扣已自动计入</p><div class="payment-list">${choices.map(({ a, i }) => `<button class="payment-option" data-action="${i}">${paymentText(a.payment)}</button>`).join("")}</div><button id="cancel-payment">取消</button>`;
    $("cancel-payment").onclick = () => {
      selectedCard = null;
      renderDecision();
    };
  } else {
    const counts = {};
    for (const a of view.actions) counts[a.kind] = (counts[a.kind] || 0) + 1;
    box.innerHTML = `<h3>玩家 ${s.current_player + 1} 行动</h3><p>点击宝石选择拿取；点击牌上的按钮购买或预留。</p><p>拿币 ${counts.take || 0} 种 · 预留 ${counts.reserve || 0} 种<br>购买 ${counts.buy || 0} 种（含支付组合）</p><p>自动走一步可驱动当前玩家；连续模拟可观察整局。</p>`;
  }
  box.querySelectorAll("[data-action]").forEach(
    (el) =>
      (el.onclick = () => {
        if (!running)
          post("/api/action", {
            action: view.actions[Number(el.dataset.action)],
          });
      }),
  );
}
function render() {
  const s = view.state,
    current = s.current_player;
  if (s.phase === "terminal") stop();
  $("round").textContent = `第 ${Math.floor(s.turn / 4) + 1} 轮 · ${s.turn} 手`;
  $("status").classList.toggle("final", s.phase === "terminal");
  $("status").textContent =
    s.phase === "terminal" && s.end_reason === "draw"
      ? `对局结束：四位玩家连续无合法动作，平局。`
      : s.phase === "terminal"
        ? `对局结束：${s.winners.map((i) => "玩家 " + (i + 1)).join("、")} 获胜。`
        : `轮到玩家 ${current + 1}${s.phase === "noble" ? "选择贵族" : s.phase === "discard" ? "退回筹码" : "行动"} · 当前持币 ${s.players[current].tokens.reduce((a, b) => a + b, 0)}/10`;
  $("nobles").innerHTML = s.nobles
    .map(
      (id) =>
        `<div class="noble"><div class="noble-top"><span>♛ ${id}</span><strong>3 分</strong></div><div class="requirements">${reqs(view.noble_data[id].cost)}</div></div>`,
    )
    .join("");
  $("market").innerHTML = [2, 1, 0]
    .map(
      (l) =>
        `<div class="market-row"><div class="deck"><span>${"◆".repeat(l + 1)}</span><strong>${s.remaining[l].length}</strong><small>剩余牌</small></div>${s.market[l].map((id) => cardHTML(id)).join("")}</div>`,
    )
    .join("");
  $("players").innerHTML = s.players
    .map(
      (p, i) =>
        `<article class="player ${i === current && s.phase !== "terminal" ? "active" : ""}"><div class="player-head"><h3>玩家 ${i + 1}${i === 0 ? " · 先手" : ""}</h3><span class="score">${p.points}<small style="font-size:11px"> 分</small></span></div><div class="player-section-label">持币 ${p.tokens.reduce((a, b) => a + b, 0)}/10</div><div class="chips">${p.tokens.map((n, c) => `<span class="chip" style="--gem:${gem(c)}"><i class="dot"></i>${n}</span>`).join("")}</div><div class="player-section-label">永久折扣 · 已购 ${p.purchased.length} 张</div><div class="chips">${p.bonuses.map((n, c) => `<span class="chip" style="--gem:${gem(c)}"><i class="dot"></i>${n}</span>`).join("")}</div><div class="player-section-label">公开预留 ${p.reserved.length}/3</div><div class="reserved">${p.reserved.map((c) => cardHTML(c, true)).join("") || '<p class="empty">暂无预留牌</p>'}</div><div class="player-section-label">贵族 ${p.nobles.length} 位 ${p.nobles.join(" · ")}</div><details><summary>查看已购牌</summary><div class="owned-list">${p.purchased.map((c) => `<span class="owned-item" style="--gem:${gem(view.cards[c].bonus)}">${c} · ${view.cards[c].points}分</span>`).join("") || '<p class="empty">暂无</p>'}</div></details></article>`,
    )
    .join("");
  document.querySelectorAll("[data-buy]").forEach(
    (el) =>
      (el.onclick = () => {
        if (busy || running) return;
        const matches = view.actions.filter(
          (a) => a.kind === "buy" && a.card_id === el.dataset.buy,
        );
        if (matches.length === 1) post("/api/action", { action: matches[0] });
        else {
          selectedCard = el.dataset.buy;
          renderDecision();
        }
      }),
  );
  document.querySelectorAll("[data-reserve]").forEach(
    (el) =>
      (el.onclick = () => {
        if (!busy && !running)
          post("/api/action", {
            action: view.actions.find(
              (a) => a.kind === "reserve" && a.card_id === el.dataset.reserve,
            ),
          });
      }),
  );
  $("action-count").textContent = `${view.actions.length} 个合法动作`;
  $("history").innerHTML =
    [...view.history]
      .reverse()
      .map(
        (e, i) =>
          `<li><small>#${view.history_count - i} · 玩家 ${e.player + 1}</small>${actionText(e.action)}</li>`,
      )
      .join("") || "<li>新牌局已就绪，等待第一手行动。</li>";
  $("seed-label").textContent = "种子 " + view.seed;
  renderBank();
  renderDecision();
  syncControls();
}
$("take").onclick = () => {
  const action = matchingTake();
  if (action && !running) post("/api/action", { action });
};
$("clear").onclick = () => {
  selection = [];
  renderBank();
  syncControls();
};
$("new").onclick = () => {
  stop();
  const value = $("seed").value.trim();
  const seed = value === "" ? null : Number(value);
  if (seed !== null && !Number.isSafeInteger(seed)) {
    error("请输入安全范围内的整数种子");
    return;
  }
  post("/api/new", { seed });
};
$("undo").onclick = () => {
  stop();
  post("/api/undo");
};
$("step").onclick = () => {
  stop();
  post("/api/bot", { policy: $("policy").value });
};
async function tick() {
  if (!running) return;
  const ok = await post("/api/bot", { policy: $("policy").value });
  if (ok && running) timer = setTimeout(tick, 280);
}
$("auto").onclick = () => {
  if (running) {
    stop();
    render();
  } else {
    running = true;
    $("auto").textContent = "暂停模拟";
    render();
    tick();
  }
};
$("export").onclick = async () => {
  stop();
  try {
    const response = await fetch("/api/export");
    if (!response.ok) throw new Error("导出失败");
    const record = await response.json();
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(record, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `splendor-${record.seed}.json`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    error(e.message);
  }
  render();
};
$("import").onclick = () => {
  stop();
  $("file").click();
};
$("file").onchange = async () => {
  const file = $("file").files[0];
  if (!file) return;
  try {
    if (file.size > 2000000) throw new Error("文件不能超过 2 MB");
    await post("/api/import", { record: JSON.parse(await file.text()) });
  } catch (e) {
    error(e.message);
  }
  $("file").value = "";
};
$("rules-button").onclick = () => $("rules").showModal();
$("close-rules").onclick = () => $("rules").close();
fetch("/api/state")
  .then((r) => {
    if (!r.ok) throw new Error("载入失败");
    return r.json();
  })
  .then((result) => {
    view = result;
    render();
  })
  .catch((e) => error(e.message));
