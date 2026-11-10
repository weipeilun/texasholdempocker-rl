"use strict";
let state = null;
let renderedVersion = -1;
let lastFrameSeq = -1;
let armFrameSeq = "";
let armImageURL = null;
let setupDirty = false;
let localError = null;

function showError(message) {
  localError = message;
  renderAlerts();
  setTimeout(() => {
    if (localError === message) {
      localError = null;
      renderAlerts();
    }
  }, 8000);
}

async function act(path, body = {}) {
  try {
    state = await api(path, body);
    renderedVersion = -1;
    render();
  } catch (e) {
    showError(e.message);
  }
}

function seatRowsHTML() {
  const policies = CARDS.policies;
  return [0, 1, 2, 3]
    .map((i) => `<div class="seat-row">
      <b>玩家${i + 1}</b>
      <input id="seat-name-${i}" type="text" placeholder="名字" value="玩家${i + 1}" />
      <select id="seat-kind-${i}"><option value="human">人类</option><option value="robot"${i === 3 ? " selected" : ""}>机器人</option></select>
      <select id="seat-policy-${i}">${Object.entries(policies)
        .map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("")}</select>
    </div>`)
    .join("");
}

function renderAlerts() {
  const g = state && state.game;
  let html = localError ? `<div class="alert error"><b>${esc(localError)}</b></div>` : "";
  html += alertsHTML(g ? g.alerts : []);
  $("alerts").innerHTML = html;
}

function renderHints(g) {
  if (!g) {
    $("hints").innerHTML = `<div class="first">还没有对局</div><div>展开下面的“新建对局”，设置四个座位。</div>`;
    return;
  }
  $("hints").innerHTML = g.hints.map((h, i) => `<div class="${i === 0 ? "first" : ""}">${esc(h)}</div>`).join("");
}

function renderInstruction(g) {
  const ins = g && g.instruction;
  const box = $("instruction");
  if (!ins) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  const steps = (ins.steps || []).map((s) => `<li>${esc(s)}</li>`).join("");
  box.innerHTML = `<h3>机器人 ${esc(ins.name)} 的指令（大屏幕同步显示）</h3>
    ${steps ? `<ol>${steps}</ol>` : ""}
    ${ins.refill ? `<div style="margin-top:6px;font-size:16px">补牌：${esc(ins.refill)}</div>` : ""}
    ${(ins.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join("")}`;
}

function renderBoard(g) {
  const s = g && g.state;
  if (!s) {
    $("nobles").innerHTML = "";
    $("market").innerHTML = `<div class="muted">开局确认后显示</div>`;
    $("bank").innerHTML = "";
    $("players").innerHTML = "";
    $("turn-label").textContent = "";
    return;
  }
  $("turn-label").textContent = `第 ${s.turn + 1} 手 · 当前：${g.names[s.current_player]}`;
  $("nobles").innerHTML = s.nobles.map(nobleHTML).join("");
  const pending = s.pending_refill;
  $("market").innerHTML = [2, 1, 0]
    .map((level) => `<div class="market-row"><div class="deck"><b>${level + 1} 级</b>剩 ${s.remaining[level].length}</div>${s.market[level]
      .map((c, slot) => cardHTML(c, pending && pending[0] === level && pending[1] === slot ? "pending" : ""))
      .join("")}</div>`)
    .join("");
  $("bank").innerHTML = tokensHTML(s.bank);
  $("players").innerHTML = s.players
    .map((p, i) => {
      const seat = g.seats[i];
      const robot = seat.kind === "robot";
      return `<div class="player ${i === s.current_player && s.phase !== "terminal" ? "current" : ""}">
        <div class="head"><span class="name">${esc(g.names[i])}<span class="tag ${robot ? "robot" : ""}">${
          robot ? "机器人·" + esc(({ model: "模型" })[seat.policy] || seat.policy) : "人类"}</span></span>
          <span class="score">${p.points}</span></div>
        <div class="kv"><b>筹码 ${p.tokens.reduce((a, b) => a + b, 0)}</b>${tokensHTML(p.tokens)}</div>
        <div class="kv"><b>发展卡</b>${p.bonuses.map((n, c) => `<span class="chip"><i class="gem" style="${gemStyle(c)}"></i>${n}</span>`).join("")}</div>
        <div class="kv"><b>预留</b>${p.reserved.length ? "" : '<span class="muted">无</span>'}</div>
        <div class="reserved">${p.reserved.map((c) => cardHTML(c)).join("")}</div>
        <div class="kv"><b>贵族</b>${p.nobles.length ? p.nobles.map((n) => CARDS.nobles[n].text).join("、") : '<span class="muted">无</span>'}</div>
      </div>`;
    })
    .join("");
}

function renderHistory(g) {
  const items = g ? g.history.slice().reverse() : [];
  $("history").innerHTML = items
    .map((h) => `<li class="h-${h.kind}"><span class="meta">${timeText(h.t)}${h.turn !== null && h.turn !== undefined ? " · 第" + (h.turn + 1) + "手" : ""}${h.by ? " · " + esc(h.by) : ""}</span>${esc(h.text)}</li>`)
    .join("");
}

function renderSetup(g) {
  const panel = $("setup-panel");
  if (!g || g.phase !== "setup") {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  if (setupDirty && $("setup-market").childElementCount) return;
  const reading = g.reading;
  const market = reading && reading.market ? reading.market : [[null, null, null, null], [null, null, null, null], [null, null, null, null]];
  const nobles = reading && reading.nobles ? reading.nobles : [];
  const byLevel = [1, 2, 3].map((l) => Object.entries(CARDS.cards).filter(([, c]) => c.level === l));
  $("setup-market").innerHTML = [2, 1, 0]
    .map((level) => `<div class="setup-row"><div class="deck"><b>${level + 1} 级</b></div>${[0, 1, 2, 3]
      .map((slot) => {
        const cur = market[level][slot];
        return `<select data-level="${level}" data-slot="${slot}" class="setup-card"><option value="">（未识别，请选择）</option>${byLevel[level]
          .map(([id, c]) => `<option value="${id}"${id === cur ? " selected" : ""}>${esc(c.text)}</option>`).join("")}</select>`;
      }).join("")}</div>`)
    .join("");
  $("setup-nobles").innerHTML = [0, 1, 2, 3, 4]
    .map((k) => `<select class="setup-noble"><option value="">贵族 ${k + 1}（未识别）</option>${Object.entries(CARDS.nobles)
      .map(([id, n]) => `<option value="${id}"${id === nobles[k] ? " selected" : ""}>${esc(n.text)}</option>`).join("")}</select>`)
    .join("");
  for (const el of document.querySelectorAll(".setup-card, .setup-noble")) el.onchange = () => (setupDirty = true);
}

function renderManual(g) {
  const legal = g ? g.legal : [];
  $("legal").innerHTML = legal.length
    ? legal.map((a, i) => `<option value="${i}">${esc(a.text)}</option>`).join("")
    : `<option value="">（现在没有可录入的玩家动作）</option>`;
  $("apply-action").disabled = !legal.length;
  const pool = g ? g.reveal_pool : [];
  $("reveal").innerHTML = pool.length
    ? pool.map((c) => `<option value="${c.id}">${esc(c.text)}</option>`).join("")
    : `<option value="">（现在不需要补牌）</option>`;
  $("apply-reveal").disabled = !pool.length;
  $("confirm-robot").disabled = !(g && g.instruction && g.instruction.actions && g.instruction.actions.length);
  $("undo").disabled = !(g && g.phase !== "setup");
  $("ambiguous").innerHTML = g && g.ambiguous.length
    ? `<div class="alert warn"><b>选择实际发生的动作：</b>${g.ambiguous
      .map((t, i) => `<div style="margin-top:6px"><button onclick="act('/api/game/choose',{index:${i}})">选这个</button> ${esc(t)}</div>`).join("")}</div>`
    : "";
}

function renderArm() {
  const arm = state.arm;
  $("arm-panel").classList.toggle("hidden", !arm);
  if (!arm) return;
  $("arm-camera-toggle").disabled = !arm.control;
  $("arm-camera-toggle").textContent = arm.running ? "关闭机械臂相机" : "打开机械臂相机";
  $("arm-auto-toggle").disabled = !arm.control || (!arm.auto && (
    !arm.running || !state.frame.time || !arm.vision_configured || !state.vision.enabled));
  $("arm-auto-toggle").textContent = arm.auto ? "停止自动取景" : "启用机械臂自动取景";
  const labels = {clear: "清楚", cropped: "出画", blocked: "被遮挡", unreadable: "看不清", unknown: "待确认", retained: "已保留"};
  const coverage = Object.entries(arm.coverage).map(([region, value]) => `${region} ${labels[value]}`);
  const messages = [arm.message, arm.error, coverage.join(" · ")];
  if (arm.running && !arm.vision_configured) messages.push("请在设置中配置视觉模型后开始识别");
  $("arm-status").textContent = messages.filter(Boolean).join(" · ");
}

async function pollArmPreview() {
  try {
    if (state?.arm?.running) {
      const r = await fetch(`/api/arm/frame.jpg?after=${armFrameSeq}`);
      if (r.status === 200) {
        const blob = await r.blob();
        if (state?.arm?.running) {
          if (armImageURL) URL.revokeObjectURL(armImageURL);
          armImageURL = URL.createObjectURL(blob);
          armFrameSeq = r.headers.get("X-Frame-Seq") || "";
          $("preview").src = armImageURL;
          $("preview").classList.remove("hidden");
        }
      }
    }
  } catch (e) {
    // The regular state poll reports connection and camera errors.
  } finally {
    setTimeout(pollArmPreview, state?.arm?.running ? 160 : 500);
  }
}

function renderVision() {
  const v = state.vision;
  const f = state.frame;
  const age = f.time ? Math.round(state.now - f.time) : null;
  $("frame-info").textContent = f.time ? `${f.source || ""} · ${age} 秒前 · ${Math.round(f.bytes / 1024)} KB` : "没有画面";
  renderArm();
  if (!state.arm?.running) {
    if (armImageURL) {
      URL.revokeObjectURL(armImageURL);
      armImageURL = null;
      armFrameSeq = "";
      lastFrameSeq = -1;
    }
    $("preview").classList.toggle("hidden", !f.time);
    if (f.time && f.seq !== lastFrameSeq) {
      lastFrameSeq = f.seq;
      $("preview").src = `/api/frame.jpg?seq=${f.seq}`;
    }
  } else if (!f.time) {
    $("preview").classList.add("hidden");
  }
  const last = v.last;
  let line = v.enabled ? "自动识别：开" : "自动识别：已暂停";
  if (v.busy) line += " · 识别中…";
  if (last) {
    line += last.ok
      ? ` · 上次识别 <span class="ok">成功</span> ${timeText(last.time)}（${esc(last.meta.model)}，${last.meta.latency} 秒）`
      : ` · 上次识别 <span class="bad">失败</span> ${timeText(last.time)}：${esc(last.error)}`;
  }
  if (state.owner) line += ` · 使用 ${esc(state.owner)} 的视觉设置`;
  if (state.camera_error) line += ` · <span class="bad">本机摄像头：${esc(state.camera_error)}</span>`;
  const m = state.model;
  line += `<br>机器人模型：${esc(m.weights)}，${m.sims} 次模拟，${m.loaded ? "已加载（" + esc(m.device) + "）" : "首次行动时加载"}`;
  $("vision-status").innerHTML = line;
  $("vision-toggle").textContent = v.enabled ? "暂停自动识别" : "恢复自动识别";
  const g = state.game;
  const reading = g?.reading || state.arm?.reading;
  $("reading").textContent = reading ? JSON.stringify(reading, null, 1) : "（还没有识别结果）";
}

function render() {
  const g = state.game;
  $("user").textContent = state.user;
  renderAlerts();
  renderVision();
  if (g && g.version === renderedVersion) return;
  renderedVersion = g ? g.version : -2;
  renderHints(g);
  renderInstruction(g);
  renderSetup(g);
  renderBoard(g);
  renderHistory(g);
  renderManual(g);
  $("new-panel").open = !g || g.phase === "terminal";
  $("robot-info").textContent = g && g.thinking ? "机器人正在计算动作…" : "";
}

async function poll() {
  try {
    state = await api("/api/state");
    render();
  } catch (e) {
    showError(e.message);
  }
  setTimeout(poll, 1000);
}

(async () => {
  await loadCards();
  $("seats").innerHTML = seatRowsHTML();
  $("new-game").onclick = () => {
    if (state && state.game && state.game.phase === "play" && !confirm("当前对局还没结束，确定新建对局？")) return;
    const seats = [0, 1, 2, 3].map((i) => ({
      name: $(`seat-name-${i}`).value,
      kind: $(`seat-kind-${i}`).value,
      policy: $(`seat-policy-${i}`).value,
    }));
    setupDirty = false;
    act("/api/game/new", { seats });
  };
  $("begin").onclick = () => {
    const market = [[null, null, null, null], [null, null, null, null], [null, null, null, null]];
    for (const el of document.querySelectorAll(".setup-card")) market[+el.dataset.level][+el.dataset.slot] = el.value || null;
    const nobles = [...document.querySelectorAll(".setup-noble")].map((el) => el.value).filter(Boolean);
    if (market.flat().some((c) => !c)) return showError("还有没选择的牌");
    act("/api/game/begin", { market, nobles });
  };
  $("refill-setup").onclick = () => {
    setupDirty = false;
    renderedVersion = -1;
    render();
  };
  $("arm-camera-toggle").onclick = () => act(state.arm.running ? "/api/arm/stop" : "/api/arm/start");
  $("arm-auto-toggle").onclick = () => act("/api/arm/auto", { enabled: !state.arm.auto });
  $("vision-now").onclick = () => act("/api/vision/now");
  $("vision-toggle").onclick = () => act("/api/vision/toggle", { enabled: !state.vision.enabled });
  $("use-my-vision").onclick = () => act("/api/game/use-my-vision");
  $("apply-action").onclick = () => {
    const a = state.game.legal[+$("legal").value];
    if (a && confirm(`录入：${a.text}？`)) act("/api/game/action", { action: a.action });
  };
  $("apply-reveal").onclick = () => {
    const id = $("reveal").value;
    if (id) act("/api/game/reveal", { card_id: id });
  };
  $("confirm-robot").onclick = () => {
    if (confirm("确认人类代理已经按指令执行完毕？")) act("/api/game/confirm-robot");
  };
  $("undo").onclick = () => {
    if (confirm("撤销上一步？实体桌面也需要恢复到上一步的局面。")) act("/api/game/undo");
  };
  poll();
  pollArmPreview();
})();
