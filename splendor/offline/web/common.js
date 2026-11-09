"use strict";
const $ = (id) => document.getElementById(id);
const CN = ["白", "蓝", "绿", "红", "黑", "金"];
const COLORS = ["white", "blue", "green", "red", "black", "gold"];
const gemStyle = (c) => `background:var(--${COLORS[c]})`;
const esc = (s) =>
  String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const r = await fetch(path, options);
  let data = {};
  try {
    data = await r.json();
  } catch (e) {
    data = { error: `HTTP ${r.status}` };
  }
  if (r.status === 401 && !location.pathname.startsWith("/login")) {
    location.href = "/login";
    throw new Error("请先登录");
  }
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

async function logout() {
  await api("/api/logout", {});
  location.href = "/login";
}

let CARDS = null;
async function loadCards() {
  if (!CARDS) CARDS = await api("/api/cards");
  return CARDS;
}

function costHTML(cost) {
  return cost
    .map((n, c) => (n ? `<span class="cost"><i class="gem" style="${gemStyle(c)}"></i>${n}</span>` : ""))
    .join("");
}

function cardHTML(id, extraClass = "") {
  if (id === null || id === undefined) return `<div class="card empty ${extraClass}">空位</div>`;
  if (id === "?") return `<div class="card empty ${extraClass}">未识别</div>`;
  const c = CARDS.cards[id];
  return `<div class="card ${extraClass}" title="${esc(c.text)}">
    <div class="top"><span class="pts">${c.points || ""}</span><span class="lvl">${c.level}级</span>
    <i class="gem" style="${gemStyle(c.bonus)};width:18px;height:18px"></i></div>
    <div class="costs">${costHTML(c.cost)}</div></div>`;
}

function nobleHTML(id) {
  const n = CARDS.nobles[id];
  return `<div class="noble" title="${esc(n.text)}"><b>3 分</b><div class="costs">${costHTML(n.cost)}</div></div>`;
}

function tokensHTML(tokens) {
  return tokens
    .map((n, c) => `<span class="chip"><i class="gem" style="${gemStyle(c)}"></i>${n}</span>`)
    .join("");
}

function timeText(t) {
  if (!t) return "—";
  const d = new Date(t * 1000);
  return d.toLocaleTimeString("zh-CN", { hour12: false });
}

function alertsHTML(alerts) {
  return (alerts || [])
    .map((a) => `<div class="alert ${a.level}"><b>${esc(a.text)}</b>${
      a.details && a.details.length ? `<ul>${a.details.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>` : ""
    }</div>`)
    .join("");
}
