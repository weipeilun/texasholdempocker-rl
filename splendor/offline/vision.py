"""Reading the table with a vision LLM over an OpenAI-compatible Chat Completions API: OpenAI itself or a relay
("中转") that speaks the same protocol, with the user's own API key and model.

recognize(settings, jpeg, hint) sends one camera frame and returns the model's JSON (schema in build_prompt) plus
call metadata; table.normalize_vision turns it into a TableState."""

from __future__ import annotations

import base64
import io
import json
import re
import time

import requests

DEFAULT_BASE_URL = "https://api.openai.com/v1"
SUGGESTED_MODELS = ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini", "gpt-5", "gpt-5-mini", "o4-mini"]
DEFAULTS = {
    "base_url": DEFAULT_BASE_URL,
    "api_key": "",
    "model": "gpt-4o",
    "interval": 5,                 # seconds between two readings
    "image_max_side": 2048,        # the long side of the frame sent to the model (pixels)
    "jpeg_quality": 90,
    "detail": "high",
    "timeout": 90,
    "hint_expected": True,         # send the last confirmed state as context
    "layout": ["画面下方（靠近画面底边）", "画面左侧", "画面上方", "画面右侧"],
    "market_rows": "公共区三行发展卡从上到下依次为 3 级、2 级、1 级；每行最左边是该级牌堆（背面朝上），右边 4 张正面朝上",
    "slot_order": "每行的 4 张牌按画面中从左到右的顺序",
    "extra_prompt": "",
}


class VisionError(Exception):
    pass


def _url(base_url, path):
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _headers(api_key):
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def list_models(base_url, api_key, timeout=20):
    """model ids the endpoint offers (GET /models); raises VisionError with the server's message"""
    if not api_key:
        raise VisionError("没有填写 API Key")
    try:
        r = requests.get(_url(base_url, "models"), headers=_headers(api_key), timeout=timeout)
    except requests.RequestException as e:
        raise VisionError(f"无法连接 {base_url}：{e}") from e
    if r.status_code != 200:
        raise VisionError(f"HTTP {r.status_code}：{r.text[:300]}")
    try:
        data = r.json()
    except ValueError as e:
        raise VisionError(f"返回的不是 JSON：{r.text[:200]}") from e
    items = data.get("data", data if isinstance(data, list) else [])
    return sorted({m.get("id") if isinstance(m, dict) else str(m) for m in items} - {None})


def prepare_image(jpeg, max_side=2048, quality=90):
    """downscale (keeping the aspect ratio) and re-encode if the frame is larger than max_side"""
    try:
        from PIL import Image
    except ImportError:
        return jpeg, None
    image = Image.open(io.BytesIO(jpeg))
    size = image.size
    if max(size) <= max_side:
        return jpeg, size
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=int(quality))
    return out.getvalue(), image.size


SCHEMA_TEXT = """{
  "visible": true,
  "issues": [],
  "bank": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0, "gold": 0},
  "nobles": [{"cost": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0}}],
  "market": {"1": [CARD, CARD, CARD, CARD], "2": [CARD, CARD, CARD, CARD], "3": [CARD, CARD, CARD, CARD]},
  "players": [
    {"seat": 1, "tokens": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0, "gold": 0},
     "cards": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0},
     "reserved": [CARD], "nobles": [{"cost": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0}}]}
  ],
  "confidence": 0.9
}
CARD = {"level": 1, "color": "red", "points": 0, "cost": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0}}"""


def build_prompt(settings, hint):
    layout = settings.get("layout") or DEFAULTS["layout"]
    seats = "\n".join(f"  - 玩家{i + 1}（seat {i + 1}）：{where}" for i, where in enumerate(layout[:4]))
    text = f"""请识别这张《璀璨宝石》(Splendor) 四人对局的俯拍照片，读出桌面上的所有游戏物件，只输出一个 JSON 对象。

## 物件
- 宝石筹码（圆形）：white 白(钻石)、blue 蓝(蓝宝石)、green 绿(祖母绿)、red 红(红宝石)、black 黑(玛瑙/棕黑)、gold 金(黄金，百搭)。
- 发展卡：分 1/2/3 级。左上角数字是分数(points，没有数字 = 0)；右上角宝石是奖励颜色(color)；左下角若干彩色圆圈里的数字是购买花费(cost，没有出现的颜色为 0)。
- 贵族：方形板块，左侧标出需要的发展卡颜色和数量（两种颜色各 4 张，或三种颜色各 3 张），每位 3 分。

## 桌面布局
- 中央公共区：一排贵族；{settings.get("market_rows") or DEFAULTS["market_rows"]}；{settings.get("slot_order") or DEFAULTS["slot_order"]}；银行筹码按颜色分堆。
- 四位玩家的区域（按行动顺序）：
{seats}
- 每位玩家面前有：筹码、已购发展卡（按颜色叠放，数清每种颜色的张数）、预留牌（正面朝上，最多 3 张）、已获得的贵族板块。

## 要求
- 数清每一堆筹码；同色筹码总数应为 7 枚（黄金 5 枚），银行与四位玩家合计。数不清时如实给出最可能的数，并在 issues 里说明。
- market 每行写 4 个位置：空位写 null，被遮挡或看不清写 "?"。
- 如果有区域被遮挡、出了画面、反光看不清，把 visible 设为 false 或在 issues 里用中文写清楚是哪个区域。
- players 必须按 seat 1–4 给出 4 项。没有的东西写 0 或空列表。
- confidence 是你对整体识别结果的把握（0–1）。

## 输出格式（严格 JSON，不要任何解释文字）
{SCHEMA_TEXT}"""
    if hint is not None:
        text += ("\n\n## 参考：上一次确认的局面（实际桌面可能已经变化，一切以照片为准；"
                 "尤其注意当前玩家的区域、银行和新翻出的牌）\n" + json.dumps(hint, ensure_ascii=False))
    if settings.get("extra_prompt"):
        text += "\n\n## 补充说明\n" + settings["extra_prompt"]
    return text


SYSTEM = "你是桌游《璀璨宝石》(Splendor) 的牌局识别器。你只输出 JSON。"


def parse_json(text):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def recognize(settings, jpeg, hint=None):
    """(model JSON, meta) for one frame; raises VisionError"""
    s = dict(DEFAULTS, **{k: v for k, v in settings.items() if v not in (None, "")})
    if not s.get("api_key"):
        raise VisionError("没有配置 API Key（设置页面）")
    image, size = prepare_image(jpeg, int(s["image_max_side"]), int(s["jpeg_quality"]))
    url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
    payload = {
        "model": s["model"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": build_prompt(s, hint)},
                {"type": "image_url", "image_url": {"url": url, "detail": s.get("detail") or "high"}},
            ]},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    started = time.time()
    attempts = []
    for _ in range(3):
        try:
            r = requests.post(_url(s["base_url"], "chat/completions"), headers=_headers(s["api_key"]),
                              json=payload, timeout=float(s["timeout"]))
        except requests.RequestException as e:
            raise VisionError(f"无法连接 {s['base_url']}：{e}") from e
        if r.status_code == 200:
            break
        body = r.text[:400]
        attempts.append(f"HTTP {r.status_code}: {body}")
        lowered = body.lower()
        # some models / relays reject these optional parameters: retry without them
        if r.status_code == 400 and "temperature" in lowered and "temperature" in payload:
            payload.pop("temperature")
            continue
        if r.status_code == 400 and ("response_format" in lowered or "json_object" in lowered) \
                and "response_format" in payload:
            payload.pop("response_format")
            continue
        raise VisionError(f"视觉模型返回错误 HTTP {r.status_code}：{body}")
    else:
        raise VisionError("视觉模型多次返回错误：" + " | ".join(attempts))
    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):     # some relays return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        result = parse_json(content)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise VisionError(f"无法解析视觉模型的输出：{r.text[:400]}") from e
    meta = {"model": data.get("model", s["model"]), "latency": round(time.time() - started, 2),
            "image_size": size, "image_bytes": len(image), "usage": data.get("usage"),
            "retried_without": [k for k in ("temperature", "response_format") if k not in payload]}
    return result, meta
