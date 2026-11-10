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
TERRA_MODEL = "gpt-6.1-terra"
SUGGESTED_MODELS = ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini", "gpt-5", "gpt-5-mini", "o4-mini"]
DEFAULTS = {
    "vision_provider": "api",
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



def arm_detail_images(jpeg, max_side, *, center_only=False):
    """Overlapping native-resolution views for small print on the same frame."""
    try:
        from PIL import Image
    except ImportError:
        return []
    with Image.open(io.BytesIO(jpeg)) as image:
        width, height = image.size
        if width < 1024:
            return []
        overlap = width // 6
        regions = (
            (width // 5, 0, width * 4 // 5, height * 3 // 4),
            (0, 0, width // 2 + overlap, height),
            (width // 2 - overlap, 0, width, height),
        )
        images = []
        for region in (regions[:1] if center_only else regions):
            out = io.BytesIO()
            detail = image.crop(region).convert("RGB")
            detail.thumbnail((max_side, max_side))
            detail.save(out, format="JPEG", quality=94)
            images.append(out.getvalue())
        return images


ARM_DETAIL_PROMPT = ("图片依次为整桌、中央公共区细节、左侧细节、右侧细节，都来自同一帧。"
                     "用整桌图定位，在中央和两侧细节图辨认数字；有重叠，同一物件只计一次。")


SCHEMA_TEXT = """{
  "visible": true,
  "issues": [],
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
    text = f"""请识别这张《璀璨宝石》(Splendor) 四人对局的俯拍照片，读出公共牌和四位玩家的游戏物件，只输出一个 JSON 对象。

## 物件
- 宝石筹码（圆形）：white 白(钻石)、blue 蓝(蓝宝石)、green 绿(祖母绿)、red 红(红宝石)、black 黑(玛瑙/棕黑)、gold 金(黄金，百搭)。
- 发展卡：分 1/2/3 级。左上角数字是分数(points，没有数字 = 0)；右上角宝石是奖励颜色(color)；左下角若干彩色圆圈里的数字是购买花费(cost，没有出现的颜色为 0)。
- 贵族：方形板块，左侧标出需要的发展卡颜色和数量（两种颜色各 4 张，或三种颜色各 3 张），每位 3 分。

## 桌面布局
- 中央公共区：一排贵族；{settings.get("market_rows") or DEFAULTS["market_rows"]}；{settings.get("slot_order") or DEFAULTS["slot_order"]}。银行筹码叠放，由系统按筹码总量扣除玩家持有量计算。
- 四位玩家的区域（按行动顺序）：
{seats}
- 每位玩家面前有：筹码、已购发展卡（按颜色叠放，数清每种颜色的张数）、预留牌（正面朝上，最多 3 张）、已获得的贵族板块。

## 要求
- 只数四位玩家持有的筹码。每位玩家输出全部六种颜色，确认某色没有才写 0。某色数量读不清时填 null，并注明哪个座位、哪种颜色。银行按颜色叠放，公共银行的筹码不计入任何玩家。不要数银行筹码，也不要用筹码总量倒推玩家持有量。
- market 每行写 4 个位置：空位写 null，被遮挡或看不清写 "?"。
- 如果有区域被遮挡、出了画面、反光看不清，把 visible 设为 false 或在 issues 里用中文写清楚是哪个区域。
- players 必须按 seat 1–4 给出 4 项。没有的东西写 0 或空列表。
- confidence 是你对整体识别结果的把握（0–1）。

## 输出格式（严格 JSON，不要任何解释文字）
{SCHEMA_TEXT}"""
    if hint is not None:
        text += ("\n\n## 参考：上一次确认的局面（实际桌面可能已经变化，一切以照片为准；"
                 "尤其注意当前玩家的区域和新翻出的牌）\n" + json.dumps(hint, ensure_ascii=False))
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


ARM_COVERAGE_PROMPT = """
## 机械臂相机取景检查（必须输出）
额外输出 coverage 对象，包含 market（公共发展卡和贵族）和
seat1、seat2、seat3、seat4（各座位实际摆放筹码和牌的区域）。只要求游戏物件摆放处完整，
不要求看到玩家本人、椅子或整个桌沿。每个区域只填：
- clear：整个区域都在画面内且物件可读，确定为空的完整区域也可填 clear。
- cropped：确认游戏物件摆放处出画或有游戏物件被画面边缘截断。仅桌沿出画不算。
- blocked：确认被手、物件或其他东西遮住。
- unreadable：区域在画面内，但反光、模糊或分辨率导致看不清。
- unknown：无法确定区域的位置、范围或可见性。
只有五项全是 clear 且物件都读清时才输出 visible=true。银行筹码不需要可见，也不计入 issues。以这一帧实际可见的像素为准，
不能用游戏规则、筹码总数、上一帧或预期局面补齐看不清或出画的信息。
不可读的计数、牌、贵族或座位字段用 null 或 ? 表示，不能填 0 或空数组假装看清。
market 的单个槽位只有确定为空时才能填 null，否则用 ?。
"""


def arm_plan_prompt(plan):
    if not plan:
        return ""
    return ("\n## 分区域扫描与下一步取景\n"
            "缺失表列出本轮尚未确认的区域。优先读缺失字段；只输出当前画面确实可核验的数据，"
            "已确认但本帧不可见的字段仍填 null 或 ?，程序会保留已有结果。"
            "贵族列表、各座位的筹码或物件仅在该字段读全时输出，不完整时填 null。"
            "若缺失内容仍看不清，JSON 顶层增加 next_view，填可用动作之一的 id；否则填 null。"
            "center 是该动作在参考画面中看向的位置（x 向右、y 向下），scale 越大物件越大。"
            "结合当前位置与画面中缺失方向选择能一次看清最多缺失信息的位置；字太小优先较高 scale。"
            "运动参数由程序决定，只选 id，不生成角度、坐标或指令。issues 最多两条，短句即可。"
            "\n实测配置与缺失表：" + json.dumps(plan, ensure_ascii=False))



def recognize(settings, jpeg, hint=None, *, arm_coverage=False, arm_plan=None):
    """(model JSON, meta) for one frame; raises VisionError"""
    s = dict(DEFAULTS, **{k: v for k, v in settings.items() if v not in (None, "")})
    if not s.get("api_key"):
        raise VisionError("没有配置 API Key（设置页面）")
    max_side = int(s["image_max_side"])
    image, size = prepare_image(jpeg, min(max_side, 1280) if arm_plan else max_side, int(s["jpeg_quality"]))
    details = arm_detail_images(jpeg, int(s["image_max_side"]), center_only=bool(arm_plan)) if arm_coverage else []
    images = [image, *details]
    prompt = build_prompt(s, hint) + (ARM_COVERAGE_PROMPT if arm_coverage else "") + arm_plan_prompt(arm_plan)
    if len(images) > 1:
        prompt += "\n" + (ARM_DETAIL_PROMPT if len(images) > 2 else "第二张是同一帧的中央细节，不要重复计数。")
    payload = {
        "model": s["model"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                *({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(part).decode(),
                    "detail": s.get("detail") or "high"}} for part in images),
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
            "image_size": size, "image_bytes": sum(map(len, images)), "usage": data.get("usage"),
            "retried_without": [k for k in ("temperature", "response_format") if k not in payload]}
    return result, meta
