# 四人璀璨宝石模拟器

独立于原德扑环境的 Python 模块，带中文网页界面，**仅依赖 Python 3.10+ 标准库**。原德扑代码与训练入口不变。

在仓库根目录运行：

```bash
python -m splendor
```

自动打开 `http://127.0.0.1:8765`。Windows 若只有 `py` 命令，使用 `py -m splendor`。

```bash
python -m splendor --seed 42 --port 8765 --no-browser
python -m splendor --simulate 100 --seed 0
python -m splendor --token-rule hard-cap   # 旧规则：禁止超过 10 枚
python -m unittest discover -s splendor/tests -v
```

网页支持四人轮流手动操作、随机/启发式单步与连续模拟、暂停、撤销、种子复现、JSON 导出与载入。所有玩家均可由本界面控制；不是在线多人房间。自动玩家是基线策略，尚未实现 MCTS 或 AlphaZero。

## 当前约定的规则

| 项目 | 实现 |
|---|---|
| 玩家 | 固定四人，座位 1 为先手，顺序轮流 |
| 库存 | 五色各 7 枚，黄金 5 枚；筹码总数包含黄金 |
| 牌库 | 基础版 90 张发展牌（40/30/20），每层公开 4 张 |
| 贵族 | 从 10 位中随机公开 5 位，每位 3 分 |
| 拿三色 | 银行可用颜色不少于三种时，必须选三种不同色各一枚 |
| 库存不足三色 | 取银行剩余的一色或两色各一枚 |
| 拿同色 | 银行该色至少 4 枚，才能拿 2 枚；不适用于黄金 |
| 十枚上限（默认，官方规则） | 拿币或预留后可以超过 10 枚，但必须在回合结束前逐枚退回到 10 枚（`discard` 阶段），可退任意颜色，包括刚拿的和黄金；退完后才结算贵族与 15 分。旧规则"禁止任何会令持币超过 10 枚的动作"可用 `--token-rule hard-cap` 或 `new_game(rng, discard_over_limit=False)` 启用 |
| 预留 | 只可预留桌面公开牌，始终公开，最多 3 张，不支持从牌堆暗扣 |
| 预留的黄金 | 银行有黄金必须领取，若因此超过 10 枚则退回；无黄金仍可预留 |
| 购买 | 可买桌面或自己的预留牌；永久折扣不消耗；支付的筹码退回银行 |
| 黄金支付 | 可替代任意颜色，包括手里已有该颜色筹码时；枚举全部合法支付结果 |
| 贵族结算 | 每次行动后必须获赠一位满足条件的贵族；多位时进入选择阶段，只取一位 |
| 结束 | 当前玩家完成本次行动及贵族结算后达到至少 15 分，立即结束，不补完本轮 |
| 排名 | 按分数，同分按已购发展牌更少，再相同共享胜利；正常可达对局中即达标玩家胜出 |
| 无合法动作 | 轮到时没有任何合法动作的玩家**被迫跳过**（不是可选动作，动作空间里没有 pass；计一手）。**四人连续跳过**说明谁都不可能再行动，判**平局**（`end_reason="draw"`，四人平分 1）。规则版本 `RULES_VERSION = 2`；版本 1 在这里进入 `blocked` 并停止、不判胜负 |

无合法动作只出现在少数局面：银行没有任何彩色宝石、该玩家已预留 3 张（或桌面无牌）且买不起任何牌。模拟报告把正常结束 `score`、平局 `draw`、运行步数限制 `truncated` 分开统计，并统计跳过次数 `passes`。

## 给后续搜索使用的接口

```python
import random
from splendor import new_game

environment_rng = random.Random(42)
state = new_game(environment_rng)
actions = state.legal_actions()

# 不改变原状态；只在真正触发补牌时，调用传入的 RNG。
next_state = state.apply(actions[0], random.Random(1001))

# 不传 RNG 时，补牌保留为概率节点，适合显式 chance-node 搜索。
reserve = next(a for a in actions if a.kind == "reserve")
chance = state.apply(reserve)
for card_id, probability in chance.chance_outcomes():
    branch = chance.clone()
    branch.resolve_chance(card_id)
    # 在 branch 上继续搜索，以 probability 加权；不能挑对自己最有利的补牌。
```

- `Game` 中只保存每层**无序剩余牌池**，不保存洗牌后的未来顺序，也不保存 RNG。开局只采样十二张公开牌和五位贵族。
- 搜索分支分别提供独立的 RNG；不要把真实环境的 RNG 拷给搜索，也不要以相同种子固定所有分支。自动策略的 RNG 与环境 RNG 独立。
- `apply` 返回克隆；`step` 是校验后的原地转移；概率节点用 `sample_chance(rng)` 或 `resolve_chance(card_id)` 处理。剩余牌抽空后保持空位，不建立空概率节点。
- `legal_actions()` 枚举当前阶段的所有合法动作，包括黄金支付组合和贵族选择。黄金支付相同的结果不会因为替代了不同颜色而重复出现。
- `phase`：`action`（主动作）、`chance`（补牌）、`discard`（退回超过 10 枚的筹码，每次一枚）、`noble`（贵族选择）、`terminal`（已结算：`end_reason` 为 `score` 或 `draw`）。退币与贵族选择都不额外增加行动次数；被迫跳过计一手（`turn` 加一，`passes` 计数）。
- `legal_actions(all_payments=False)` 只列出每次购买中用金币最少的支付方式；`step(action, rng, validate=False)` 跳过合法性复查，仅用于取自当前状态 `legal_actions()` 的动作（搜索与自对弈的热路径）。
- `observation()` 提供公开信息，包括可由完整牌表推导的无序剩余牌池；不含种子、随机数状态或未来牌序。HTTP 控制界面额外展示种子用于复盘，策略应使用引擎观察接口。
- `key()` 可用于搜索中的重复局面合并，不含历史和随机状态；`rewards()` 只在正常结算后返回四位玩家收益，共享胜利平分 1，平局四人各 0.25。
- `assert_valid()` 检查筹码守恒、十枚限制（退币阶段及其之前的补牌阶段允许当前玩家暂时超过）、预留限制、90 张牌唯一性、贵族唯一性及牌层归属。

JSON 存档保存规则版本、初始种子和完整动作，载入时重新执行并验证；错误存档不会覆盖当前游戏。撤销使用相同流程恢复环境随机状态。种子复现面向相同规则版本与 Python 随机实现，不承诺跨版本永久稳定。

本地 API 为 `GET /api/state`、`GET /api/export`、`POST /api/new`、`/api/action`、`/api/bot`、`/api/undo`、`/api/import`；写请求需携带最新 `version`，避免重复点击或陈旧局面修改。默认仅监听本机，单服务实例共享一局游戏。

## 数据依据

- [出版社规则页](https://www.spacecowboys-games.com/game/splendor/) 与 [官方基础规则 PDF](https://cdn.svc.asmodee.net/production-spacecowboys/uploads/2025/10/SCSPL01EN_SPLENDOR_RULES_LIGHT.pdf)。公开预留、立即结束和无跳过是本项目约定，优先于原规则；十枚上限默认按原规则退回。
- `data/cards.csv` 使用 [anicolao/splendor 的实物照片核验数据](https://github.com/anicolao/splendor/blob/main/data/verified_card_properties.csv)，源文件 SHA `e5c5d8ca04e5c2c82a17553fd5c9eb496e1b3a99`。牌 ID 和字段名称保持来源形式，不使用网上部分旧 CSV 中错误的 `blue-L1-03` 费用。
- 十位贵族的需求与 [seal256/splendor 的 NOBLES 常量](https://github.com/seal256/splendor/blob/master/pysplendor/splendor.py) 交叉核对。此处按颜色需求编号 `N1` 至 `N10`，不使用牌面人物图片。

界面全部用 HTML/CSS 绘制，不包含商业牌面美术资源。
