# 璀璨宝石（Splendor）最终模型

当前版本：自动实验循环 `auto_research` 第 8 轮（R008）第 190 个 iteration 的网络，加上部署搜索。2026-11-09 05:30 更新。

- 之前的版本：R008 第 90 个 iteration，在 `r008_it90/`；R007 第 200 个 iteration，在 `r007_it200/`；R007 第 80 个 iteration，在 `it80/`。
- 2026-11-07 用 R007 it80 首次确认目标达成。

**目标"超越一般水平玩家"：已达到。** 目标的代理是：在 4 人局中，1 个模型座位对 3 个 lookahead 规则机器人：
- 胜率份额 ≥ 0.50，且 95% 区间下界 > 0.40；
- 同时对 3 个 heuristic 机器人 ≥ 0.70。

势均力敌时份额为 0.25。记录页：https://notebook.infimory.ai/splendor-auto.html

## 文件

| 文件 | 内容 |
|---|---|
| `weights.pth` | 网络权重，`experiments/auto_r008/checkpoints/weights_iteration_00190.pth` 的副本 |
| `config.yml` | 训练配置 `config/splendor_auto_r008.yml` 的副本（网络结构、搜索参数），文件末尾注明了部署搜索 |
| `play_agent.py` | 单进程的智能体：`Agent.act(env)`，以及一个对规则机器人下几局的演示 |
| `benchmarks/` | 确认基准（json；其中 `model` 字段是基准时用的临时副本路径，权重与 `weights.pth` 相同） |
| `MD5SUMS` | 校验和 |
| `r008_it90/` | 上一版（R008 第 90 个 iteration）及其基准、说明 |
| `r007_it200/` | R007 第 200 个 iteration |
| `it80/` | 首次达标的版本（R007 第 80 个 iteration） |

## 部署搜索

Gumbel MCTS（`rl/splendor_mcts.py`），以下设置都在 `config.yml` 里：

- 512 次模拟（`--sims 512`）。
- 根节点考虑 8 个动作（`eval_mcts_max_num_considered_actions: 8`）。
- **对手模型** `eval_mcts_opponent_policy: prior`：搜索树里其他玩家走网络先验的 argmax、不分支，模拟次数都花在自己的回合上。这是本模型变强的关键，见下面"经验"。
- greedy（gumbel_scale 0），value_scale 0.1，不做 Q 缩放。

观测是**版本 2**（`splendor/rl_env.OBSERVATION_VERSION = 2`，348 个字段，每张牌 / 贵族带"还差多少"的派生字段），要用当前代码加载。

## 强度

独立基准 `benchmark_splendor.py`，1 个模型座位对 3 个机器人，同一副发牌模型轮流坐四个座位，512 次模拟。发牌用确认专用的一批（`--seed 5000000000`），从没用于选择设置或检查点。

| 对手（3 个） | 局数 | **R008 it190（当前）** | R008 it90 | R007 it200 | R007 it80 | 该机器人自身对 3×lookahead |
|---|---|---|---|---|---|---|
| lookahead（"一般玩家"代理） | 800 | **0.84**（0.815–0.865） | 0.7562 | 0.6975 | 0.5637 | 0.25 |
| heuristic | 400 | **0.96**（0.942–0.978） | 0.9275 | 0.8925 | 0.7825 | 0.115 |
| lookahead_alt1（没见过：权重扰动的 lookahead） | 400 | **0.84**（0.807–0.873） | 0.7675 | 0.7125 | 0.5975 | 0.2225 |
| lookahead_alt2（没见过） | 400 | **0.7925**（0.753–0.832） | 0.79 | 0.7325 | 0.62 | 0.2575 |

直接对局（两个座位对两个座位，与训练中评估相同的安排，势均力敌为 0.5）：
- R008 it190 对 R008 it90：0.6042（192 局，0.533–0.675）；
- R008 it90 对 R007 it200：0.5903（288 局）；
- R007 it200 对 it80：0.7188（96 局）。

作为对照，同样的发牌下 lookahead 对 3×heuristic 是 0.7975。

## 用法

独立基准：

```
python benchmark_splendor.py --config models/splendor_final/config.yml --model models/splendor_final/weights.pth \
    --opponents bot:lookahead bot:heuristic --games 400 --sims 512 --num-game-process 20 --out result.json
```

在代码里使用：

```python
from models.splendor_final.play_agent import Agent   # 在仓库根目录运行，或把目录加进 sys.path
agent = Agent(device='cuda')                         # 默认 512 次模拟、considered 8、opponent_policy prior
action_index = agent.act(env)                        # env: splendor.rl_env.SplendorRLEnv，处在该玩家的决策状态
```

演示：`python models/splendor_final/play_agent.py --games 4 --opponent lookahead`。单进程运行，每步一次搜索，不跨局批量，所以很慢。

作为新实验的起点：在配置里设 `init_weights_path: models/splendor_final/weights.pth`、`init_weights_strict: True`。

**网页对弈**：`splendor/server.py` 目前只有规则机器人，不加载网络模型；接入网络还没有实现。

## 训练链

| 轮 | 改动 | 对 3×lookahead |
|---|---|---|
| v4（循环之前） | — | 0.24 |
| R001–R005 | 学习率衰减、关 Q 缩放、TD(λ) 价值目标、对手多样化、观测版本 2 | 0.21 → 0.295（平台） |
| R006 | 策略带教：自对弈记录的策略目标混入 25% 的 lookahead 动作，修复先验塌缩 | 旧标准 0.3175；部署搜索 0.43 |
| R007 | 自对弈与评估的搜索都用对手模型（prior），considered 8；12.9 h，200 个 iteration | it80 0.5637 → it200 0.6975（512 次，确认发牌） |
| R008 | 从 R007 it200 继续，学习率 2e-5 并按训练步衰减（1.0 → 0.75 → 0.5） | it90 0.7562 → it190 **0.84**；训练继续中（上限 300 个 iteration） |

细节、诊断与判定见 `auto_research/rounds/R00N/`（plan.md、analysis.md、decision.json）和 `auto_research/program.md`。

## 经验

1. **先验塌缩。** 模型几乎从不"同色拿 2 个"、很少预留 3 级牌：可走时先验质量只有 0.3%。Gumbel 的 σ(q) 项太小，抬不起这些动作。用规则机器人带教可以修复。
2. **4 人局里搜索的价值取决于对手模型。**
   - 对 lookahead 的平台不是网络的问题：max^n 把对手当成"像我一样"的玩家，又把模拟次数花在对手节点的分支上，到不了自己的下一回合。
   - 对手走先验 argmax 之后：
     - 同一网络对 lookahead 从 0.3175 提到 0.43；
     - 更多模拟次数也开始有用（256 → 512 次：+0.05）；
     - 训练也持续有效（R007 it80 → it200：+0.13；R008 it90：+0.06；R008 it190：再 +0.08）。

## 已知局限

- 带教老师就是基准对手 lookahead，网络先验因此是它的好模型。
  - 复核：对两个没见过的权重扰动变体（`splendor/bots.py` 的 `lookahead_alt1` / `lookahead_alt2`），份额与对 lookahead 本身相近（0.84 / 0.79 对 0.84）。
  - 但它们和 lookahead 是同一算法。对真人玩家的强度没有测过。
- 4 人局方差大：400 局的 95% 区间约 ±0.04–0.05。
- 512 次模拟在单进程里每步要几秒（GPU）；基准用的是批量对局。
