# 璀璨宝石（Splendor）最终模型

自动实验循环 `auto_research` 第 7 轮（R007）第 80 个 iteration 的网络，加上部署搜索。

**2026-11-07 达到目标"超越一般水平玩家"。** 目标的代理是：在 4 人局中，1 个模型座位对 3 个 lookahead 规则机器人：
- 胜率份额 ≥ 0.50，且 95% 区间下界 > 0.40；
- 同时对 3 个 heuristic 机器人 ≥ 0.70。

势均力敌时份额为 0.25。记录页：https://notebook.infimory.ai/splendor-auto.html

## 文件

| 文件 | 内容 |
|---|---|
| `weights.pth` | 网络权重，`experiments/auto_r007/checkpoints/weights_iteration_00080.pth` 的副本 |
| `config.yml` | 训练配置 `config/splendor_auto_r007.yml` 的副本（网络结构、搜索参数），文件末尾注明了部署搜索 |
| `play_agent.py` | 单进程的智能体：`Agent.act(env)`，以及一个对规则机器人下几局的演示 |
| `benchmarks/` | 达标确认与检查时的独立基准（json） |
| `MD5SUMS` | 校验和 |

## 部署搜索

Gumbel MCTS（`rl/splendor_mcts.py`），以下设置都在 `config.yml` 里：

- 512 次模拟（`--sims 512`）。
- 根节点考虑 8 个动作（`eval_mcts_max_num_considered_actions: 8`）。
- **对手模型** `eval_mcts_opponent_policy: prior`：搜索树里其他玩家走网络先验的 argmax、不分支，模拟次数都花在自己的回合上。这是本模型变强的关键，见下面"经验"。
- greedy（gumbel_scale 0），value_scale 0.1，不做 Q 缩放。

观测是**版本 2**（`splendor/rl_env.OBSERVATION_VERSION = 2`，348 个字段，每张牌 / 贵族带"还差多少"的派生字段），要用当前代码加载。

## 强度

独立基准 `benchmark_splendor.py`，1 个模型座位对 3 个机器人，同一副发牌模型轮流坐四个座位。

**达标确认：没用过的发牌（`--seed 5000000000`），512 次模拟**

| 对手（3 个） | 局数 | 模型胜率份额（95% 区间） | 该机器人自身对 3×lookahead |
|---|---|---|---|
| lookahead（"一般玩家"代理） | 800 | **0.5637**（0.531–0.597） | 0.25 |
| heuristic | 400 | **0.7825**（0.744–0.821） | 0.115 |
| lookahead_alt1（没见过：权重扰动的 lookahead） | 400 | **0.5975**（0.554–0.641） | 0.2225 |
| lookahead_alt2（没见过） | 400 | **0.62**（0.568–0.672） | 0.2575 |

**检查时的发牌（seed 4e9，400 局）**

| 对手 | 256 次 | 512 次 |
|---|---|---|
| lookahead | 0.4825 | 0.5325 |
| heuristic | 0.775 | — |
| lookahead_alt1 / alt2 | 0.52 / 0.5525 | — |
| 对 R006 latest（96 局，256 次） | 0.583 | — |

作为对照，同样的发牌下 lookahead 对 3×heuristic 是 0.7975。

## 用法

独立基准：

```
python benchmark_splendor.py --config models/splendor_final/config.yml --model models/splendor_final/weights.pth \
    --opponents bot:lookahead bot:heuristic --games 400 --sims 512 --num-game-process 20 --out result.json
```

在代码里使用：

```python
from models.splendor_final.play_agent import Agent   # 或者把目录加进 sys.path
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
| R007 | 自对弈与评估的搜索都用对手模型（prior），considered 8 | 0.4825（256 次）/ 0.5637（512 次，确认） |

细节、诊断与判定见 `auto_research/rounds/R00N/`（plan.md、analysis.md、decision.json）和 `auto_research/program.md`。

## 经验

1. **先验塌缩。** 模型几乎从不"同色拿 2 个"、很少预留 3 级牌：可走时先验质量只有 0.3%。Gumbel 的 σ(q) 项太小，抬不起这些动作。用规则机器人带教可以修复。
2. **4 人局里搜索的价值取决于对手模型。**
   - 对 lookahead 的平台不是网络的问题：max^n 把对手当成"像我一样"的玩家，又把模拟次数花在对手节点的分支上，到不了自己的下一回合。
   - 对手走先验 argmax 之后：
     - 同一网络对 lookahead 从 0.3175 提到 0.43；
     - 更多模拟次数也开始有用（256 → 512 次：+0.05）。

## 已知局限

- 带教老师就是基准对手 lookahead，网络先验因此是它的好模型。
  - 复核：对两个没见过的权重扰动变体（`splendor/bots.py` 的 `lookahead_alt1` / `lookahead_alt2`），份额不低于对 lookahead 本身。
  - 但它们和 lookahead 是同一算法。对真人玩家的强度没有测过。
- 4 人局方差大：400 局的 95% 区间约 ±0.05，相邻检查点之间约 ±0.04。
- "从预留区买牌"仍少于 lookahead（每局约 0.3–0.5 次，lookahead 0.6–0.7）。
- 512 次模拟在单进程里每步要几秒（GPU）；基准用的是批量对局。
