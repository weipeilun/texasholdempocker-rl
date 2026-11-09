# R007 计划：搜索里用对手模型

## 这一轮要回答的问题

R006（策略带教）让塌缩的动作类回来了，对 heuristic 到 0.5375（各轮最高），对 R005 0.5625，对 lookahead 仍是 0.28。

R006 的诊断找到了"对 lookahead 卡住"的真正原因：4 人局里多看一层自己的回合，**只有在对手模型准确时才有用**。

lookahead2 的做法：每个动作之后让对手按模型走到我的下一回合，再评估（`rounds/R006/diag/lookahead2.json`）。

| 对手模型 | 对手 | 胜率份额（lookahead 自身） |
|---|---|---|
| heuristic（不准） | 3×lookahead | 0.25（0.25） |
| lookahead（准确） | 3×lookahead | 0.39（0.25） |
| heuristic（准确） | 3×heuristic | 0.89（0.80） |

之前的搜索是 max^n，每个对手节点都用自己的网络分支选择。这带来两个问题：
- 对手模型是"像我一样的玩家"。在自对弈里这是准确的，所以自对弈强度一直在涨；对规则机器人则不准。
- 256 次模拟分到 32 个根动作、再在每个对手节点分支，几乎到不了自己的下一回合。

于是新增 `opponent_policy`（rl/splendor_mcts.py）：其他玩家走网络先验的 argmax、不分支（`prior`）。R006 第 90 个 iteration 上的评估（256 次模拟，`/tmp/r007/diag`）：

| 搜索 | vs 3×lookahead | vs 3×heuristic |
|---|---|---|
| considered 32，64 次模拟，max_n（之前的标准） | 0.28 | 0.5375 |
| considered 8，256 次，max_n | 0.335 | 0.605 |
| **considered 8，256 次，prior** | **0.3875**（0.344–0.431） | **0.6475**（0.60–0.69） |
| considered 8，256 次，bot:lookahead（准确对手模型，仅作上限） | 0.4025 | — |

- R005 加深搜索只有 +0.015–0.03，R006 的先验经过带教，加深搜索有了效果。
- prior 已经接近"准确对手模型"的上限（0.3875 vs 0.4025）。
- bot:lookahead 在基准里等于已知对手的代码，只作参考，不作方案。

本轮回答：在自对弈搜索里也用对手模型，搜索目标从此包含"自己下一回合"的信息，能否让网络本身继续变强。

## 改动（对照 R006）

| 内容 | R006 | R007 |
|---|---|---|
| 搜索中的其他玩家 | max_n | **prior**（`mcts_opponent_policy` / `eval_mcts_opponent_policy`） |
| 全量搜索考虑的动作 | 32 | **8**（256 次模拟） |
| 训练中的评估与基准 | 64 次模拟，considered 32 | **128 次，considered 8，prior** |
| 起点 | R005 latest | **R006 latest** |

其余相同：
- 带教 w = 0.25（同时让先验保持为"一般玩家"的模型）；
- 观测版本 2；
- 50% 对局有 1–2 个机器人座位；
- TD(λ) 0.95；
- lr 5e-5，6000 步 ×0.5，12000 步 ×0.3。

**部署搜索 = 256 次模拟、considered 8、prior。** 从本轮起，独立基准同时报两种：
- 旧标准（64 次、considered 32、max_n，与前几轮可比）；
- 部署搜索（`--config` 用 R007 配置、`--sims 256`）。

## 判据（部署搜索）

- **supported**：最后的检查点 vs 3×lookahead ≥ 0.50、vs 3×heuristic ≥ 0.70，对 R006 latest ≥ 0.55。
  - 原定 0.45；看到 R006 latest 的部署搜索基线是 0.43 后，在 R007 有独立基准数据之前改为 0.50。
- **refuted**：vs lookahead ≤ R006 latest（部署搜索）+ 0.02，且对 R006 < 0.52。
- 注意：带教老师就是基准对手 lookahead，先验因此是 lookahead 的好模型。
  - 对 heuristic（不是老师）的结果是泛化的证据。
  - 达标后还要对一个没见过的对手（例如权重扰动过的 lookahead）复核。

## 运行记录
- 2026-11-07 09:52 启动（pid 448434），严格加载 R006 latest（98 个张量）。
  - R006 latest 的部署搜索基线：vs 3×lookahead 0.43（0.38–0.48）、vs 3×heuristic 0.665（0.62–0.71）。
  - refuted 线：vs lookahead ≤ 0.45 且对 R006 < 0.52。
  - 前 5 个 iteration：
    - 2.7 → 5.9 局/s（前两个 iteration 与 R006 收尾基准同时跑）；
    - 老师一致率 0.443–0.458；
    - 第 4 个 iteration 的内置基准（128 次模拟、considered 8、prior，96 局）：vs heuristic 0.635、vs lookahead 0.458。
- 2026-11-07 12:50 第 1 次检查（2.97 h，50 个 iteration，3364 步，lr 3.6e-5，5.6–5.9 局/s）：
  - 老师一致率 0.449 → 0.467。held-out value_ce 0.991 → 0.980。第 29 个 iteration 晋升。
  - 内置基准（128 次、considered 8、prior，96 局）：
    - vs heuristic 0.635 → 0.729（it34–44 都 ≥ 0.70）；
    - vs lookahead 0.417–0.49，无趋势。
  - **独立基准（it50，部署搜索，400 局）**：
    - vs lookahead **0.4525**（0.403–0.502，R006 latest 0.43）；
    - vs heuristic **0.75**（0.704–0.796，R006 latest 0.665，**过了 0.70 的达标线**）；
    - 对 R006 latest **0.604**（96 局，0.51–0.70）。
  - 没见过的对手 `lookahead_alt1` / `lookahead_alt2`（splendor/bots.py，见 `ladder_alt.json`）：
    - 做法：lookahead 的权重各乘 exp(U(−0.25, 0.25))；种子在 1–10 中只按强度挑最接近 lookahead 的两个，不看模型的成绩。
    - 阶梯：alt1 对 3×lookahead 0.2225、对 3×heuristic 0.7225；alt2 为 0.2575 / 0.7825。lookahead 对 3×alt1 / 3×alt2 为 0.26 / 0.275。
    - 它们不当老师、不进自对弈。之后的独立基准都加上它们。
    - 扰动 ±0.4 的第一版里，seed 1 对 3×heuristic 只有 0.05（权重组合坏掉），已弃用。
  - 判断：对 heuristic 达标，对 lookahead 还差 0.05，继续。
  - 风格（it50，先验贪心，对 3×heuristic）：同色拿 2 个 3.27、预留 3 级牌 1.645、从预留区买牌 0.32、买 1 / 2 / 3 级牌 6.4 / 3.5 / 0.98、贵族 0.22、抢先预留 0.175。离 lookahead 的风格（4.24 / 1.99 / 0.59 / 3.3 / 3.6 / 1.5 / 0.06 / 0.265）更近了。
- 没见过的对手（部署搜索，400 局，`benchmark_watch1_it50_alt.json`、`../R006/benchmark_end_deploy_alt.json`）：

| 模型 | vs 3×lookahead_alt1 | vs 3×lookahead_alt2 | vs 3×lookahead（对照） |
|---|---|---|---|
| R006 latest | 0.40（0.36–0.44） | 0.5225（0.47–0.57） | 0.43 |
| R007 it50 | **0.50**（0.45–0.55） | **0.535**（0.49–0.58） | 0.4525 |

  对没见过的 lookahead 变体，份额不低于对老师本身（alt 自身比 lookahead 略弱：对 3×lookahead 0.2225 / 0.2575），说明强度不是靠记住老师的走法。
- 2026-11-07 15:20 第 2 次检查（5.47 h，87 个 iteration，lr 2.6e-5）：
  - 第 64 / 79 / 84 个 iteration 晋升。老师一致率约 0.47。held-out value_ce 0.981。
  - 内置基准：vs heuristic 0.69–0.875（it69 0.875、it84 0.844），vs lookahead 0.40–0.46，无趋势。
  - **独立基准（it80，部署搜索 256 次，400 局，`benchmark_watch2_it80.json`）**：

| 对手 | it50 | **it80** |
|---|---|---|
| 3×lookahead | 0.4525 | **0.4825**（0.437–0.528） |
| 3×heuristic | 0.75 | **0.775**（0.728–0.822） |
| 3×lookahead_alt1 | 0.50 | **0.52**（0.469–0.571） |
| 3×lookahead_alt2 | 0.535 | **0.5525**（0.504–0.601） |
| 对 R006 latest（96 局） | 0.604 | 0.583（0.49–0.68） |

  - **更多模拟次数**（it80，`diag/it80_c8_s512.json`）：512 次、considered 8 时 vs 3×lookahead **0.5325**（0.486–0.579）。这是第一次点估计过 0.50、区间下界 > 0.40。
    - 对手模型让搜索深度有了用处：R005 上加模拟次数没用，现在 256 → 512 次 +0.05。
  - 风格（it80，先验贪心）：
    - 对 3×heuristic 份额 0.62（不搜索），同色拿 2 个 3.2，买 3 级牌 1.06，从预留区买牌 0.315；
    - 对 3×lookahead 份额 0.29，从预留区买牌 0.54。
  - 下一步：达标确认。
    - 同一检查点（it80，md5 71995a06…），部署搜索改为 512 次；
    - 换一批没用过的发牌（`--seed 5000000000`）：vs lookahead 800 局，vs heuristic / alt1 / alt2 各 400 局（`confirm_it80_s512_*.json`）。
    - 先等 considered 16 / 4 的诊断跑完。
- 2026-11-07 16:05 评估搜索的诊断（it80，vs 3×lookahead，400 局，seed 4e9）：
  - considered 8 + 512 次：0.5325；
  - considered 16 + 512 次：0.52；
  - considered 4 + 256 次：0.505。
  - 部署搜索定为 considered 8 + 512 次。
- **2026-11-07 17:10 达标确认**（it80，部署搜索 512 次，没用过的发牌 `--seed 5000000000`；`confirm_it80_s512_*.json`）：

| 对手（3 个） | 局数 | 胜率份额（95% 区间） |
|---|---|---|
| lookahead | 800 | **0.5637**（0.531–0.597） |
| heuristic | 400 | **0.7825**（0.744–0.821） |
| lookahead_alt1（没见过） | 400 | **0.5975**（0.554–0.641） |
| lookahead_alt2（没见过） | 400 | **0.62**（0.568–0.672） |

  - **目标达成**：vs lookahead ≥ 0.50 且下界 0.531 > 0.40；vs heuristic ≥ 0.70。
  - 模型与配置保存在 `models/splendor_final/`：
    - weights.pth（md5 71995a06…）、config.yml（末尾注明部署搜索）；
    - play_agent.py（`Agent.act(env)`，已用 64 次模拟、2 局试跑通过）；
    - README.md、benchmarks/、MD5SUMS。
  - R007 继续跑到预算结束（18:52）。之后写 analysis / decision，并看后面的检查点是否更强。
- 2026-11-07 22:45 跑满 200 个 iteration，正常结束。收尾时会话连不到服务器，之后服务器重启。
- 2026-11-08 R007 latest（it200）确认基准（512 次，seed 5e9）：
  - vs 3×lookahead **0.6975**（0.664–0.731，800 局）；
  - vs heuristic 0.8925；
  - vs alt1 / alt2 0.7125 / 0.7325；
  - 对 it80 0.7188（96 局）。
  - models/splendor_final 换成 it200，it80 移到 `it80/`。见 analysis.md。
