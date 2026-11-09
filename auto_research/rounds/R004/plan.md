# R004 计划：对手多样化

## 这一轮要回答的问题

R003 的模型越来越善于对付自己的旧版本：第 100 个 iteration 对 R002 latest 的直接对局是 0.615。但对规则机器人的独立基准
从第 50 个 iteration 起就停在 0.26–0.275。自对弈的样本几乎全来自"四个相似的我"，学到的打法和价值都是针对这种对手的。
lookahead 这类对手抢牌更早、更少囤币、更早卡位贵族，我们只在 15% 的对局里、每局 1 个座位见过它。

本轮回答：让训练对手多样化后，对机器人的强度能否越过平台，同时自对弈强度不崩。

## 改动（对照 R003）

| 参数 | R003 | R004 | 对应指标 |
|---|---|---|---|
| self_play_bot_prob | 0.15 | **0.5** | metrics 的 bot_games 占比约 50% |
| self_play_bot_seat_counts | —（1 个座位） | **[1, 2]**：有机器人的对局里 1 或 2 个座位，各占一半 | bot_game_model_seat_share（持平 0.25） |
| learning_rate / lr_schedule | 5e-5，6000 步 0.5，12000 步 0.3 | **3e-5**，6000 步 0.6，12000 步 0.4 | lr |
| 起点 | R002 latest | **R003 latest**（第 104 个 iteration），新优化器，回放池从空开始 | — |

- 其余同 R003：TD(λ) 0.95、关缩放、价值权重 1.0、全量 0.15 × 256、终局 13 分以上全量搜索。
- 机器人是 lookahead 与 heuristic 各半，每个机器人座位独立抽取。
- 机器人不产生样本。有机器人时每局平均 3.25 个模型座位，样本数略少。

## 代码

- `train_splendor.self_play_seats`：新增 `self_play_bot_seat_counts`。不设时抽法与 R002 / R003 相同，同样的种子得到同样的座位。
- 本轮部署了 R004 诊断用的启发式叶节点评估（`mcts_heuristic_leaf_weight`、`mcts_heuristic_temperature`、`eval_mcts_*`）和 `eval_mcts_value_scale`。
  本轮都不启用（默认 0 / 不设），留给之后的轮次。
- `loop.py stop` 先只向训练主进程发 SIGINT。
- 新测试：`HeuristicLeafTest`（4 个，含多机器人座位）。

## 判定

- **supported**：同时满足三条。
  - 独立基准（seed 4e9，400 局）中最后的检查点 vs 3×lookahead ≥ 0.35（起点 R003 latest 0.295，见执行记录）；
  - vs 3×heuristic ≥ 0.45；
  - 对 R003 latest 的直接对局 ≥ 0.45。
- **refuted**：vs lookahead < 0.31（不高于起点 + 0.02），或者对 R003 latest < 0.40（为对付机器人而丢了一般强度）。
- **inconclusive**：其余情况。

## 风险

- 搜索把机器人座位当成和自己一样的玩家来建模（用网络的先验与价值），对它们的应对不精确。真实的胜负与 TD 回报的最后一段会纠正价值，但策略目标仍来自这种近似的搜索。
- 过拟合到这两个机器人：用对 R003 的直接对局监控。等强度接近达标线时，再用网页对局请真人复核。
- lookahead 每步要逐个试走全部合法动作，是 Python 实现，自对弈吞吐可能下降。

## 执行记录

- 2026-11-06 21:26 R003 停止（见 R003/analysis.md）。部署前，R004 补丁的基线 md5 与服务器一致，原文件备份在 backup_pre_r004_20261106/。
  先在 /tmp 的副本里跑测试：test_training / test_rl_pipeline / test_bots 共 37 个全部通过，其中 HeuristicLeafTest 4 个、TDLambdaTest 4 个。
- 起点 R003 latest（第 104 个 iteration）的独立基准（`benchmark_start.json`，400 局）：
  - vs 3×lookahead **0.295**（0.25–0.34）；
  - vs 3×heuristic 0.405（0.355–0.455）；
  - 对 v4 latest 直接对局 **0.615**（0.52–0.71）。
  - 按先验 8，R003 latest 成为最好模型。同一轮第 100 个 iteration 是 0.2575，相邻检查点之间约有 ±0.04 的波动。
  - 判据据此改为：vs lookahead ≥ 0.35（supported），< 0.31（refuted）。
- 2026-11-06 21:32 启动（pid 377405），从 R003 latest 初始化。第 0 个 iteration：
  - 1024 局 190 s（5.38 局/秒，与 R003 相当）；
  - 581 局有机器人座位，偏多于 50%，因为有机器人的对局结束得快，先完成的那一批里占比高；
  - 机器人对局里模型座位的平均份额 0.266（持平 0.25）；
  - held-out：价值 MSE 0.1469、策略 CE 0.759 / 目标熵 0.734。
- 2026-11-07 00:16 WATCH（2.73 h，59 个 iteration，3908 训练步，6.36 局/秒，lr 2.2e-5）：
  - 训练中基准（96 局）：
    - vs heuristic 0.427 → 0.51，最近 3 次 0.486；
    - vs lookahead 在 0.229–0.375 之间，没有趋势，最近 3 次 0.285；
  - 晋升 3 次（第 4、9、44 个 iteration）；
  - 每个 iteration 约 50% 的对局有机器人，模型座位份额稳定在 0.25–0.27；
  - held-out 价值 MSE 0.1448 → 0.1460，策略 CE 0.754 → 0.739。
  - **独立基准**（400 局，同一批发牌）：

| 检查点 | vs 3×lookahead | vs 3×heuristic | 对 R003 latest |
|---|---|---|---|
| 起点 R003 latest | 0.295 | 0.405 | — |
| R004 第 50 个 iteration | 0.2875（0.24–0.33） | **0.4625**（0.41–0.51） | 0.49（0.40–0.58） |
| R004 latest（约第 59 个） | 0.2825（0.23–0.33） | 0.455（0.40–0.51） | — |

  - 判断：
    - 对 heuristic 提高了约 0.05，对 lookahead 持平，对 R003 不丢强度。也就是说，对手多样化让模型学会了对付较弱的贪心对手，没有学会对付 lookahead。
    - 第 3、4 轮的共同点：网络学不到 lookahead 用的那类特征，比如"这张牌还差几个宝石""离贵族还差几张"，要从原始字段里算出来，且价值头看不到合法动作掩码。
    - 下一步（R005）：给观测加上这些派生特征（观测版本 2）。R004 在 R005 实现期间继续跑。
- 2026-11-07 00:38 停止（3.1 h，66 个 iteration）。latest 的独立基准（`benchmark_end.json`）：
  - vs 3×lookahead 0.2525（0.21–0.30）；
  - vs 3×heuristic 0.4525；
  - 对 R003 latest 0.552。
  - 判定 refuted，转 R005（见 analysis.md）。
