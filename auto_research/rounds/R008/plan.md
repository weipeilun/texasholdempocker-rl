# R008 计划：继续训练（同 R007 设置）

## 背景

- 目标已于 2026-11-07 17:10 达到：R007 第 80 个 iteration，部署搜索 512 次，在没用过的发牌上 vs 3×lookahead 0.5637（800 局）。模型在 `models/splendor_final/`。
- R007 之后跑满了 200 个 iteration（12.9 h，2026-11-07 22:45 结束）。
  - 收尾时会话连不到服务器，GPU 服务器随后重启，/tmp 下的副本丢失；实验目录完好。
  - 训练中基准（128 次、considered 8、prior，96 局）在第 80 个 iteration 之后继续上升：
    - vs lookahead：it54–84 约 0.42 → it134–199 0.51–0.625，it154–199 均值 0.57；
    - vs heuristic 约 0.80；
    - held-out value_ce 0.981（it87）→ 0.928（it200），winner_acc 0.587。
  - 训练没到平台。

## 改动（对照 R007）

| 内容 | R007 | R008 |
|---|---|---|
| 起点 | R006 latest | **R007 latest（第 200 个 iteration）** |
| 学习率 | 5e-5，6000 步 ×0.5，12000 步 ×0.3（末段 1.5e-5） | **2e-5**，10000 步 ×0.75，20000 步 ×0.5 |
| 上限 | 200 个 iteration，预算 9 h | 300 个 iteration，预算 18 h |
| 种子 | 20261113 | 20261114 |

其余完全相同：对手模型 prior、considered 8、带教 w 0.25、观测版本 2、50% 对局有 1–2 个机器人座位、TD(λ) 0.95。

## 判据（部署搜索：512 次、considered 8、prior；发牌 seed 5e9）

- supported：最后的检查点对 R007 latest ≥ 0.55，且 vs 3×lookahead 比 R007 latest 高 ≥ 0.03。
- refuted：对 R007 latest < 0.50。
- 同时跑 R007 latest 的确认基准，同样的发牌：
  - vs lookahead 800 局；
  - heuristic、alt1、alt2 各 400 局；
  - 对 it80（`models/splendor_final`）96 局。
  - 若明显强于 it80，更新 `models/splendor_final/`，旧版移到 `it80/` 子目录。

## 运行记录

- 2026-11-08 18:25 启动（pid 9753），严格加载 R007 latest（98 个张量）。
- 2026-11-08 19:18：13 个 iteration，6–8 局/s（前 10 个 iteration 与 R007 latest 的确认基准同时跑）。
  - 老师一致率 0.46–0.47。held-out value_ce 0.936 → 0.926。
  - 内置基准：it4 vs heuristic 0.865、vs lookahead 0.594；it9 为 0.771、0.562。
  - R007 latest 的确认基准：vs lookahead 0.6975、heuristic 0.8925、alt1 / alt2 0.7125 / 0.7325、对 it80 0.7188。这是 R008 的基线。
- 2026-11-08 22:31 第 1 次检查（4.10 h，92 个 iteration，6224 步，lr 1.69e-5，7.8 局/s）：
  - 老师一致率 0.455–0.467，持平。
  - held-out value_ce 0.936 → **0.876**，winner_acc 0.581 → 0.615，各轮最好。
  - 内置基准（128 次、considered 8、prior，96 局）：
    - vs lookahead it54–89 均值 **0.72**（R007 it154–199 为 0.57）；
    - vs heuristic 0.81–0.875。
  - 正在跑 it90 的部署搜索确认基准（512 次，seed 5e9）：vs lookahead 800 局 + 对 R007 it200 96 局；之后是 heuristic / alt1 / alt2 各 400 局。
- **it90 确认基准**（部署搜索 512 次，seed 5e9；`benchmark_watch1_it90*.json`）：

| 对手（3 个） | 局数 | R008 it90 | R007 it200（当前最终模型） |
|---|---|---|---|
| lookahead | 800 | **0.7562**（0.724–0.788） | 0.6975（0.664–0.731） |
| heuristic | 400 | **0.9275**（0.902–0.953） | 0.8925 |
| lookahead_alt1 | 400 | **0.7675**（0.727–0.808） | 0.7125 |
| lookahead_alt2 | 400 | **0.79**（0.753–0.827） | 0.7325 |
| 对 R007 it200 | 96 | 0.531（0.44–0.63） | — |

  - 判据：对 R007 ≥ 0.55 且 vs lookahead ≥ 0.7275。vs lookahead 满足；对 R007 只有 0.531，96 局、区间 ±0.09。
  - 先不更新最终模型。另跑 192 局直接对局（新发牌，seed 6e9），合计 288 局再判断（`benchmark_watch1_it90_vsR007_more.json`）。
- 2026-11-08 23:33 加跑的 192 局直接对局（另一批发牌）：对 R007 it200 **0.6198**（0.552–0.687）。合计 288 局 **0.5903**（约 0.533–0.647），≥ 0.55，判为强于 R007 it200。
- 2026-11-08 23:54 用户要重启机器，停在第 115 个 iteration（7780 步）。
- 2026-11-09 01:30 机器重启后续训（`--resume experiments/auto_r008`，pid 5357）：从第 115 个 iteration、第 7780 步接着跑，学习率 1.611e-05，回放池 150000 个样本、下一局 id 118351。第一次启动的记录在 `launch_part1.json`。
- 2026-11-09 01:35 更新最终模型：`models/splendor_final/` 换成 R008 it90（md5 3c2599de…），R007 it200 移到 `models/splendor_final/r007_it200/`（MD5SUMS 校验通过）。
- 2026-11-09 04:36 第 2 次检查（续训后 3.10 h，220 个 iteration，14902 步，lr 1.25e-5，10.0 局/s）：
  - held-out value_ce 0.936 → **0.846**（it90 时 0.876），winner_acc 0.581 → 0.628，policy_ce 1.297 → 1.246，都还在改善。
  - 内置基准（128 次、considered 8、prior，96 局）：
    - vs lookahead it119–214 均值约 0.77（it54–109 约 0.72），最高 0.833（it154、it174）；
    - vs heuristic 0.83–0.958，it184 之后都 ≥ 0.906。
  - 对 best 的评估：it194、it214 晋升。
  - 选 it190 跑部署搜索确认基准（512 次，seed 5e9）：vs lookahead 800 局 + 对当前最终模型（R008 it90）192 局（`benchmark_watch2_it190.json`）。it190 两侧的内置基准 0.823 / 0.781；96 局的内置基准区间约 ±0.09，挑选主要是噪声，确认基准用的是没用过的发牌。
  - 替换判据：对 R008 it90 ≥ 0.55，且 vs lookahead ≥ 0.7562 + 0.03 = 0.786。
- **it190 确认基准**（部署搜索 512 次，seed 5e9；`benchmark_watch2_it190.json`）：
  - vs 3×lookahead **0.84**（0.815–0.865，800 局）；it90 为 0.7562。
  - 对当前最终模型 R008 it90（两座对两座，192 局）：**0.6042**（0.533–0.675）。
  - 两条判据都满足（≥ 0.786、≥ 0.55）。05:01 起跑 heuristic / alt1 / alt2 各 400 局（`benchmark_watch2_it190_others.json`），之后决定是否替换最终模型。
- it190 其他对手（各 400 局，`benchmark_watch2_it190_others.json`）：heuristic **0.96**（0.942–0.978）、alt1 **0.84**（0.807–0.873）、alt2 **0.7925**（0.753–0.832）；it90 为 0.9275 / 0.7675 / 0.79，都不低于 it90。
- 2026-11-09 05:30 更新最终模型：`models/splendor_final/` 换成 R008 it190（md5 c8a06fb9…），R008 it90 移到 `models/splendor_final/r008_it90/`（各子目录 MD5SUMS 校验通过）。
- 2026-11-09 07:13 训练结束（300 个 iteration）。held-out value_ce 0.823，winner_acc 0.635；it249 之后不再晋升。
- 收尾基准（部署搜索 512 次，seed 5e9，20 个对局进程；`benchmark_final_*.json`）：
  - best（it249）：vs lookahead 0.8263（0.802–0.851），对 it190 0.5521（0.493–0.611）；
  - it280：vs lookahead 0.8563（0.831–0.882），对 it190 0.5573（0.482–0.633）。
  - 都未达替换线（vs lookahead ≥ 0.87），最终模型保持 it190。R009 从 it280 起步。
- 判定 supported，见 analysis.md / decision.json。
