目标为 agent 在 splendor 项目里超越一般水平玩家能力，产出模型并保存配置。

——以上为终极目标原文（weipeilun，2026-11-06）。同一条指令里的要求：先做两项改动并同时生效开始实验——
"1. 尝试衰减学习率，修复问题，确保生效。2. 提高自对弈中全量搜索的比例和模拟次数"；之后由 agent 自动执行实验：
一轮之后自己判断时间或检视间隔检查训练情况，自己分析问题、提出解决方案、执行优化、开启下一轮；
每一轮的实验路径写到 notebook.infimory.ai 的一个新专项记录里。

背景与约束（事实，不改变目标）：

- 代码：仓库 texasholdempocker-rl（分支 splendor-training），GPU 服务器 infimory-gpu 上的 `/data/playground/texasholdempocker-rl`。
  Gumbel AlphaZero（rl/splendor_mcts.py + C 扩展），6 层 transformer（d=256）策略 / 价值网络，常驻对局进程池，
  playout cap randomization（全量搜索才记录为样本），新模型对 best 的评估。
- 规则：四人局，`splendor/engine.py` RULES_VERSION 2：有人到 15 分立即结束（不补完本圈），无合法动作被迫跳过，四人连续跳过判平局。
- 起点 v4（`experiments/splendor_v4_20261104`，2026-11-04 17:15 起，自 v3b 权重初始化）：约 16.7 局/秒；
  固定对手评估显示长期在进步（对 v3 第 39 轮 0.849、对 v3b 终版 0.599），但最近停滞（对自己第 50 轮 0.52、对 best 第 99 轮 0.539），
  学习率一直是恒定的 2e-4。
- 资源：一张 RTX PRO 6000（96 GB）、i7-13700KF（24 线程）；同一时间只跑一个训练。

## 达成记录（2026-11-07 17:10）

- 达到。R007 第 80 个 iteration，部署搜索：512 次模拟、considered 8、对手走先验 argmax。
- 在没用过的发牌（seed 5e9）上：
  - vs 3×lookahead 0.5637（0.531–0.597，800 局）；
  - vs 3×heuristic 0.7825（400 局）；
  - vs 没见过的 lookahead_alt1 / alt2 0.5975 / 0.62。
- 模型与配置：`models/splendor_final/`（weights.pth、config.yml、play_agent.py、README.md、benchmarks/、MD5SUMS）。
- 2026-11-08 更新：models/splendor_final 换成 R007 第 200 个 iteration。确认发牌、512 次模拟：
  - vs 3×lookahead 0.6975（0.664–0.731，800 局）；
  - vs heuristic 0.8925；
  - vs alt1 / alt2 0.7125 / 0.7325；
  - 对 it80 0.7188。
  - it80 保留在 models/splendor_final/it80/。
- 2026-11-09 更新：models/splendor_final 换成 R008 第 90 个 iteration。确认发牌、512 次模拟：
  - vs 3×lookahead 0.7562（0.724–0.788，800 局）；
  - vs heuristic 0.9275；
  - vs alt1 / alt2 0.7675 / 0.79；
  - 对 R007 it200 0.5903（288 局，两座对两座）。
  - R007 it200 移到 models/splendor_final/r007_it200/，it80 仍在 models/splendor_final/it80/。
- 2026-11-09 05:30 更新：models/splendor_final 换成 R008 第 190 个 iteration。确认发牌、512 次模拟：
  - vs 3×lookahead 0.84（0.815–0.865，800 局）；
  - vs heuristic 0.96；
  - vs alt1 / alt2 0.84 / 0.7925；
  - 对 R008 it90 0.6042（192 局，两座对两座）。
  - R008 it90 移到 models/splendor_final/r008_it90/。
