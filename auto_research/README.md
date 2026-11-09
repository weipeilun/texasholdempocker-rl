# Splendor 自动实验循环（auto-research）

仿照 auto-research 在 Nested-Learning-Pytorch 上的做法：给定终极目标，agent 每一轮自己决定验证什么、改什么，
启动训练，按间隔检查，跑完后分析数据、下判定、定下一轮，并把每一轮写进笔记页
<https://notebook.infimory.ai/splendor-auto.html>。账本就是本目录下的普通文件，页面由它渲染，任何一个新会话
读这些文件就能接着跑。

## 文件

| 文件 | 内容 | 谁写 |
|---|---|---|
| `goal.md` | 终极目标原文与背景事实 | 用户（agent 只补背景） |
| `benchmark.md` | "一般水平玩家"的代理基准与达标线 | agent，改动需写明理由 |
| `priors.md` | 设计先验 / 约束：所有轮次都要遵守 | agent，偏离须在 plan.md 写明 |
| `program.md` | 研究纲领：路线、当前结论、已证伪、待验证、里程碑 | agent 每轮维护 |
| `state.json` | status / phase / current_round / goal_status / best_model | `loop.py` 与 agent |
| `rounds/RNNN/spec.yaml` | 名称、类型、假设、判定问题、配置路径、预算、达标线 | agent（PLAN） |
| `rounds/RNNN/plan.md` | 为什么做、改了什么、怎么判定、预期 | agent（PLAN） |
| `rounds/RNNN/launch.json` | 启动时间、pid / pgid、命令、实验目录、代码版本 | `loop.py launch` |
| `rounds/RNNN/watch.json` | 最近一次检查的汇总（来自实验的 metrics.jsonl） | `loop.py watch` |
| `rounds/RNNN/benchmark_*.json` | 独立基准（`benchmark_splendor.py`）的结果 | agent |
| `rounds/RNNN/analysis.md` | 数据、分析、结论 | agent（ANALYZE） |
| `rounds/RNNN/decision.json` | verdict、goal_status、reason、next | agent（DECIDE） |

每个实验写在自己的目录 `experiments/auto_rNNN/`（config.yml、train.log、metrics.jsonl、train_data.txt、
checkpoints/），配置同时存为 `config/splendor_auto_rNNN.yml`。

## 一轮的阶段

1. **PLAN**：读 program.md、上一轮的 analysis.md / decision.json，写 spec.yaml、plan.md、配置文件。一轮只验证一组改动；
   用户指定的组合改动照做，但在 plan.md 里写明哪些指标对应哪项改动。
2. **BASELINE**（需要时）：训练开始前用 `benchmark_splendor.py` 测起点模型，存 `benchmark_start.json`。
3. **LAUNCH**：`python auto_research/loop.py launch RNNN`（同一时间只允许一个训练）。
4. **WATCH**：每 1.5–3 小时 `python auto_research/loop.py watch RNNN --enforce-budget`。检查：进程是否在跑、有无报错、
   吞吐、学习率是否按计划、held-out 价值 / 策略指标、评估晋升、基准胜率份额。出错就修复后 `--resume` 续训（记入 plan.md 的执行记录）；
   明显失败（基准连续 3 次低于起点 0.05 以上、或 held-out value CE 持续上升）可以提前停。
5. **ANALYZE**：训练结束（num_iterations 或预算）后，用独立基准测本轮最好的检查点（400 局 vs lookahead / heuristic，
   加起点与上一轮最好模型作锚点），写 analysis.md：数据、和假设对照、原因分析。
6. **DECIDE**：decision.json：`verdict`（supported / refuted / inconclusive）、`goal_status`（open / reached）、
   `reason`、`next`（下一轮的名称、假设、要点）。更新 program.md 与 state.json 的 best_model。
7. **PUBLISH**：`python auto_research/loop.py render --out /tmp/splendor-auto.html`，备份后覆盖笔记站的页面。

达标（goal_status = reached）：独立基准中对 3 个 lookahead 的胜率份额 ≥ 0.50 且 95% 区间下界 > 0.40（见 benchmark.md），
然后把模型与配置固定到 `models/splendor_auto_final.pth` 与 `config/splendor_auto_final.yml`。
