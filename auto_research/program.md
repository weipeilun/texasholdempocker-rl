### 路线

- **M0（循环之前，已完成）**：管线修正——规则 v2（被迫跳过、四人连续跳过判平）、截断局不监督价值、按编号派生种子、逐层初始化、
  数据 / checkpoint 版本号、完整续训、常驻对局进程池（v4）。
- **M1：R001 学习率衰减 + 加强自对弈搜索**（用户指定）。问题：v4 后期停滞是恒定学习率的噪声地板，还是搜索太弱、策略目标质量不够？
- **M2：按 R001 的结论选方向**（候选见"待验证"），直到对 lookahead 的基准过 0.25（与代理持平）。
- **M3：对 lookahead ≥ 0.50**（达标线），独立基准确认，固定模型与配置。

### 当前实验组

R009（带教权重 0.25 → 0.1，其余同 R008，从 R008 it280 起步；2026-11-09 08:10 起）。**目标已于 2026-11-07 17:10 达到**，模型在 models/splendor_final/（R008 it190）。已结束：R001（inconclusive）、R002（refuted）、R003（inconclusive）、R004（refuted）、R005（inconclusive）、R006（inconclusive）、R007（supported）、R008（supported）。

### 当前结论

- 所有模型都比"一般玩家"代理弱：独立基准对 3×lookahead v4 0.24、R001 0.145、R002 0.205（对 heuristic 0.295 / 0.41 / 0.415）。
- **搜索几乎没有收益**：评估模拟次数 1 → 64 → 256，v4 为 0.16 / 0.205 / 0.19，R002 为 0.21 / 0.21 / 0.24。强度基本等于策略本身。
- **价值是瓶颈，具体是局部排序**：全局价值 R² 约 0.12–0.25，跨模型、跨阶段几乎一样（开局约 0.06，13 分以上约 0.5），1000 个 iteration 不动。
  但价值一步贪心在同一轮相邻检查点之间 0.055–0.20 跳动，说明兄弟局面之间的差异没有被学到，只是随权重的微小变化重新洗牌。
- Q 的最小最大缩放把几乎相同的 Q 放大成近似 one-hot 的目标（目标熵 0.5）。R002 关掉后目标熵约 1.0–1.2，策略变强：
  对 R001 直接对局 0.604，策略一步贪心 0.145 → 0.18–0.26。
- 学习率衰减按计划生效（R001）。它对价值排序的作用无法与检查点噪声区分。
- R003（TD(λ) 价值目标）：独立基准从 0.21（R002 latest）提到 0.26–0.295（第 50 / 70 / 100 / 104 个 iteration：0.26 / 0.275 / 0.2575 / 0.295），
  第 50 个 iteration 之后持平。价值一步贪心反而更低（均值 0.116），所以它不能预测搜索强度，只作诊断，判断以独立基准为准。
- 自对弈强度与对机器人强度分化：R003 对 R002 latest 0.615、对 v4 0.615，对机器人却持平。训练对手几乎都是"四个相似的我"，这是 R004 要检验的问题。
- R004（对手多样化）：对 heuristic +0.05（0.405 → 0.4525）、对 lookahead 不动（0.2525）、对 R003 0.552。多见对手不能代替 lookahead 评估所用的特征。
- R005（观测版本 2）：
  - 对 heuristic 0.4875、对 lookahead 0.275（第 30 个 iteration 0.295）、对 R004 0.5625；
  - 两套基准都是 +0.02–0.05，特征有一点用，但不是平台的原因。
- **平台的原因：策略先验在整类动作上塌缩**（R005 诊断）：
  - 模型几乎从不走"同色拿 2 个"：每局 0.005 次，lookahead 4.24 次；可走时先验质量 0.3%。
  - 模型很少预留 3 级牌：0.59 vs 1.99。它多买便宜的 1 级牌（9.4 vs 3.3）、凑贵族（0.74 vs 0.06），对手下回合能赢时多半不理会（0.545 vs 0.11）。
  - Gumbel 的改进策略 π' = softmax(logits + σ(q))，c_scale 0.1 且不缩放，兄弟动作之间 σ 只差约 0.5 个 logit，抬不起落后 5–6 的动作；快速搜索只考虑前 8 个动作。
- 阶梯校准（同样的发牌）：lookahead 对 3×heuristic 0.7975，heuristic 对 3×lookahead 0.115。模型对 lookahead 略强于 lookahead 自身，对 heuristic 远不如它（0.49 vs 0.80）。
- 加深评估搜索仍然没用（R005 第 50 个 iteration）：considered 8 / 4 + 256 次模拟、启发式 rollout λ 0.5，都 ≤ +0.04。
- R006（带教 w 0.25）：
  - 先验塌缩的动作类回来了：同色拿 2 个 0.005 → 2.2，预留 3 级牌 0.59 → 1.43；
  - 旧标准 vs lookahead 0.3175、vs heuristic 0.5575，都是各轮最高；
  - 预留牌很少被买下，这部分还没学会。
- **4 人局里搜索的价值取决于对手模型**（R006 诊断）：
  - lookahead2（多看一层自己的回合）在对手模型准确时 vs 3×lookahead 0.39、vs 3×heuristic 0.89；对手模型不准时 0.25；
  - max^n 把对手当成"像我一样"，在自对弈中准确，所以自对弈强度一直在涨，对机器人却没用；
  - 对手节点的分支吃掉了模拟次数，到不了自己的下一回合。
- **部署搜索**（256 次模拟、considered 8、对手走先验 argmax）：
  - R006 latest 达到 vs 3×lookahead 0.43、vs 3×heuristic 0.665（旧标准 0.3175 / 0.5575）；
  - prior 已接近准确对手模型的上限（it90：0.3875 vs 0.4025）。
- R007（对手模型 prior 用于自对弈与评估、considered 8）：
  - 部署搜索 512 次、确认发牌：it80 vs 3×lookahead 0.5637（达标），it200 0.6975 / heuristic 0.8925 / alt1 0.7125 / alt2 0.7325；
  - held-out value_ce 0.991 → 0.928；
  - 训练 200 个 iteration 仍在变强。
- R008（同设置续训 300 个 iteration）：
  - 部署搜索 vs 3×lookahead：R007 it200 0.6975 → it90 0.7562 → it190 0.84（heuristic 0.96，alt1 / alt2 0.84 / 0.7925），直接对局每 100 个 iteration 约 0.59–0.60；
  - it190 之后平台：it249 / it280 vs lookahead 0.83 / 0.86、对 it190 0.55 / 0.56（区间含 0.5），it249 之后训练中评估不再晋升；
  - held-out value_ce 0.936 → 0.823 仍在降，但不再转化为强度；
  - 老师一致率 0.468 → 0.450：模型远强于带教老师，带教 w 0.25 可能是平台的原因之一（R009）。
- 相邻检查点之间的独立基准约有 ±0.04 的波动（R003 第 100 / 104 个 iteration：0.2575 / 0.295），比较模型时要看多个检查点或多跑局数。
- 评估时的搜索设置没有现成的杠杆：value_scale 0.3 / 1.0、256 次模拟、叶节点混入 lookahead 的局面评估（μ 0.5 / 1），都在 0.25–0.29 之间（±0.04）。

### 已证伪

- R005 附带：评估时加深搜索（considered 8 / 4、256 次模拟）或混入启发式 rollout（λ 0.5）能越过平台（都 ≤ +0.04）。
- R004：对手多样化（50% 对局有 1–2 个机器人座位）不能让对 lookahead 的强度提高（0.295 → 0.2525），只提高了对 heuristic 的强度。
- R002：关缩放 + 价值权重 1.0 + 终局（≥ 11 分）全量搜索 + 15% 机器人座位，不能让价值排序或训练中基准提高（价值一步贪心均值 0.12，基准 +0.035）。
- R001 补充中的"学习率衰减把价值一步贪心从 0.05 提到 0.20"：是检查点噪声（R001 四个检查点 0.055–0.20）。

### 待验证

- R009：带教权重 0.25 → 0.1（其余同 R008，从 R008 it280 起步，200 个 iteration）。
  判据（部署搜索）：对 R008 it280 ≥ 0.55，vs 3×lookahead 与 alt1 / alt2 不低于起点 0.03 以上；同时复查先验塌缩的动作类。
  风险：先验也是搜索里的对手模型，带教减弱后它对 lookahead 的预测可能变差。
- 之后的候选：
  - 单独的对手模型头：预测实际对手的动作，与自己的策略头分开，带教只训练对手头；
  - 降低带教 w，同时把 c_scale 0.1 → 0.5，让搜索自己改进策略；
  - 学会"预留后再买下"（预留区买牌仍只有 0.23 次）；
  - 对没见过的对手复核（权重扰动过的 lookahead）。
- 更早的候选（R006 之前列的）：
  - 加大 c_scale / 恢复 min-max 缩放（只在全量搜索），快速搜索 considered 8 → 16；
  - 搜索正确建模对手：机器人座位在树里按机器人的走法展开，不再用网络先验；
  - 直接的排序监督：全量搜索中被访问较多的子节点，用它们的 Q 向量作为子局面的价值目标（一次搜索给出一组兄弟局面的价值）；
  - 价值头的辅助目标：各座位的最终分数、剩余回合；
  - 更大的网络（num_layers 8、d=384）；
  - value_scale 0.1 → 0.3（价值变好之后，让搜索更多地改变策略）。
- 其他候选：更大的回放池或更多样本 / 局；对历史版本的对手池（防止自对弈循环）。

### 设计变更记录

- 2026-11-09（评估方法，用户确认）：
  - 确认评估统一为历史版本池（`auto_research/benchmark_pool.py`、`pool.yaml`、`pool_results.jsonl`，见 benchmark.md 末节）：规则机器人作为池中成员，与历史 checkpoint 同一种两座对两座安排；
  - 判据：对当前最终模型 ≥ 0.55 且下界 > 0.50（576 局），对其他成员不比最终模型同发牌成绩低 0.05 以上；Bradley–Terry 等级分（lookahead = 0）作汇总；
  - 训练中的规则机器人小基准从 R010 起缩到 lookahead 48 局，只看趋势；
  - 2026-11-09 11:00 起跑池的初始化：当前最终模型 R008 it190 对其余 8 个成员（672 局）。
- 2026-11-09（线下对局）：`splendor/offline/`（摄像头 + 视觉模型 + 机器人座位的线下对局网页，记录每局结果，用来测对真人的强度）。

- 2026-11-07（R007）：
  - 搜索的对手模型 `opponent_policy`（rl/splendor_mcts.SplendorMCTS；配置 mcts_opponent_policy / eval_mcts_opponent_policy）：
    - max_n：旧行为；
    - prior：其他玩家走先验 argmax、不分支；
    - bot:<policy>：只作诊断。
  - 测试 OpponentPolicyTest。
  - 从本轮起，独立基准同时报旧标准和部署搜索。

- 2026-11-07（R006）：
  - 策略带教 `policy_teacher` / `policy_teacher_weight`（train_splendor.play_game_iter；metrics teacher_samples / teacher_agree）；
  - 诊断脚本 auto_research/style_stats.py（动作类型统计、威胁处理），rounds/R005/diag/prior_check.py（先验与 lookahead 的一致性）。

- 2026-11-07（R005）：
  - **观测版本 2**（splendor/rl_env.DERIVED_FIELD_GROUPS，348 个字段）；
  - rl/splendor_model.remap_observation_embedding / convert_checkpoint_to_current_observation 与 auto_research/convert_obs_v2.py：
    版本 1 → 2 的同函数转换，新字段嵌入为 0，牌 / 贵族行按 sqrt(字段数之比) 缩放；
  - 版本 1 的 checkpoint 与数据要转换后才能用于新代码，锚点也要转换（experiments/auto_r00{3,4}/checkpoints/latest_obs_v2.pth）。

- 2026-11-06（R004）：
  - self_play_bot_seat_counts（多机器人座位）；
  - 启发式叶节点评估 mcts_heuristic_leaf_weight / mcts_heuristic_temperature（已部署，未启用）；
  - eval_mcts_value_scale；
  - loop.py stop 先只向训练主进程发信号。

- 2026-11-06（R003）：
  - TD(λ) 价值目标：`value_target_lambda`、`train_splendor.lambda_returns`、`SplendorMCTS.root_search_value`；
  - held-out 价值指标改为对真实结果计算，另记对训练目标的 held_out_target；
  - 价值一步贪心改为 400 局、多检查点。
- 2026-11-06（R002）：
  - mcts_rescale_values 关闭（自对弈与评估）；
  - 终局全量搜索、自对弈中的机器人座位；
  - 数据文件尾部快速读取；
  - 价值一步贪心诊断（auto_research/value_greedy.py）。
- 2026-11-06：
  - 学习率改为训练步的纯函数（lr_schedule），续训跟随配置；
  - 新增规则机器人 lookahead、训练中基准（bench_*）、metrics.jsonl、独立基准 benchmark_splendor.py、本账本与页面。

### 里程碑

| 里程碑 | 判据 | 状态 |
|---|---|---|
| 与"一般玩家"代理持平 | 基准 vs 3×lookahead ≥ 0.25 | **达到**：R006 latest 旧标准 0.3175（0.27–0.36），部署搜索 0.43（0.38–0.48） |
| 超越"一般玩家"代理 | 独立基准 vs 3×lookahead ≥ 0.50，区间下界 > 0.40；vs 3×heuristic ≥ 0.70 | **达到（2026-11-07）**：R007 it80，部署搜索（512 次、considered 8、prior），没用过的发牌上 vs lookahead 0.5637（0.531–0.597，800 局）、vs heuristic 0.7825、vs alt1 / alt2 0.5975 / 0.62。模型：models/splendor_final/ |
