# R005 计划：观测版本 2——派生特征

## 这一轮要回答的问题

R001–R004 都在 0.26–0.30（对 3×lookahead）附近停住：
- 学习率衰减、关缩放、TD(λ) 价值目标、对手多样化都只带来一两个百分点的变化；
- 评估时加深搜索、加启发式叶节点评估也没用。

强度几乎等于策略网络本身。lookahead 机器人只做一步试走，靠的是一个手写的局面评估：
- 每张牌"还差几颗宝石""几回合能买到"；
- 每个贵族"还差几张"；
- 对手能否下回合获胜。

这些量都要先算"成本 − 折扣 − 宝石 − 金币"的逐色截断和。网络只从原始字段（成本、折扣、宝石数）学，价值头也看不到合法动作掩码。
R004 让模型对 heuristic 提高了约 0.05，对 lookahead 仍不动，说明多见对手不能代替这些特征。

本轮回答：把这些量直接放进观测，网络能否越过平台。

## 改动（对照 R004）

| 内容 | R004 | R005 | 对应指标 |
|---|---|---|---|
| 观测 | 版本 1，290 个字段 | **版本 2，348 个字段**：每张牌（市场 12 张 + 各玩家预留 12 张）加 2 个字段，每个贵族加 2 个字段，见下 | 独立基准 vs lookahead / heuristic |
| 起点 | R003 latest | **R004 latest 转成版本 2**（网络函数不变，见下） | 转换前后输出一致 |
| 学习率 | 3e-5，6000 步 0.6，12000 步 0.4 | **5e-5**，6000 步 0.5，12000 步 0.3（新字段的嵌入从 0 学起） | lr |

新增字段（`splendor/rl_env.DERIVED_FIELD_GROUPS`，0–8，9 表示 9 或更多，10 表示空槽位）：
- `card_short_owner`：买方还差几颗宝石（成本减去折扣、彩色宝石后的逐色缺口之和，再减金币）。
  - 市场牌的买方是当前行动者；
  - 玩家 k 的预留牌，买方是玩家 k；
  - 0 等价于"现在就买得起"。
- `card_short_rival`：对手中最小的缺口（只对市场牌有意义），即"这张牌离哪个对手最近"。
- `noble_missing_owner` / `noble_missing_rival`：我 / 最接近的对手离这个贵族还差几张折扣牌。

这些字段与牌的其他字段一起组成同一个牌 token，槽位置换时随牌一起移动（对称性不变）。

转换（`auto_research/convert_obs_v2.py`，`rl/splendor_model.remap_observation_embedding`）：
- 新字段的嵌入行置 0；
- 牌与贵族各组的行乘 sqrt(新字段数 / 旧字段数)，即牌 sqrt(10/8)、贵族 sqrt(7/5)，抵消 token 归一化的变化；
- 其他参数原样复制。

所以转换后的网络与 R004 latest 计算的是同一个函数。在 728 个随机局面上，旧代码 + 旧权重与新代码 + 转换权重的输出差：
- 策略最大 9e-7；
- 价值最大 1e-7。

训练从这个函数出发，新字段的嵌入从 0 开始按梯度学，不会一开始就扰乱已有的策略。
锚点（R004、R003）也用同样的转换在新代码里对局。

观测编码变慢：每次约 77 µs。自对弈中每个叶节点编码一次，预计吞吐下降不到 10%。

测试：
- `ObservationV2Test`（4 个）：
  - 派生字段与定义一致，缺口为 0 ⇔ 买牌动作合法；
  - 派生字段随槽位置换；
  - 版本 1 checkpoint 转换后每个 token 不变；
  - 非严格初始化与转换文件一致。
- 全部 79 个测试通过（test_training / test_rl_pipeline / test_bots / test_engine / test_server）。

## 判定

- **supported**：独立基准（seed 4e9，400 局）中最后的检查点满足以下全部：
  - vs 3×lookahead ≥ 0.35（R004 末 0.2525，R003 latest 0.295）；
  - vs 3×heuristic ≥ 0.50；
  - 对 R004 latest（转换后，函数相同）≥ 0.55。
- **refuted**：vs lookahead 不高于 R004 末 + 0.02，且对 R004 latest < 0.52。
- **inconclusive**：其余情况。

## 风险

- 新字段有冗余（缺口可由原始字段算出），网络可能不用它们。看对 lookahead 的强度，必要时看新字段嵌入的范数增长。
- 版本 2 的数据与版本 1 不兼容：回放池从空开始。旧数据文件不能 warm start（版本号校验会拒绝）。
- 观测编码变慢，吞吐略降。

## 执行记录

- 2026-11-07 00:38 R004 停止。R005 的代码先在 /tmp 的副本里测试：79 个测试全部通过；在 728 个局面上验证了转换前后网络函数不变。
  之后部署到主仓库，原文件备份在 backup_pre_r005_20261107/。
- 转换出 experiments/auto_r004/checkpoints/latest_obs_v2.pth（R005 起点）和 experiments/auto_r003/checkpoints/latest_obs_v2.pth（锚点）。
- 起点 = R004 latest，函数相同。它的独立基准（R004/`benchmark_end.json`）：vs 3×lookahead 0.2525、vs 3×heuristic 0.4525，对 R003 latest 0.552。

## 运行记录
- 2026-11-07 00:43 启动（pid 396332）。iteration 0 严格加载 `experiments/auto_r004/checkpoints/latest_obs_v2.pth`（98 个张量全部加载），对局速度与 R004 相当（开局约 95 s 完成约 270 局）。
- 2026-11-07 03:15 第 1 次检查（2.54 h，51 个 iteration，3392 步，lr 3.6e-5，5.9 局/s）：
  - held-out value_ce 1.052 → 1.038，value_mse 0.144 → 0.142（常数基线 0.1875），policy_ce 0.749 → 0.746。
  - 4 人评估对起点 0.46–0.53，没有晋升。
  - 内置基准（96 局，seed 3e9）前 10 次的均值：
    - vs heuristic 0.528（R004 0.474，R003 0.431）；
    - vs lookahead 0.326（R004 0.299，R003 0.280）。
  - 独立基准（seed 4e9，400 局）：
    - it30：vs lookahead 0.295，vs heuristic 0.4625；
    - it50：vs lookahead 0.2825（0.24–0.33），vs heuristic 0.47（0.42–0.52），对 R004 0.552（96 局，0.45–0.65）。
  - 判定未定：比 R004 末（0.2525 / 0.4525）略好，未到 supported 线；也不满足 refuted 条件。
  - 阶梯校准（/tmp/r005/ladder，同样的 deal seed，400 局）：
    - lookahead vs 3×heuristic 0.7975；
    - heuristic vs 3×lookahead 0.115。
  - 发现：模型对 lookahead 略强于 lookahead 自身（0.28–0.30 > 0.25），但对 heuristic 远不如 lookahead（0.47 vs 0.80）。模型不会利用弱对手；对 lookahead 的强度至今各轮都卡在 0.26–0.30。
  - 下一步诊断（it50 固定权重，/tmp/r005/diag）：
    - 评估时 considered 8 / 4 + 256 次模拟（让搜索能看到对手下一手的威胁）；
    - 启发式 rollout λ=0.5（多回合评估）。
