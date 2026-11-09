#!/bin/bash
# R009 第 1 次池评估：等池初始化（pid 52640）结束，取当时最新的 R009 checkpoint 做候选
cd /data/playground/texasholdempocker-rl
while kill -0 52640 2>/dev/null; do sleep 30; done
CK=$(ls experiments/auto_r009/checkpoints/weights_iteration_*.pth | sort | tail -1)
IT=$(basename $CK .pth | sed 's/weights_iteration_0*//')
echo "candidate r009_it$IT $CK $(date)" > auto_research/rounds/R009/pool_eval_1.info
/home/weipeilun/.conda/envs/py310/bin/python auto_research/benchmark_pool.py --candidate r009_it$IT=$CK \
    --config config/splendor_auto_r009.yml --h2h 576 --budget 1152 --min-games 48 --sims 512 --num-game-process 10 \
    --out auto_research/rounds/R009/pool_eval_1.json > auto_research/rounds/R009/pool_eval_1.log 2>&1
