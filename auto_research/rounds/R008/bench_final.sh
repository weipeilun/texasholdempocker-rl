#!/bin/bash
# R008 收尾：两个候选的部署搜索确认基准（顺序运行）
cd /data/playground/texasholdempocker-rl
PY=/home/weipeilun/.conda/envs/py310/bin/python
for tag in best_it249 it280; do
  $PY benchmark_splendor.py --config config/splendor_auto_r008.yml --model /tmp/r008/r008_${tag}.pth \
      --opponents bot:lookahead --games 800 --anchor final=models/splendor_final/weights.pth --anchor-games 192 \
      --sims 512 --seed 5000000000 --num-game-process 20 \
      --out auto_research/rounds/R008/benchmark_final_${tag}.json > auto_research/rounds/R008/benchmark_final_${tag}.log 2>&1
done
