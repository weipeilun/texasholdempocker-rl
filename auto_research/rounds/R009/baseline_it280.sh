#!/bin/bash
# R009 起点（R008 it280）的基线：heuristic / alt1 / alt2 各 400 局（部署搜索 512 次，seed 5e9），再做风格统计（先验贪心，对 3×heuristic）
cd /data/playground/texasholdempocker-rl
PY=/home/weipeilun/.conda/envs/py310/bin/python
$PY benchmark_splendor.py --config config/splendor_auto_r008.yml --model /tmp/r008/r008_it280.pth \
    --opponents bot:heuristic bot:lookahead_alt1 bot:lookahead_alt2 --games 400 --sims 512 --seed 5000000000 --num-game-process 10 \
    --out auto_research/rounds/R009/baseline_it280_others.json > auto_research/rounds/R009/baseline_it280_others.log 2>&1
$PY auto_research/style_stats.py config/splendor_auto_r008.yml /tmp/r008/r008_it280.pth 50 auto_research/rounds/R009/baseline_it280_style.json \
    > auto_research/rounds/R009/baseline_it280_style.log 2>&1
