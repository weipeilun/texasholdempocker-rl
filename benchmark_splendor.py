"""Strength benchmark of a trained Splendor agent: rule bots and fixed anchor models.

    python benchmark_splendor.py --config experiments/<name>/config.yml --model experiments/<name>/checkpoints/best.pth
           [--opponents bot:lookahead bot:heuristic] [--games 400] [--anchor NAME=PATH ...] [--anchor-games 192]
           [--sims 64] [--num-game-process 22] [--out result.json]

Bots: the model in one seat against three copies of the bot, every deal played four times with the model in
each seat (equal strength = win share 0.25). Anchors: two seats each, the evaluation arrangement of
train_splendor.eval_tasks (equal strength = 0.5). Both report a 95% interval from the spread over deals (the
games of one deal are averaged first). The search settings are the config's evaluation settings (greedy,
eval_num_mcts_simulation_per_step simulations; --sims overrides). Uses the GPU and all game workers: run it
when no training is running, or pass a smaller --num-game-process.

Ladder (splendor/bots.py): random < heuristic < lookahead. The "average player" proxy of the auto-research
loop is lookahead (it beats three heuristic players with a win share of about 0.75-0.8 and ends games among
itself in about 25 rounds); the target is a win share of at least 0.50 against three of them.
"""
import argparse
import json
import logging
import math
import os
import time

import numpy as np
import torch.multiprocessing as mp
import yaml

import train_splendor as ts
from rl.splendor_model import load_checkpoint, model_state_of


def _interval(per_deal):
    """mean and 95% normal interval of per-deal scores"""
    x = np.asarray(per_deal, dtype=np.float64)
    if len(x) < 2:
        return float(x.mean()) if len(x) else None, None
    half = 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean()), [float(x.mean() - half), float(x.mean() + half)]


def bot_summary(results, slot):
    summary = ts.bench_score(results, slot)
    for opponent in summary:
        per_deal = dict()
        for r in results:
            if opponent not in r['seat_server_ids']:
                continue
            if r.get('failed') or r['end_reason'] not in ('score', 'draw'):
                share = 0.25
            else:
                share = r['rewards'][list(r['seat_server_ids']).index(slot)]
            per_deal.setdefault(r['seed'], list()).append(share)
        mean, ci = _interval([np.mean(v) for v in per_deal.values()])
        summary[opponent].update(deals=len(per_deal), win_share_ci95=ci)
    return summary


def anchor_summary(results, num_games):
    score = ts.eval_score(results, num_games)
    per_deal = dict()
    for r in results:
        if r.get('failed') or r['end_reason'] != 'score':
            share = 0.5
        else:
            share = sum(w for w, s in zip(r['rewards'], r['seat_server_ids']) if s == ts.SLOT_NEW)
        per_deal.setdefault(r['seed'], list()).append(share)
    mean, ci = _interval([np.mean(v) for v in per_deal.values()])
    return dict(score, deals=len(per_deal), win_rate_ci95=ci)


def run(pool, tasks, label):
    pool.submit(tasks)
    results, begin, last_log = list(), time.time(), time.time()
    while len(results) < len(tasks):
        result = pool.get_result(timeout=1.)
        if result is not None:
            results.append(result)
        if time.time() - last_log > 60:
            logging.info(f'{label}: {len(results)}/{len(tasks)} games, {time.time() - begin:.0f}s')
            last_log = time.time()
    logging.info(f'{label}: {len(tasks)} games in {time.time() - begin:.0f}s')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True, help="an experiment's config.yml (search settings, model shape)")
    parser.add_argument('--model', required=True, help='checkpoint of the agent (any format with the weights)')
    parser.add_argument('--opponents', nargs='*', default=['bot:lookahead', 'bot:heuristic'])
    parser.add_argument('--games', type=int, default=400, help='games per bot opponent (multiple of 4)')
    parser.add_argument('--anchor', action='append', default=[], metavar='NAME=PATH', help='fixed model opponent')
    parser.add_argument('--anchor-games', type=int, default=192, help='games per anchor (multiple of 6)')
    parser.add_argument('--sims', type=int, default=None, help='override eval_num_mcts_simulation_per_step')
    parser.add_argument('--seed', type=int, default=4000000000, help='deals (different from the training benchmark)')
    parser.add_argument('--num-game-process', type=int, default=None)
    parser.add_argument('--out', default=None, help='write the summary as JSON')
    args = parser.parse_args()

    ts.init_logging(logging.INFO)
    params = yaml.safe_load(open(args.config, encoding='UTF-8'))
    params['experiment_dir'] = os.path.dirname(os.path.abspath(args.config))
    params['_log_file'] = None
    if args.num_game_process is not None:
        params['num_game_process'] = args.num_game_process
    if args.sims is not None:
        params['eval_num_mcts_simulation_per_step'] = args.sims
    params['model_param_dict']['device'] = ts.resolve_device(params['model_param_dict'].get('device'))
    agent = model_state_of(load_checkpoint(args.model))
    anchors = [(name, model_state_of(load_checkpoint(path))) for name, path in (a.split('=', 1) for a in args.anchor)]
    pool = ts.make_game_pool(params, [agent, anchors[0][1] if anchors else agent, agent], logging.INFO)
    summary = {'model': os.path.abspath(args.model), 'config': os.path.abspath(args.config),
               'sims': params['eval_num_mcts_simulation_per_step'], 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
    try:
        if args.opponents:
            bench_params = dict(params, bench_opponents=args.opponents, bench_num_games=args.games, bench_seed=args.seed)
            results = run(pool, ts.bench_tasks(bench_params, 0, slot=ts.SLOT_NEW), 'bots')
            summary['bots'] = bot_summary(results, ts.SLOT_NEW)
        summary['anchors'] = dict()
        for k, (name, state_dict) in enumerate(anchors):
            pool.set_weights(ts.SLOT_BEST, state_dict, version=k + 1)
            eval_params = dict(params, eval_task_num_games=args.anchor_games, eval_seed=args.seed + 1)
            tasks = ts.eval_tasks(eval_params, k + 1)
            summary['anchors'][name] = anchor_summary(run(pool, tasks, f'anchor {name}'), len(tasks))
    finally:
        pool.close()
    text = json.dumps(summary, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, 'w', encoding='UTF-8') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
