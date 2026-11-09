"""History-pool evaluation: a candidate model against every member of the pool (auto_research/pool.yaml): frozen
past models and rule bots alike, in one arrangement (two seats each, the arrangement of train_splendor.eval_tasks,
equal strength = 0.5) with the deployment search.

    python auto_research/benchmark_pool.py --candidate r009_it120=experiments/auto_r009/checkpoints/weights_iteration_00120.pth \
        [--config config/splendor_auto_r009.yml] [--sims 512] [--h2h 576] [--budget 1152] [--min-games 48] [--num-game-process 10]
    python auto_research/benchmark_pool.py --ratings          # refit the ratings from the stored results only

Games: --h2h games against the pool's current final model (the decision), the rest of --budget spread over the other
members in proportion to p(1-p) of the expected score (most games where the result is least certain), at least
--min-games each (the regression check). Every member has its own fixed deals (by its position in pool.yaml), so
two candidates meet a member on the same deals (paired).

Results are appended to auto_research/pool_results.jsonl. Ratings: a Bradley-Terry (Elo-scale) fit of every stored
result, the pool's reference member (lookahead) at 0. Decision for replacing the final model: head-to-head >= 0.55
with the 95% lower bound > 0.50, and against no older member more than 0.05 below the final model's own result on
the same deals (regression / cycling check).
"""
import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.join(ROOT, 'auto_research')
POOL = os.path.join(HERE, 'pool.yaml')
RESULTS = os.path.join(HERE, 'pool_results.jsonl')
SEED_BASE = 7_000_000_000
ELO = 400. / math.log(10.)


def load_pool(path=POOL):
    pool = yaml.safe_load(open(path, encoding='UTF-8'))
    for i, m in enumerate(pool['members']):
        m['index'] = i
        m['seed'] = SEED_BASE + 1_000_000 * i
    return pool


def load_results(path=RESULTS):
    if not os.path.exists(path):
        return []
    with open(path, encoding='UTF-8') as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------------------------- ratings
def fit_ratings(results, reference='lookahead', prior=1e-4):
    """Bradley-Terry on the Elo scale from (a, b, games, score of a) records; reference fixed at 0. A tiny ridge
    (prior) keeps players with one-sided results finite. -> {name: (rating, standard error)}"""
    pairs = [(r['candidate'], r['member'], r['games'], r['score']) for r in results if r.get('games')]
    names = sorted({a for a, _, _, _ in pairs} | {b for _, b, _, _ in pairs})
    if reference not in names:
        names.append(reference)
    idx = {n: i for i, n in enumerate(names)}
    free = [i for n, i in idx.items() if n != reference]
    x = np.zeros(len(names))
    for _ in range(100):
        grad, hess = np.zeros(len(names)), np.zeros((len(names), len(names)))
        for a, b, n, s in pairs:
            i, j = idx[a], idx[b]
            p = 1. / (1. + math.exp(-(x[i] - x[j])))
            g = n * (s - p)
            w = n * p * (1. - p)
            grad[i] += g
            grad[j] -= g
            hess[i, i] -= w
            hess[j, j] -= w
            hess[i, j] += w
            hess[j, i] += w
        grad -= prior * x
        hess -= prior * np.eye(len(names))
        h = hess[np.ix_(free, free)]
        step = np.linalg.solve(h, grad[free])
        x[free] -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.inv(-hess[np.ix_(free, free)])
    se = np.zeros(len(names))
    se[free] = np.sqrt(np.diag(cov))
    return {n: (float(x[i] * ELO), float(se[i] * ELO)) for n, i in idx.items()}


def expected(r_a, r_b):
    return 1. / (1. + 10 ** (-(r_a - r_b) / 400.))


# ---------------------------------------------------------------------------------------------- allocation
def allocate(pool, candidate, ratings, h2h, budget, min_games):
    """games per member: h2h against the final model, the rest by p(1-p), multiples of 6 (seat arrangements)"""
    six = lambda n: max(6, int(round(n / 6.)) * 6)       # noqa: E731
    final = pool['final']
    base = ratings.get(final, (0., 0.))[0]               # prior: the candidate is about as strong as the final model
    games, weights = {}, {}
    for m in pool['members']:
        name = m['name']
        if name == candidate:
            continue
        if name == final:
            games[name] = six(h2h)
            continue
        p = expected(base, ratings[name][0]) if name in ratings else .5
        weights[name] = p * (1. - p)
    rest = max(0, budget - sum(games.values()))
    total = sum(weights.values()) or 1.
    for name, w in weights.items():
        games[name] = six(max(min_games, rest * w / total))
    return games


# ---------------------------------------------------------------------------------------------- playing
def interval(per_deal):
    x = np.asarray(per_deal, dtype=np.float64)
    if len(x) < 2:
        return float(x.mean()) if len(x) else None, None
    half = 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean()), [float(x.mean() - half), float(x.mean() + half)]


def play(args, pool, games):
    import torch.multiprocessing as mp       # noqa: F401  (spawn start method set in main)
    import train_splendor as ts
    from benchmark_splendor import run
    from rl.splendor_model import load_checkpoint, model_state_of

    params = yaml.safe_load(open(args.config, encoding='UTF-8'))
    params['experiment_dir'] = os.path.dirname(os.path.abspath(args.config))
    params['_log_file'] = None
    params['num_game_process'] = args.num_game_process
    params['eval_num_mcts_simulation_per_step'] = args.sims
    params['model_param_dict']['device'] = ts.resolve_device(params['model_param_dict'].get('device'))
    name, path = args.candidate.split('=', 1)
    agent = model_state_of(load_checkpoint(path))
    members = {m['name']: m for m in pool['members']}
    gp = ts.make_game_pool(params, [agent, agent, agent], logging.INFO)
    out = []
    try:
        for k, (member, n) in enumerate(games.items()):
            m = members[member]
            if m['kind'] == 'model':
                gp.set_weights(ts.SLOT_BEST, model_state_of(load_checkpoint(os.path.join(ROOT, m['path']))), version=k + 1)
            tasks = ts.eval_tasks(dict(params, eval_task_num_games=n, eval_seed=m['seed']), k + 1)
            if m['kind'] == 'bot':
                tasks = [(i, seed, [f"bot:{m['policy']}" if s == ts.SLOT_BEST else s for s in slots], mode, rnd)
                         for i, seed, slots, mode, rnd in tasks]
            results = run(gp, tasks, f'{name} vs {member}')
            per_deal = {}
            for r in results:
                if r.get('failed') or r['end_reason'] != 'score':
                    s = .5
                else:
                    s = sum(w for w, slot in zip(r['rewards'], r['seat_server_ids']) if slot == ts.SLOT_NEW)
                per_deal.setdefault(str(r['seed']), []).append(s)
            total = sum(sum(v) for v in per_deal.values())
            missing = n - len(results)
            score = (total + .5 * missing) / n
            mean, ci = interval([np.mean(v) for v in per_deal.values()])
            record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'candidate': name, 'candidate_path': path,
                      'member': member, 'games': n, 'score': score, 'ci95': ci, 'sims': args.sims,
                      'config': os.path.relpath(os.path.abspath(args.config), ROOT), 'per_deal': per_deal}
            with open(RESULTS, 'a', encoding='UTF-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logging.info(f'{name} vs {member}: {score:.4f} {ci} over {n} games')
            out.append(record)
    finally:
        gp.close()
    return out


# ---------------------------------------------------------------------------------------------- report
def paired_reference(results, final, member, deals):
    """the final model's own score against `member` on the same deals (the regression baseline)"""
    scores = []
    for r in results:
        if r['candidate'] == final and r['member'] == member and r.get('per_deal'):
            for seed, v in r['per_deal'].items():
                if seed in deals:
                    scores.extend(v)
    return float(np.mean(scores)) if scores else None


def report(pool, candidate, records, all_results):
    ratings = fit_ratings(all_results, pool.get('reference', 'lookahead'))
    final = pool['final']
    lines, ok = [], True
    summary = {'candidate': candidate, 'final': final, 'members': {}, 'ratings': ratings}
    for r in records:
        member = r['member']
        base = None if member == final else paired_reference(all_results, final, member, set(r.get('per_deal', {})))
        entry = {'score': r['score'], 'ci95': r['ci95'], 'games': r['games'], 'final_score_same_deals': base}
        if member == final:
            entry['h2h_pass'] = r['score'] >= .55 and r['ci95'] is not None and r['ci95'][0] > .5
            ok &= entry['h2h_pass']
        elif base is not None:
            entry['regression'] = r['score'] < base - .05
            ok &= not entry['regression']
        summary['members'][member] = entry
        ci = '—' if r['ci95'] is None else f"{r['ci95'][0]:.3f}–{r['ci95'][1]:.3f}"
        extra = '' if base is None else f'（最终模型同发牌 {base:.3f}）'
        lines.append(f"  vs {member:16s} {r['score']:.4f}（{ci}，{r['games']} 局）{extra}")
    summary['replace_final'] = bool(ok and any(m == final for m in summary['members']))
    lines.append('等级分（lookahead = 0，±1 标准误）：')
    for n, (r, se) in sorted(ratings.items(), key=lambda kv: -kv[1][0]):
        lines.append(f'  {n:16s} {r:7.1f} ± {se:5.1f}')
    return summary, '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--candidate', help='NAME=PATH of the checkpoint to evaluate')
    parser.add_argument('--config', default=os.path.join(ROOT, 'models', 'splendor_final', 'config.yml'))
    parser.add_argument('--sims', type=int, default=512)
    parser.add_argument('--h2h', type=int, default=576, help='games against the final model')
    parser.add_argument('--budget', type=int, default=1152, help='games in total')
    parser.add_argument('--min-games', type=int, default=48, help='at least this many games per other member')
    parser.add_argument('--members', default=None, help='comma-separated subset of the pool (default: all)')
    parser.add_argument('--num-game-process', type=int, default=10)
    parser.add_argument('--ratings', action='store_true', help='only refit and print the ratings')
    parser.add_argument('--dry-run', action='store_true', help='print the game allocation and stop')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    pool = load_pool()
    results = load_results()
    if args.ratings:
        for n, (r, se) in sorted(fit_ratings(results, pool.get('reference', 'lookahead')).items(), key=lambda kv: -kv[1][0]):
            print(f'{n:16s} {r:7.1f} ± {se:5.1f}')
        return
    if not args.candidate or '=' not in args.candidate:
        parser.error('--candidate NAME=PATH')
    name = args.candidate.split('=', 1)[0]
    ratings = fit_ratings(results, pool.get('reference', 'lookahead')) if results else {}
    games = allocate(pool, name, ratings, args.h2h, args.budget, args.min_games)
    if args.members:
        keep = set(args.members.split(','))
        games = {k: v for k, v in games.items() if k in keep}
    print('分配：' + json.dumps(games, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    records = play(args, pool, games)
    summary, text = report(pool, name, records, load_results())
    print(text, flush=True)
    if args.out:
        with open(args.out, 'w', encoding='UTF-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    main()
