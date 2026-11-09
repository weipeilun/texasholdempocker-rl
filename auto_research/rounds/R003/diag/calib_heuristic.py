"""Calibrate a win-share model from the lookahead bot's position evaluation: P(seat i wins) = softmax_i(score_i / T).

Plays games between rule bots (seats rotated, 3 lookahead + 1 heuristic mixed in some games), records at every
decision the evaluate_position score of all four players and the final winner, and fits T by maximum likelihood
(grid search), overall and by game stage (leader's points). Also reports the log loss of the constant 0.25."""
import math, random, sys, json
import numpy as np
sys.path.insert(0, '.')
from splendor.rl_env import SplendorRLEnv, action_to_index
from splendor.bots import choose_action, evaluate_position, WIN_VALUE

n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rows = []   # (scores[4], leader_points, winner)
for g in range(n_games):
    env = SplendorRLEnv(max_turns=400)
    env.reset(seed=900000 + g)
    rng = random.Random(g)
    policies = ['lookahead'] * 4
    if g % 2:
        policies[g % 4] = 'heuristic'
    states = []
    while not env.terminated:
        game = env.game
        if game.phase == 'action':
            scores = [evaluate_position(game, i) for i in range(4)]
            if max(scores) < WIN_VALUE:
                states.append((scores, max(p.points for p in game.players)))
        action = choose_action(game, rng, policies[game.current_player])
        env.step(action_to_index(game, action))
    rewards = env.final_rewards()
    if env.game.phase != 'terminal' or max(rewards) < 1:
        continue
    winner = int(np.argmax(rewards))
    rows.extend((s, lead, winner) for s, lead in states)

S = np.array([r[0] for r in rows]); L = np.array([r[1] for r in rows]); W = np.array([r[2] for r in rows])

def nll(T, sel):
    x = S[sel] / T
    x = x - x.max(1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(1, keepdims=True))
    return -logp[np.arange(sel.sum()), W[sel]].mean()

grid = [0.25, 0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8.]
out = {'games': n_games, 'positions': len(rows), 'const_nll': math.log(4)}
all_sel = np.ones(len(rows), bool)
best = min(grid, key=lambda T: nll(T, all_sel))
out['T_all'] = best; out['nll_all'] = nll(best, all_sel)
out['by_stage'] = {}
for a, b in [(0, 4), (4, 8), (8, 11), (11, 13), (13, 16)]:
    sel = (L >= a) & (L < b)
    if sel.sum() < 100:
        continue
    t = min(grid, key=lambda T: nll(T, sel))
    out['by_stage'][f'{a}-{b - 1}'] = {'n': int(sel.sum()), 'T': t, 'nll': round(nll(t, sel), 4),
                                       'nll_at_T_all': round(nll(best, sel), 4)}
print(json.dumps(out, indent=1))
