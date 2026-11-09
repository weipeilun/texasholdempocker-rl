"""Diagnostic: does v_theta rank sibling after-states? One seat plays 1-ply value-greedy (argmax over legal actions of
its own win share in the after-state, refills averaged over up to K sampled cards, terminal states exact); three
rule bots in the other seats; every deal played with the agent in each seat. Also a policy-greedy agent (argmax prior)
on the same deals, for reference."""
import random, sys, json, time, yaml, numpy as np
sys.path.insert(0, '.')
from rl.splendor_model import SplendorAlphaGoZero, load_checkpoint, model_state_of
from splendor.rl_env import SplendorRLEnv, action_to_index, to_absolute
from splendor.bots import choose_action
ckpt, opponent, n_deals, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
K = 4
params = yaml.safe_load(open('experiments/splendor_v4_20261104/config.yml'))
mp = dict(params['model_param_dict']); mp['device'] = 'cuda'
model = SplendorAlphaGoZero(**mp); model.load_state_dict(model_state_of(load_checkpoint(ckpt))); model.eval()

def after_states(env, a, rng):
    e = env.copy_for_search(); e.step(a, sample_chance=False, check=False)
    if not e.at_chance_node:
        return [e]
    pool = e.chance_pool(); cards = rng.sample(pool, min(K, len(pool))); out = []
    for c in cards:
        e2 = e.copy_for_search(); e2.resolve_chance(c); out.append(e2)
    return out

def value_greedy(env, rng):
    me = env.acting_player; legal = list(env.legal_indices())
    scores, obs, masks, refs = {}, [], [], []
    for a in legal:
        states = after_states(env, a, rng); scores[a] = [0., len(states)]
        for e in states:
            if e.terminated and e.game.phase == 'terminal':
                scores[a][0] += float(e.final_rewards()[me])
            else:
                obs.append(np.frombuffer(e.observation_bytes(), dtype=np.uint8)); masks.append(np.frombuffer(e.legal_mask_bytes(), dtype=np.uint8).astype(bool)); refs.append((a, e.acting_player))
    if obs:
        _, values = model.predict_batch(obs, masks)
        for (a, seat), v in zip(refs, values):
            scores[a][0] += float(to_absolute(v, seat)[me])
    return max(legal, key=lambda a: scores[a][0] / scores[a][1] + rng.random() * 1e-6)

def policy_greedy(env, rng):
    probs, _ = model.predict(np.frombuffer(env.observation_bytes(), dtype=np.uint8), np.frombuffer(env.legal_mask_bytes(), dtype=np.uint8).astype(bool))
    return int(np.argmax(probs))

agent = value_greedy if mode == 'value' else policy_greedy
share, wins, t0 = 0., 0, time.time()
for d in range(n_deals):
    for seat in range(4):
        env = SplendorRLEnv(max_turns=400); env.reset(seed=7000 + d); rng = random.Random(d * 10 + seat)
        while not env.terminated:
            s = env.acting_player
            if s == seat:
                env.step(agent(env, rng))
            else:
                env.step(action_to_index(env.game, choose_action(env.game, rng, opponent)))
        r = env.final_rewards()[seat] if env.game.phase == 'terminal' else 0.25
        share += r; wins += r == 1.
n = 4 * n_deals
print(json.dumps({'ckpt': ckpt, 'mode': mode, 'opponent': opponent, 'games': n, 'win_share': share / n, 'win_rate': wins / n, 'seconds': round(time.time() - t0)}))
