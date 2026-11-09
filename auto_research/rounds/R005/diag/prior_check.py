import random, sys, json, yaml, collections, numpy as np
sys.path.insert(0, '.')
from rl.splendor_model import SplendorAlphaGoZero, load_checkpoint, model_state_of
from splendor.rl_env import SplendorRLEnv, action_to_index, index_to_action, OFFSET_TAKE_DOUBLE, OFFSET_RESERVE
from splendor.bots import choose_action
cfg, ckpt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
params = yaml.safe_load(open(cfg)); mp = dict(params['model_param_dict']); mp['device'] = 'cuda'
model = SplendorAlphaGoZero(**mp); model.load_state_dict(model_state_of(load_checkpoint(ckpt))); model.eval()
def kind(g, i):
    a = index_to_action(g, i)
    if a.kind == 'take':
        return 'take2same' if len(a.colors) == 2 and a.colors[0] == a.colors[1] else 'take3' if len(a.colors) == 3 else 'take_other'
    if a.kind in ('buy', 'reserve'):
        from splendor.engine import CARDS
        return f'{a.kind}_L{CARDS[a.card_id].level}'
    return a.kind
agg = collections.defaultdict(list); n_dec = 0
for d in range(n):
    env = SplendorRLEnv(max_turns=400); env.reset(seed=5000 + d); rng = random.Random(d)
    while not env.terminated:
        g = env.game
        a = action_to_index(g, choose_action(g, rng, 'lookahead'))
        if g.phase == 'action':
            legal = list(env.legal_indices())
            probs, _ = model.predict(np.frombuffer(env.observation_bytes(), dtype=np.uint8), np.frombuffer(env.legal_mask_bytes(), dtype=np.uint8).astype(bool))
            mine = int(np.argmax(probs)); n_dec += 1
            k = kind(g, a)
            agg['lookahead_kind:' + k].append(1)
            agg['agree'].append(mine == a)
            agg[f'agree|{k}'].append(mine == a)
            agg[f'prior_of_lookahead_choice|{k}'].append(float(probs[a]))
            rank = int((probs[legal] > probs[a]).sum())
            agg[f'rank|{k}'].append(rank)
            m2 = [i for i in legal if OFFSET_TAKE_DOUBLE <= i < OFFSET_RESERVE]
            if m2:
                agg['mass_take2same_when_legal'].append(float(probs[m2].sum()))
                agg['n_take2same_legal'].append(len(m2))
            agg['model_kind:' + kind(g, mine)].append(1)
        env.step(a)
out = {k: (round(float(np.mean(v)), 4) if not k.startswith(('lookahead_kind', 'model_kind')) else len(v)) for k, v in sorted(agg.items())}
out['decisions'] = n_dec
print(json.dumps(out, indent=0))
