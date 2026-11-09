"""Diagnostic: how does the agent play against rule bots? One seat is either the model (policy-greedy: argmax of the
prior, no search) or the lookahead bot; three rule bots in the other seats; every deal is played with the agent in
each seat. Per agent seat: result, points, length, action mix, missed immediate wins, opponent threats (a market card
that would let an opponent reach 15 on his next turn) that were ignored and whether that opponent then won."""
import random, sys, json, time, yaml, collections, numpy as np
sys.path.insert(0, '.')
from rl.splendor_model import SplendorAlphaGoZero, load_checkpoint, model_state_of
from splendor.rl_env import SplendorRLEnv, action_to_index, index_to_action
from splendor.bots import choose_action, _winning_cards
from splendor.engine import CARDS
cfg, ckpt, n_deals, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
params = yaml.safe_load(open(cfg))
mp = dict(params['model_param_dict']); mp['device'] = 'cuda'
model = SplendorAlphaGoZero(**mp); model.load_state_dict(model_state_of(load_checkpoint(ckpt))); model.eval()

def policy_greedy(env):
    probs, _ = model.predict(np.frombuffer(env.observation_bytes(), dtype=np.uint8),
                             np.frombuffer(env.legal_mask_bytes(), dtype=np.uint8).astype(bool))
    return int(np.argmax(probs))

def play(agent, opp, deal, seat):
    env = SplendorRLEnv(max_turns=400); env.reset(seed=4000000000 + deal); rng = random.Random(deal * 10 + seat)
    st = collections.Counter(); pending_threat = None
    while not env.terminated:
        g = env.game; s = env.acting_player
        if s != seat:
            env.step(action_to_index(g, choose_action(g, rng, opp))); continue
        idx = policy_greedy(env) if agent == 'model' else action_to_index(g, choose_action(g, rng, 'lookahead'))
        act = index_to_action(g, idx)
        st[act.kind] += 1
        if g.phase == 'action':
            st['turns'] += 1
            if act.kind == 'take':
                st['take_%s' % ('2same' if len(act.colors) == 2 and act.colors[0] == act.colors[1] else len(set(act.colors)))] += 1
            if act.kind == 'buy':
                st['buy_L%d' % CARDS[act.card_id].level] += 1
                st['buy_from_reserve'] += act.card_id in g.players[seat].reserved
                st['buy_points'] += CARDS[act.card_id].points
                st['buy_with_gold'] += len(act.payment) > 5 and act.payment[5] > 0
            if act.kind == 'reserve':
                st['reserve_L%d' % CARDS[act.card_id].level] += 1
            wins = _winning_cards(g, seat)
            if wins:
                st['could_win'] += 1
                st['missed_win'] += not (act.kind == 'buy' and act.card_id in wins)
            threats = {o: _winning_cards(g, o) for o in range(4) if o != seat}
            threats = {o: c for o, c in threats.items() if c}
            if threats:
                st['threat_turns'] += 1
                denied = act.kind == 'reserve' and any(act.card_id in c for c in threats.values())
                st['denied'] += denied
                if not denied and not (act.kind == 'buy' and act.card_id in wins):
                    st['ignored'] += 1
        env.step(idx)
    final = env.game.players
    r = env.final_rewards()[seat] if env.game.phase == 'terminal' else 0.25
    pts = [p.points for p in final]
    st.update({'share': r, 'points': pts[seat], 'max_other': max(p for i, p in enumerate(pts) if i != seat),
               'rounds': env.game.turn / 4, 'cards': len(final[seat].purchased), 'nobles': len(final[seat].nobles),
               'reserved_left': len(final[seat].reserved), 'games': 1})
    return st

res = {}
t0 = time.time()
for agent in ('model', 'lookahead'):
    for opp in ('heuristic', 'lookahead'):
        tot = collections.Counter()
        for d in range(n_deals):
            for seat in range(4):
                tot.update(play(agent, opp, d, seat))
        n = tot['games']
        summary = {k: round(v / n, 3) for k, v in sorted(tot.items()) if k != 'games'}
        summary['games'] = n
        res[f'{agent}_vs_3{opp}'] = summary
        print(agent, opp, json.dumps(summary), round(time.time() - t0), flush=True)
json.dump({'ckpt': ckpt, 'results': res}, open(out, 'w'), indent=1)
