"""Diagnostic: is own-turn depth what beats lookahead? 'lookahead2' = for each own action, sample the card draw,
let the opponents play (opponent model: lookahead or heuristic) until it is our action phase again, then the best
evaluate_position over our next actions (with lookahead's deny bonus at the first step). K samples per action."""
import random, sys, json, time, math
sys.path.insert(0, '/data/playground/texasholdempocker-rl')
from splendor.engine import new_game
from splendor.bots import choose_action, evaluate_position, lookahead_action, _winning_cards, DEFAULT_WEIGHTS, WIN_VALUE
from multiprocessing import Pool
OPP_MODEL = sys.argv[1] if len(sys.argv) > 1 else 'lookahead'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N_DEALS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
OPP = sys.argv[4] if len(sys.argv) > 4 else 'lookahead'

def best_next(game, seat):
    best = -math.inf
    for a in game.legal_actions(all_payments=False):
        best = max(best, evaluate_position(game.apply(a), seat))
    return best

def la2(game, rng):
    seat = game.current_player
    if game.phase != 'action':
        return lookahead_action(game, rng)
    w = DEFAULT_WEIGHTS
    threats = set()
    for o in range(4):
        if o != seat:
            threats.update(_winning_cards(game, o))
    best, best_score = None, -math.inf
    for a in game.legal_actions(all_payments=False):
        first = evaluate_position(game.apply(a), seat)
        if first >= WIN_VALUE:
            return a
        total = 0.
        for k in range(K):
            g = game.clone(); g.step(a, rng, validate=False)
            while g.phase != 'terminal' and not (g.current_player == seat and g.phase == 'action') and g.turn < 400:
                pol = OPP_MODEL if g.current_player != seat else 'lookahead'
                g.step(choose_action(g, rng, pol), rng, validate=False)
            if g.phase == 'terminal':
                total += WIN_VALUE if g.rewards()[seat] > 0.5 else -WIN_VALUE
            else:
                total += best_next(g, seat)
        score = total / K
        if a.kind == 'reserve' and a.card_id in threats:
            score += w['deny']
        score += rng.random() * .01
        if score > best_score:
            best, best_score = a, score
    return best

def play(args):
    seed = args
    out = []
    for seat in range(4):
        rng = random.Random(seed * 4 + seat); g = new_game(random.Random(seed))
        while g.phase in ('action', 'noble', 'discard') and g.turn < 400:
            if g.current_player == seat:
                a = la2(g, rng)
            else:
                a = choose_action(g, rng, OPP)
            g.step(a, rng, validate=False)
        if g.phase == 'terminal':
            out.append(float(g.rewards()[seat]))
    return out

if __name__ == '__main__':
    t = time.time()
    with Pool(12) as pool:
        rs = [x for lst in pool.map(play, [4000000000 + i for i in range(N_DEALS)]) for x in lst]
    res = {'agent': f'lookahead2(opp_model={OPP_MODEL},K={K})', 'opponents': OPP, 'games': len(rs), 'win_share': sum(rs) / len(rs), 'sec': round(time.time() - t)}
    print(json.dumps(res), flush=True)
