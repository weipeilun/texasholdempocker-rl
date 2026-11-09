"""The final Splendor agent (auto_research R007, iteration 200): the network plus the deployment search
(512 simulations, 8 considered actions, the other players modelled by the argmax of the network's prior).

    python models/splendor_final/play_agent.py --games 4 --opponent lookahead [--sims 512] [--device cuda]

Plays games in one process, the agent in seat k = game index mod 4 against three rule bots, and prints each
result. Slow (a search per agent move, no batching across games); for benchmarks use benchmark_splendor.py
(see README.md). Agent.act(env) is the interface to embed the agent elsewhere."""
import argparse
import os
import random
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from rl.splendor_model import SplendorAlphaGoZero, load_checkpoint, model_state_of   # noqa: E402
from rl.splendor_mcts import SplendorMCTS                                           # noqa: E402
from splendor.rl_env import SplendorRLEnv, action_to_index, N_ACTIONS               # noqa: E402
from splendor.bots import choose_action                                             # noqa: E402
import train_splendor as ts                                                         # noqa: E402  (make_local_predictor)


class Agent:
    def __init__(self, weights=os.path.join(HERE, 'weights.pth'), config=os.path.join(HERE, 'config.yml'),
                 device='cuda', sims=512, considered=8, opponent_policy='prior', seed=0):
        params = yaml.safe_load(open(config, encoding='UTF-8'))
        model_params = dict(params['model_param_dict'], device=device)
        self.model = SplendorAlphaGoZero(**model_params)
        self.model.load_state_dict(model_state_of(load_checkpoint(weights)))
        self.model.eval()
        self.predictor = ts.make_local_predictor(self.model)
        self.params = params
        self.sims, self.considered, self.opponent_policy = sims, considered, opponent_policy
        self.rng = random.Random(seed)

    def act(self, env):
        """action index (splendor.rl_env) for env's acting player; env must be at a decision state"""
        p = self.params
        mcts = SplendorMCTS(N_ACTIONS, self.predictor, n_simulation=self.sims,
                            max_num_considered_actions=self.considered, gumbel_scale=0.,
                            value_scale=p.get('eval_mcts_value_scale') or p['mcts_value_scale'],
                            maxvisit_init=p['mcts_maxvisit_init'], rescale_values=p['mcts_rescale_values'],
                            max_search_depth=p['mcts_max_search_depth'], opponent_policy=self.opponent_policy,
                            seed=self.rng.getrandbits(32))
        action, _ = mcts.simulate(env)
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--opponent', default='lookahead', help='rule bot of the other three seats (splendor/bots.py)')
    parser.add_argument('--sims', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    agent = Agent(device=args.device, sims=args.sims, seed=args.seed)
    total = 0.
    for g in range(args.games):
        seat = g % 4
        env = SplendorRLEnv(max_turns=400)
        env.reset(seed=args.seed * 1000 + g // 4)
        rng = random.Random(args.seed * 1000 + g)
        t0 = time.time()
        while not env.terminated:
            if env.acting_player == seat:
                env.step(agent.act(env))
            else:
                env.step(action_to_index(env.game, choose_action(env.game, rng, args.opponent)))
        share = float(env.final_rewards()[seat]) if env.game.phase == 'terminal' else 0.25
        total += share
        points = [pl.points for pl in env.game.players]
        print(f'game {g}: agent seat {seat}, points {points}, win share {share:.2f}, {time.time() - t0:.0f}s', flush=True)
    print(f'mean win share {total / args.games:.3f} over {args.games} games (equal strength: 0.25)')


if __name__ == '__main__':
    main()
