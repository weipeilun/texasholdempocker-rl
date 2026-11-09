"""Training-side checks (need torch): model init, value masking, checkpoints, evaluation score, seeds, resume."""
import copy
import json
import logging
import math
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
    import yaml
except ImportError:   # pragma: no cover
    torch = None

if torch is not None:
    import train_splendor as ts
    from rl.splendor_model import SplendorAlphaGoZero, init_weights_from, load_checkpoint, lr_factor, save_checkpoint
    from splendor.rl_env import N_ACTIONS, N_PLAYERS, SplendorRLEnv

TINY = {'device': 'cpu', 'embedding_dim': 32, 'num_layers': 2, 'dim_feedforward': 64, 'transformer_head_dim': 16,
        'batch_size': 8, 'num_inference_per_step': 0, 'inference_cuda_graph_batch_sizes': None, 'dropout': 0.,
        'train_symmetry_augmentation': False, 'transition_buffer_len': 1000, 'lr_warmup_steps': 1}


def tiny_params(experiment_dir, **overrides):
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'splendor_default.yml'), encoding='UTF-8') as f:
        params = yaml.safe_load(f)
    params.update({'num_game_process': 0, 'num_iterations': 2, 'num_self_play_games_per_iteration': 2,
                   'eval_model_per_iteration': 1, 'eval_task_num_games': 6, 'max_game_turns': 10,
                   'first_train_data_step': 1, 'num_mcts_simulation_per_step': 4, 'mcts_max_num_considered_actions': 4,
                   'mcts_fast_num_simulation': 2, 'mcts_fast_max_num_considered_actions': 2, 'mcts_full_search_prob': 0.5,
                   'eval_num_mcts_simulation_per_step': 4, 'eval_mcts_max_num_considered_actions': 4,
                   'train_sample_reuse': 4., 'eval_min_completion_rate': 0., 'checkpoint_archive_every': 1,
                   'experiment_dir': experiment_dir})
    params['model_param_dict'] = dict(params['model_param_dict'], **TINY)
    params.update(overrides)
    return params


def sample_batch(n, seed=0):
    rng = np.random.default_rng(seed)
    obs, probs, masks, values = [], [], [], []
    for k in range(n):
        env = SplendorRLEnv()
        env.reset(seed * 100 + k)
        obs.append(np.frombuffer(env.observation_bytes(), dtype=np.uint8).astype(np.int32))
        mask = env.legal_mask()
        p = np.where(mask, rng.random(N_ACTIONS), 0.)
        probs.append((p / p.sum()).astype(np.float32))
        masks.append(mask)
        v = np.zeros(N_PLAYERS, dtype=np.float32)
        v[k % N_PLAYERS] = 1.
        values.append(v)
    return np.stack(obs), np.stack(probs), np.stack(masks), np.stack(values)


@unittest.skipIf(torch is None, 'needs torch')
class ModelTest(unittest.TestCase):
    def test_encoder_layers_start_with_different_weights(self):
        torch.manual_seed(0)
        model = SplendorAlphaGoZero(**dict(TINY, num_layers=3))
        layers = model.model.transform_encoder.layers
        for name in ('self_attn.in_proj_weight', 'self_attn.out_proj.weight', 'linear1.weight', 'linear2.weight'):
            weights = [dict(layer.named_parameters())[name] for layer in layers]
            for a in range(len(weights)):
                for b in range(a + 1, len(weights)):
                    self.assertFalse(torch.equal(weights[a], weights[b]), name)

    def test_value_loss_ignores_truncated_samples(self):
        torch.manual_seed(1)
        model = SplendorAlphaGoZero(**TINY)
        obs, probs, masks, values = sample_batch(8)
        weights = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
        values_masked = values.copy()
        values_masked[4:] = 0.
        full = model.evaluate_samples(obs[:4], probs[:4], masks[:4], values[:4])
        masked = model.evaluate_samples(obs, probs, masks, values_masked, weights)
        self.assertAlmostEqual(full['value_ce'], masked['value_ce'], places=5)
        self.assertEqual(masked['num_value_samples'], 4)
        _, value_loss = model.learn(obs, probs, masks, values_masked, weights)
        self.assertAlmostEqual(value_loss, full['value_ce'], places=4)

    def test_init_weights_strict_and_partial(self):
        with tempfile.TemporaryDirectory() as d:
            torch.manual_seed(2)
            source = SplendorAlphaGoZero(**TINY)
            path = os.path.join(d, 'old.pth')
            torch.save({'model': source.state_dict(), 'optimizer': source.optimizer.state_dict()}, path)   # older format
            same = SplendorAlphaGoZero(**TINY)
            init_weights_from(same, path)
            for k, v in source.state_dict().items():
                self.assertTrue(torch.equal(v, same.state_dict()[k]), k)
            deeper = SplendorAlphaGoZero(**dict(TINY, num_layers=3))
            with self.assertRaises(ValueError):
                init_weights_from(deeper, path)
            report = init_weights_from(deeper, path, strict=False)
            self.assertTrue(report['missing'])
            self.assertTrue(torch.equal(deeper.state_dict()['model.reward_value_dense.weight'],
                                        source.state_dict()['model.reward_value_dense.weight']))


@unittest.skipIf(torch is None, 'needs torch')
class LearningRateTest(unittest.TestCase):
    def test_schedules(self):
        self.assertAlmostEqual(lr_factor(0, 10), 0.1)
        self.assertAlmostEqual(lr_factor(50, 10), 1.)
        cosine = {'type': 'cosine', 'decay_steps': 100, 'min_factor': 0.1}
        self.assertAlmostEqual(lr_factor(10, 10, cosine), 1.)
        self.assertAlmostEqual(lr_factor(60, 10, cosine), 0.55)
        self.assertAlmostEqual(lr_factor(500, 10, cosine), 0.1)
        piecewise = {'type': 'piecewise', 'points': [[0, 1.], [100, 0.5], [300, 0.1]]}
        self.assertAlmostEqual(lr_factor(50, 0, piecewise), 0.75)
        self.assertAlmostEqual(lr_factor(1000, 0, piecewise), 0.1)
        with self.assertRaises(ValueError):
            SplendorAlphaGoZero(**dict(TINY, lr_schedule={'type': 'nope'}))

    def test_every_step_uses_the_schedule(self):
        torch.manual_seed(4)
        schedule = {'type': 'piecewise', 'points': [[0, 1.], [4, 0.25]]}
        model = SplendorAlphaGoZero(**dict(TINY, learning_rate=1e-3, lr_warmup_steps=0, lr_schedule=schedule))
        obs, probs, masks, values = sample_batch(8)
        applied = list()
        for _ in range(6):
            model.learn(obs, probs, masks, values)
            applied.append(model.optimizer.param_groups[0]['lr'])
        np.testing.assert_allclose(applied, [1e-3 * lr_factor(k, 0, schedule) for k in range(6)])
        self.assertAlmostEqual(applied[-1], 2.5e-4)

    def test_resume_follows_the_config_not_the_checkpoint(self):
        """the learning rate, weight decay and betas of a resumed run come from its (edited) config.yml"""
        with tempfile.TemporaryDirectory() as d:
            experiment = os.path.join(d, 'exp')
            params = tiny_params(experiment)
            ts.train_process(copy.deepcopy(params), 0, logging.WARNING)
            paths = ts.experiment_paths(experiment)
            first = load_checkpoint(paths['latest'])
            steps = first['train_state']['train_step_num']
            self.assertGreater(steps, 0)
            with open(paths['config'], encoding='UTF-8') as f:
                resume_params = yaml.safe_load(f)
            resume_params['num_iterations'] = 3
            resume_params['experiment_dir'] = experiment
            schedule = {'type': 'piecewise', 'points': [[0, 1.], [steps, 0.1]]}
            resume_params['model_param_dict'].update(learning_rate=5e-3, lr_schedule=schedule, weight_decay=0.02,
                                                     adam_betas=[0.8, 0.95])
            with open(paths['config'], 'w', encoding='UTF-8') as f:
                yaml.safe_dump(resume_params, f)
            ts.train_process(copy.deepcopy(resume_params), 0, logging.WARNING, resume=True)
            records = [json.loads(line) for line in open(paths['metrics'], encoding='UTF-8')]
            last = [r for r in records if r['kind'] == 'iteration'][-1]
            self.assertGreater(last['train_step_num'], steps)
            self.assertAlmostEqual(last['lr'], 5e-3 * lr_factor(last['train_step_num'] - 1, 1, schedule))
            self.assertAlmostEqual(last['lr'], 5e-4)          # past the last point of the schedule
            second = load_checkpoint(paths['latest'])
            groups = second['optimizer']['param_groups']
            self.assertEqual([g['weight_decay'] for g in groups], [0.02, 0.])
            self.assertEqual([tuple(g['betas']) for g in groups], [(0.8, 0.95)] * 2)


@unittest.skipIf(torch is None, 'needs torch')
class BenchmarkTest(unittest.TestCase):
    def test_bot_seats_and_bench_records(self):
        torch.manual_seed(5)
        model = SplendorAlphaGoZero(**TINY)
        params = tiny_params('unused', max_game_turns=40)
        predictor = {ts.SLOT_NEW: ts.make_local_predictor(model)}
        seats = ['bot:lookahead', ts.SLOT_NEW, 'bot:heuristic', 'bot:random']
        a = ts.play_game(1, 777, seats, predictor, params, 'bench')
        b = ts.play_game(1, 777, seats, predictor, params, 'bench')
        self.assertEqual((a['turns'], a['points']), (b['turns'], b['points']))
        self.assertGreater(a['bot_decisions'], 0)
        self.assertGreater(a['decisions'], a['bot_decisions'])
        with self.assertRaises(ValueError):
            ts.play_game(1, 777, ['bot:nobody', 0, 0, 0], predictor, params, 'bench')
        with tempfile.TemporaryDirectory() as d:
            experiment = os.path.join(d, 'exp')
            params = tiny_params(experiment, num_iterations=1, bench_opponents=['bot:heuristic', 'bot:random'],
                                 bench_num_games=4, max_game_turns=40)
            ts.train_process(copy.deepcopy(params), 0, logging.WARNING)
            records = [json.loads(line) for line in open(ts.experiment_paths(experiment)['metrics'], encoding='UTF-8')]
            bench = [r for r in records if r['kind'] == 'bench']
            self.assertEqual(len(bench), 1)
            self.assertEqual(sorted(bench[0]['opponents']), ['bot:heuristic', 'bot:random'])
            self.assertEqual(bench[0]['opponents']['bot:random']['games'], 4)
            self.assertTrue([r for r in records if r['kind'] == 'eval'])

    def test_bench_score(self):
        new = ts.SLOT_NEW
        bot = 'bot:heuristic'
        results = [
            {'end_reason': 'score', 'rewards': [1., 0, 0, 0], 'points': [15, 3, 4, 5], 'turns': 100, 'seat_server_ids': [new, bot, bot, bot]},
            {'end_reason': 'score', 'rewards': [0, 1., 0, 0], 'points': [9, 15, 4, 5], 'turns': 120, 'seat_server_ids': [new, bot, bot, bot]},
            {'end_reason': 'truncated', 'rewards': [0.] * 4, 'points': [1, 1, 1, 1], 'turns': 400, 'seat_server_ids': [bot, new, bot, bot]},
            {'end_reason': 'draw', 'rewards': [.25] * 4, 'points': [2, 2, 2, 2], 'turns': 40, 'seat_server_ids': [bot, bot, new, bot]},
        ]
        m = ts.bench_score(results)[bot]
        self.assertAlmostEqual(m['win_share'], (1 + 0 + .25 + .25) / 4)
        self.assertAlmostEqual(m['win_rate'], 0.25)
        self.assertAlmostEqual(m['completion'], 0.75)
        self.assertAlmostEqual(m['mean_rounds_when_winning'], 25.)
        self.assertAlmostEqual(m['mean_points'], (15 + 9 + 2) / 3)


@unittest.skipIf(torch is None, 'needs torch')
class SelfPlayVariantsTest(unittest.TestCase):
    def test_endgame_full_search(self):
        torch.manual_seed(6)
        model = SplendorAlphaGoZero(**TINY)
        predictor = {ts.SLOT_CURRENT: ts.make_local_predictor(model)}
        seats = [ts.SLOT_CURRENT] * 4
        off = ts.play_game(3, 991, seats, predictor, tiny_params('unused', mcts_full_search_prob=0., max_game_turns=400), 'train')
        self.assertEqual(off['full_searches'], 0)
        self.assertEqual(len(off['samples']['obs']), 0)
        on = ts.play_game(3, 991, seats, predictor, tiny_params('unused', mcts_full_search_prob=0., max_game_turns=400,
                                                                mcts_endgame_full_search_points=1), 'train')
        self.assertGreater(on['full_searches'], 0)
        self.assertEqual(len(on['samples']['obs']), on['full_searches'])

    def test_self_play_seats(self):
        none = ts.self_play_seats({'self_play_bot_prob': 0.}, 5)
        self.assertEqual(none, [ts.SLOT_CURRENT] * 4)
        params = {'self_play_bot_prob': 1., 'self_play_bot_opponents': ['bot:lookahead', 'bot:heuristic']}
        for seed in range(20):
            seats = ts.self_play_seats(params, seed)
            self.assertEqual(sum(isinstance(x, str) for x in seats), 1)
            self.assertEqual(seats, ts.self_play_seats(params, seed))
        some = [ts.self_play_seats({'self_play_bot_prob': 0.3}, seed) for seed in range(200)]
        self.assertTrue(20 < sum(any(isinstance(x, str) for x in seats) for seats in some) < 100)
        with self.assertRaises(ValueError):
            ts.self_play_seats({'self_play_bot_prob': 1., 'self_play_bot_opponents': ['bot:nobody']}, 1)

    def test_training_with_bot_opponents(self):
        with tempfile.TemporaryDirectory() as d:
            experiment = os.path.join(d, 'exp')
            params = tiny_params(experiment, num_iterations=2, num_self_play_games_per_iteration=6, self_play_bot_prob=1.,
                                 mcts_endgame_full_search_points=3, bench_opponents=[])
            ts.train_process(copy.deepcopy(params), 0, logging.WARNING)
            records = [json.loads(line) for line in open(ts.experiment_paths(experiment)['metrics'], encoding='UTF-8')]
            iterations = [r for r in records if r['kind'] == 'iteration']
            self.assertTrue(iterations)
            self.assertTrue(all(r['bot_games'] == r['games'] for r in iterations))


@unittest.skipIf(torch is None, 'needs torch')
class TDLambdaTest(unittest.TestCase):
    def test_lambda_returns(self):
        z = np.array([1., 0., 0., 0.])
        v = [np.array([.4, .2, .2, .2]), np.array([.1, .5, .2, .2]), np.array([.3, .3, .2, .2])]
        np.testing.assert_allclose(ts.lambda_returns(v, z, 1.), np.tile(z, (3, 1)))
        lam = .8
        g = ts.lambda_returns(v, z, lam)
        np.testing.assert_allclose(g[2], z)                                   # last decision: the outcome
        np.testing.assert_allclose(g[1], (1 - lam) * v[2] + lam * z)
        np.testing.assert_allclose(g[0], (1 - lam) * v[1] + lam * g[1])
        np.testing.assert_allclose(ts.lambda_returns(v, z, 0.)[0], v[1])      # one-step TD
        # the network's values leave out the "no winner" class: returns are renormalised win shares
        g = ts.lambda_returns([np.array([.3, .3, .1, .1]), np.array([.2, .2, .2, .2])], z, .5)
        np.testing.assert_allclose(g.sum(axis=1), 1.)
        self.assertEqual(len(ts.lambda_returns([], z, .5)), 0)

    def test_root_search_value(self):
        torch.manual_seed(8)
        model = SplendorAlphaGoZero(**TINY)
        env = SplendorRLEnv()
        env.reset(5)
        mcts = ts.SplendorMCTS(N_ACTIONS, ts.make_local_predictor(model), n_simulation=16, max_num_considered_actions=4)
        mcts.simulate(env)
        value = mcts.root_search_value
        self.assertEqual(value.shape, (N_PLAYERS,))
        self.assertTrue(np.isfinite(value).all() and (value >= 0).all())
        self.assertLessEqual(value.sum(), 1. + 1e-6)

    def test_train_game_targets(self):
        from rl.splendor_data import check_sample
        torch.manual_seed(9)
        model = SplendorAlphaGoZero(**TINY)
        predictor = {ts.SLOT_CURRENT: ts.make_local_predictor(model)}
        seats = [ts.SLOT_CURRENT] * 3 + ['bot:heuristic']
        base = dict(mcts_full_search_prob=.5, max_game_turns=400)
        plain = ts.play_game(4, 123, seats, predictor, tiny_params('unused', **base), 'train')
        td = ts.play_game(4, 123, seats, predictor, tiny_params('unused', value_target_lambda=.9, **base), 'train')
        self.assertEqual(td['end_reason'], 'score')
        s = td['samples']
        self.assertGreater(len(s['obs']), 2)
        np.testing.assert_array_equal(s['outcome'], plain['samples']['values'])   # same game, same outcome
        np.testing.assert_allclose(s['values'].sum(axis=1), 1., atol=1e-5)
        self.assertFalse(np.allclose(s['values'], s['outcome']))
        for i in range(len(s['obs'])):                                             # writable samples
            self.assertIsNone(check_sample(s['obs'][i], s['probs'][i], s['mask'][i], s['values'][i],
                                           float(s['value_weight'][i])))

    def test_training_records_both_held_out_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            experiment = os.path.join(d, 'exp')
            params = tiny_params(experiment, num_iterations=2, value_target_lambda=.9, bench_opponents=[])
            ts.train_process(copy.deepcopy(params), 0, logging.WARNING)
            records = [json.loads(line) for line in open(ts.experiment_paths(experiment)['metrics'], encoding='UTF-8')]
            iterations = [r for r in records if r['kind'] == 'iteration' and r['held_out'] is not None]
            self.assertTrue(iterations)
            self.assertTrue(all(r['held_out_target'] is not None for r in iterations))


@unittest.skipIf(torch is None, 'needs torch')
class ObservationV2Test(unittest.TestCase):
    def _state(self, seed=11):
        env = SplendorRLEnv()
        env.reset(seed)
        rng = np.random.default_rng(seed)
        for _ in range(int(rng.integers(30, 60))):          # some purchases, tokens and reservations
            legal = np.flatnonzero(np.frombuffer(env.legal_mask_bytes(), dtype=np.uint8))
            env.step(int(rng.choice(legal)))
            if env.terminated:
                break
        return env

    def test_derived_fields(self):
        from splendor.rl_env import (FIELD_LAYOUT, CARDS, NOBLES, card_shortfall, noble_missing, encode_observation,
                                     DERIVED_EMPTY)
        for seed in (11, 12, 13):
            env = self._state(seed)
            game = env.game
            obs = encode_observation(game)
            me = game.current_player
            seats = [game.players[(me + k) % N_PLAYERS] for k in range(N_PLAYERS)]
            at = {key: i for i, key in enumerate(FIELD_LAYOUT)}
            clip = lambda v: min(v, 9)
            for level in range(3):
                for slot in range(4):
                    card_id = game.market[level][slot]
                    name = f'L{level + 1}S{slot + 1}'
                    own, rival = obs[at[('card_short_owner', 'market', name)]], obs[at[('card_short_rival', 'market', name)]]
                    if card_id is None:
                        self.assertEqual((own, rival), (DERIVED_EMPTY, DERIVED_EMPTY))
                        continue
                    cost = CARDS[card_id].cost
                    self.assertEqual(own, clip(card_shortfall(cost, seats[0].bonuses, seats[0].tokens)))
                    self.assertEqual(rival, clip(min(card_shortfall(cost, p.bonuses, p.tokens) for p in seats[1:])))
                    if game.phase == 'action':                    # shortfall 0 <=> the buy action is legal
                        from splendor.rl_env import OFFSET_BUY_MARKET
                        legal = np.frombuffer(env.legal_mask_bytes(), dtype=np.uint8)[OFFSET_BUY_MARKET + level * 4 + slot]
                        self.assertEqual(bool(legal), own == 0)
            for k in range(N_PLAYERS):
                for i in range(3):
                    own = obs[at[('card_short_owner', f'player{k}', f'reserved{i + 1}')]]
                    if i < len(seats[k].reserved):
                        cost = CARDS[seats[k].reserved[i]].cost
                        self.assertEqual(own, clip(card_shortfall(cost, seats[k].bonuses, seats[k].tokens)))
                    else:
                        self.assertEqual(own, DERIVED_EMPTY)
            for i in range(5):
                own, rival = obs[at[('noble_missing_owner', 'global', f'noble{i + 1}')]], obs[at[('noble_missing_rival', 'global', f'noble{i + 1}')]]
                if i < len(game.nobles):
                    cost = NOBLES[game.nobles[i]].cost
                    self.assertEqual(own, clip(noble_missing(cost, seats[0].bonuses)))
                    self.assertEqual(rival, clip(min(noble_missing(cost, p.bonuses) for p in seats[1:])))
                else:
                    self.assertEqual((own, rival), (DERIVED_EMPTY, DERIVED_EMPTY))

    def test_shortfall_matches_affordability(self):
        from splendor.rl_env import card_shortfall
        self.assertEqual(card_shortfall([3, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]), 1)
        self.assertEqual(card_shortfall([3, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1]), 0)
        self.assertEqual(card_shortfall([2, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1]), 2)

    def test_derived_fields_move_with_their_slot(self):
        from splendor.rl_env import encode_observation, symmetry_indices, SYMMETRY_GROUPS, FIELD_LAYOUT
        env = self._state(12)
        obs = encode_observation(env.game)
        perms = [list(range(len(o))) for o, _ in SYMMETRY_GROUPS]
        perms[0] = [2, 0, 3, 1]                               # level-1 market slots
        obs_index, _ = symmetry_indices(perms)
        moved = obs[obs_index]
        at = {key: i for i, key in enumerate(FIELD_LAYOUT)}
        for j, src in enumerate(perms[0]):
            for g in ('cost_white', 'card_short_owner', 'card_short_rival'):
                self.assertEqual(moved[at[(g, 'market', f'L1S{j + 1}')]], obs[at[(g, 'market', f'L1S{src + 1}')]])

    def _version1_state(self, model):
        """the state dict a version-1 network with the same weights would have: derived groups removed"""
        from splendor.rl_env import ATTRS, VOCAB_SIZES, DERIVED_FIELD_GROUPS
        state = {k: v.clone() for k, v in model.state_dict().items()}
        key = [k for k in state if k.endswith('env_embedding.field_embedding.weight')][0]
        akey = [k for k in state if k.endswith('env_embedding.attr_embedding.weight')][0]
        table, n_starters = state[key], state[key].shape[0] - sum(VOCAB_SIZES[a] for a in ATTRS)
        rows, offset, old_offsets, o = [table[:n_starters]], n_starters, dict(), n_starters
        for a in ATTRS:
            if a not in DERIVED_FIELD_GROUPS:
                rows.append(table[offset:offset + VOCAB_SIZES[a]])
                old_offsets[a] = o
                o += VOCAB_SIZES[a]
            offset += VOCAB_SIZES[a]
        state[key] = torch.cat(rows)
        attrs = list(ATTRS) + ['card_entity', 'noble_entity']
        state[akey] = state[akey][[i for i, a in enumerate(attrs) if a not in DERIVED_FIELD_GROUPS]]
        return state, key, akey, old_offsets, n_starters

    def test_version1_checkpoint_keeps_its_function(self):
        from splendor.rl_env import (ATTRS, VOCAB_SIZES, DERIVED_FIELD_GROUPS, FIELD_GROUPS, encode_observation,
                                     CARD_FIELD_GROUPS, NOBLE_FIELD_GROUPS)
        from rl.splendor_model import build_token_layout, convert_checkpoint_to_current_observation
        torch.manual_seed(3)
        model = SplendorAlphaGoZero(**TINY)
        state, key, akey, old_offsets, n_starters = self._version1_state(model)
        old_table = state[key]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'v1.pth')
            torch.save({'model': state}, path)
            torch.manual_seed(4)
            fresh = SplendorAlphaGoZero(**TINY)
            with self.assertRaises(ValueError):
                init_weights_from(fresh, path, strict=True)
            report = init_weights_from(fresh, path, strict=False)
            self.assertEqual(sorted(report['remapped']), sorted([key, akey]))
            self.assertEqual(report['mismatched'], [])
            converted = convert_checkpoint_to_current_observation(SplendorAlphaGoZero(**TINY), path, os.path.join(d, 'v2.pth'))
            conv_state = load_checkpoint(converted)['model']
            for k, v in fresh.state_dict().items():           # the conversion = the non-strict initialisation
                self.assertTrue(torch.equal(conv_state[k], v), k)
        for k, v in model.state_dict().items():               # everything but the embedding tables is untouched
            if k not in (key, akey):
                self.assertTrue(torch.equal(fresh.state_dict()[k], v), k)
        # function preserved: every token of the converted embedding equals the version-1 token (fields of the
        # version-1 layout only, pooled with the version-1 scale)
        layout = build_token_layout()
        env = self._state(13)
        obs = torch.as_tensor(encode_observation(env.game), dtype=torch.int64).unsqueeze(0)
        emb = fresh.model.env_embedding
        with torch.no_grad():
            got = emb(obs)[0, emb.num_starters:]
            position = emb.position_embedding()[emb.num_starters:]
            counts = dict()
            for f, t in enumerate(layout['field_token']):
                if FIELD_GROUPS[f] not in DERIVED_FIELD_GROUPS:
                    counts[t] = counts.get(t, 0) + 1
            want = torch.zeros_like(got)
            for f, t in enumerate(layout['field_token']):
                g = FIELD_GROUPS[f]
                if g not in DERIVED_FIELD_GROUPS:
                    want[t] += old_table[old_offsets[g] + int(obs[0, f])] / math.sqrt(counts[t])
            want += position
        self.assertTrue(torch.allclose(got, want, atol=1e-5), float((got - want).abs().max()))


@unittest.skipIf(torch is None, 'needs torch')
class OpponentPolicyTest(unittest.TestCase):
    def _walk(self, node, own, out):
        """(seat, visited action count) of every decision node below `node`"""
        n = np.asarray(node.stats()[0])
        out.append((node.seat, int((n > 0).sum())))
        for child in node.children.values():
            for c in (child.values() if isinstance(child, dict) else (child,)):
                self._walk(c, own, out)
        return out

    def test_opponents_do_not_branch(self):
        torch.manual_seed(11)
        model = SplendorAlphaGoZero(**TINY)
        env = SplendorRLEnv()
        env.reset(7)
        for opponent_policy in ('prior', 'bot:lookahead', 'bot:heuristic'):
            mcts = ts.SplendorMCTS(N_ACTIONS, ts.make_local_predictor(model), n_simulation=48,
                                   max_num_considered_actions=4, gumbel_scale=0., opponent_policy=opponent_policy)
            action, policy = mcts.simulate(env)
            self.assertTrue(np.frombuffer(env.legal_mask_bytes(), dtype=np.uint8)[action])
            self.assertAlmostEqual(float(policy.sum()), 1., places=6)
            nodes = self._walk(mcts.root, mcts.root.seat, [])
            others = [k for seat, k in nodes[1:] if seat != mcts.root.seat]
            self.assertTrue(others)
            self.assertTrue(all(k <= 1 for k in others), (opponent_policy, others))       # one move per opponent node
            self.assertTrue(any(seat == mcts.root.seat for seat, _ in nodes[1:]), opponent_policy)   # own moves reached
        max_n = ts.SplendorMCTS(N_ACTIONS, ts.make_local_predictor(model), n_simulation=48,
                                max_num_considered_actions=4, gumbel_scale=0.)
        max_n.simulate(env)
        self.assertTrue(any(k > 1 for seat, k in self._walk(max_n.root, max_n.root.seat, [])[1:] if seat != max_n.root.seat))
        with self.assertRaises(AssertionError):
            ts.SplendorMCTS(N_ACTIONS, None, opponent_policy='bot:nobody')

    def test_eval_game_with_opponent_model(self):
        torch.manual_seed(9)
        model = SplendorAlphaGoZero(**TINY)
        predictor = {ts.SLOT_NEW: ts.make_local_predictor(model)}
        seats = [ts.SLOT_NEW] + ['bot:heuristic'] * 3
        r = ts.play_game(4, 77, seats, predictor,
                         tiny_params('unused', max_game_turns=60, eval_mcts_opponent_policy='prior'), 'bench')
        self.assertIn(r['end_reason'], ('score', 'truncated', 'draw'))


@unittest.skipIf(torch is None, 'needs torch')
class HeuristicLeafTest(unittest.TestCase):
    def test_heuristic_win_shares(self):
        from rl.splendor_mcts import heuristic_win_shares
        env = SplendorRLEnv()
        env.reset(3)
        h = heuristic_win_shares(env.game, 1.)
        self.assertEqual(len(h), N_PLAYERS)
        self.assertAlmostEqual(sum(h), 1., places=9)
        flat = heuristic_win_shares(env.game, 1e6)
        self.assertTrue(all(abs(x - .25) < 1e-3 for x in flat))
        for _ in range(40):                                  # still win shares deep into a game
            env.step(int(np.flatnonzero(np.frombuffer(env.legal_mask_bytes(), dtype=np.uint8))[0]))
            if env.terminated:
                break
            self.assertAlmostEqual(sum(heuristic_win_shares(env.game, .5)), 1., places=9)

    def test_search_with_heuristic_leaves(self):
        torch.manual_seed(8)
        model = SplendorAlphaGoZero(**TINY)
        env = SplendorRLEnv()
        env.reset(5)
        for mu in (0., .5, 1.):
            mcts = ts.SplendorMCTS(N_ACTIONS, ts.make_local_predictor(model), n_simulation=16,
                                   max_num_considered_actions=4, heuristic_leaf_weight=mu, heuristic_temperature=2.)
            action, policy = mcts.simulate(env)
            self.assertTrue(np.frombuffer(env.legal_mask_bytes(), dtype=np.uint8)[action])
            self.assertAlmostEqual(float(policy.sum()), 1., places=6)
            self.assertLessEqual(mcts.root_search_value.sum(), 1. + 1e-6)

    def test_self_play_seats_with_several_bots(self):
        params = dict(self_play_bot_prob=1., self_play_bot_opponents=['bot:lookahead', 'bot:heuristic'],
                      self_play_bot_seat_counts=[1, 2])
        counts = set()
        for seed in range(200):
            seats = ts.self_play_seats(params, seed)
            bots = [s for s in seats if isinstance(s, str)]
            self.assertIn(len(bots), (1, 2))
            self.assertIn(ts.SLOT_CURRENT, seats)
            counts.add(len(bots))
            self.assertEqual(seats, ts.self_play_seats(params, seed))      # depends only on the seed
        self.assertEqual(counts, {1, 2})
        # without seat counts: the one-bot draws of R002 / R003
        one = dict(self_play_bot_prob=1., self_play_bot_opponents=['bot:lookahead'])
        self.assertTrue(all(sum(isinstance(s, str) for s in ts.self_play_seats(one, seed)) == 1 for seed in range(50)))

    def test_bench_game_uses_eval_weight(self):
        torch.manual_seed(9)
        model = SplendorAlphaGoZero(**TINY)
        predictor = {ts.SLOT_NEW: ts.make_local_predictor(model)}
        seats = [ts.SLOT_NEW] + ['bot:heuristic'] * 3
        r = ts.play_game(4, 77, seats, predictor,
                         tiny_params('unused', max_game_turns=60, eval_mcts_heuristic_leaf_weight=.5,
                                     mcts_heuristic_temperature=2., mcts_value_scale=.3,
                                     eval_mcts_value_scale=.1), 'bench')
        self.assertIn(r['end_reason'], ('score', 'truncated', 'draw'))


@unittest.skipIf(torch is None, 'needs torch')
class PolicyTeacherTest(unittest.TestCase):
    def _game(self, **overrides):
        torch.manual_seed(9)
        model = SplendorAlphaGoZero(**TINY)
        predictor = {ts.SLOT_CURRENT: ts.make_local_predictor(model)}
        seats = [ts.SLOT_CURRENT] * 3 + ['bot:heuristic']
        params = tiny_params('unused', mcts_full_search_prob=.5, max_game_turns=400, **overrides)
        return ts.play_game(4, 123, seats, predictor, params, 'train')

    def test_teacher_mixes_into_the_policy_target(self):
        from rl.splendor_data import check_sample
        plain = self._game()
        w = .25
        mixed = self._game(policy_teacher='bot:lookahead', policy_teacher_weight=w)
        a, b = plain['samples'], mixed['samples']
        np.testing.assert_array_equal(a['obs'], b['obs'])                         # the same game is played
        self.assertEqual(plain['teacher_samples'], 0)
        self.assertEqual(mixed['teacher_samples'], len(b['obs']))
        self.assertGreater(len(b['obs']), 2)
        np.testing.assert_allclose(b['probs'].sum(axis=1), 1., atol=1e-5)
        diff = b['probs'] - (1. - w) * a['probs']                                  # = w at the teacher's action
        for i in range(len(b['obs'])):
            t = int(np.argmax(diff[i]))
            self.assertAlmostEqual(float(diff[i, t]), w, places=5)
            self.assertTrue(b['mask'][i, t])                                       # a legal action
            np.testing.assert_allclose(np.delete(diff[i], t), 0., atol=1e-6)
            self.assertIsNone(check_sample(b['obs'][i], b['probs'][i], b['mask'][i], b['values'][i],
                                           float(b['value_weight'][i])))
        self.assertLessEqual(mixed['teacher_agree'], mixed['teacher_samples'])

    def test_teacher_off(self):
        plain = self._game()
        zero = self._game(policy_teacher='bot:lookahead', policy_teacher_weight=0.)
        np.testing.assert_array_equal(plain['samples']['probs'], zero['samples']['probs'])
        with self.assertRaises(ValueError):
            self._game(policy_teacher='lookahead', policy_teacher_weight=.2)
        with self.assertRaises(AssertionError):
            self._game(policy_teacher='bot:lookahead', policy_teacher_weight=1.5)


@unittest.skipIf(torch is None, 'needs torch')
class EvalScoreTest(unittest.TestCase):
    def test_non_score_games_are_draws_in_the_denominator(self):
        new, best = ts.SLOT_NEW, ts.SLOT_BEST
        results = [
            {'end_reason': 'score', 'rewards': [1., 0, 0, 0], 'seat_server_ids': [new, best, new, best]},   # new wins
            {'end_reason': 'score', 'rewards': [0, 1., 0, 0], 'seat_server_ids': [new, best, new, best]},   # best wins
            {'end_reason': 'draw', 'rewards': [.25] * 4, 'seat_server_ids': [new, best, new, best]},
            {'end_reason': 'truncated', 'rewards': [0.] * 4, 'seat_server_ids': [new, best, new, best]},
            {'failed': True, 'seat_server_ids': [new, best, new, best]},
        ]
        score = ts.eval_score(results, 6)       # one game missing
        self.assertAlmostEqual(score['win_rate'], (1 + 0 + .5 + .5 + .5 + .5) / 6)
        self.assertAlmostEqual(score['completion'], 3 / 6)
        self.assertEqual(score['counts'], {'score': 2, 'draw': 1, 'truncated': 1, 'failed': 1, 'missing': 1})


@unittest.skipIf(torch is None, 'needs torch')
class SeedTest(unittest.TestCase):
    def test_a_game_depends_only_on_its_seed(self):
        torch.manual_seed(3)
        model = SplendorAlphaGoZero(**TINY)
        params = tiny_params('unused', max_game_turns=30, mcts_full_search_prob=0.3)
        predictor = {0: ts.make_local_predictor(model)}
        a = ts.play_game(7, 12345, [0] * 4, predictor, params, 'train')
        np.random.seed(999)          # global RNG state must not matter
        b = ts.play_game(7, 12345, [0] * 4, predictor, params, 'train')
        c = ts.play_game(8, 54321, [0] * 4, predictor, params, 'train')
        self.assertEqual((a['turns'], a['decisions'], a['full_searches']), (b['turns'], b['decisions'], b['full_searches']))
        np.testing.assert_array_equal(a['samples']['obs'], b['samples']['obs'])
        np.testing.assert_array_equal(a['samples']['probs'], b['samples']['probs'])
        self.assertFalse(len(a['samples']['obs']) == len(c['samples']['obs'])
                         and np.array_equal(a['samples']['obs'], c['samples']['obs']))
        self.assertEqual(a['end_reason'], 'truncated')
        self.assertTrue((a['samples']['value_weight'] == 0).all())


@unittest.skipIf(torch is None, 'needs torch')
class ExperimentTest(unittest.TestCase):
    def test_new_experiment_then_exact_resume(self):
        with tempfile.TemporaryDirectory() as d:
            experiment = os.path.join(d, 'exp')
            params = tiny_params(experiment)
            ts.train_process(copy.deepcopy(params), 0, logging.WARNING)
            paths = ts.experiment_paths(experiment)
            for key in ('config', 'data', 'latest', 'best', 'log'):
                self.assertTrue(os.path.exists(paths[key]), key)
            first = load_checkpoint(paths['latest'])
            self.assertEqual(first['train_state']['iteration'], 2)
            self.assertEqual(first['versions']['rules'], 2)
            self.assertGreater(first['train_state']['train_step_num'], 0)
            # a new experiment never reuses the directory
            with self.assertRaises(RuntimeError):
                ts.train_process(copy.deepcopy(params), 0, logging.WARNING)
            # exact resume: same config file, one more iteration
            with open(paths['config'], encoding='UTF-8') as f:
                resume_params = yaml.safe_load(f)
            resume_params['num_iterations'] = 3
            resume_params['experiment_dir'] = experiment
            with open(paths['config'], 'w', encoding='UTF-8') as f:
                yaml.safe_dump(resume_params, f)
            with open(paths['data'], 'a') as f:
                f.write('garbage written after the checkpoint\n')
            ts.train_process(copy.deepcopy(resume_params), 0, logging.WARNING, resume=True)
            second = load_checkpoint(paths['latest'])
            self.assertEqual(second['train_state']['iteration'], 3)
            self.assertGreaterEqual(second['train_state']['next_game_id'], first['train_state']['next_game_id'])
            self.assertGreater(second['train_state']['train_step_num'], first['train_state']['train_step_num'])
            with open(paths['data']) as f:
                self.assertNotIn('garbage', f.read())
            # strict: another rules version is refused
            second['versions']['rules'] = 1
            save_checkpoint(paths['latest'], second)
            with self.assertRaises(ValueError):
                ts.train_process(copy.deepcopy(resume_params), 0, logging.WARNING, resume=True)


if __name__ == '__main__':
    unittest.main()
