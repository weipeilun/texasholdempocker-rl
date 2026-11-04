"""Training-side checks (need torch): model init, value masking, checkpoints, evaluation score, seeds, resume."""
import copy
import logging
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
    from rl.splendor_model import SplendorAlphaGoZero, init_weights_from, load_checkpoint, save_checkpoint
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
