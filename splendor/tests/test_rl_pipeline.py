"""RL-side checks that need no torch: environment at truncation, search bootstrap, rollout, data file, seeds."""
import os
import random
import tempfile
import unittest

import numpy as np

from rl.splendor_data import (DATA_HEADER_PREFIX, TrainDataWriter, check_sample, derive_seed, format_sample,
                              parse_sample, read_samples, truncate_data_file)
from rl.splendor_mcts import SplendorMCTS
from splendor.engine import Action
from splendor.rl_env import N_ACTIONS, N_OBSERVATION, N_PLAYERS, SplendorRLEnv, to_absolute, version_info

RELATIVE_VALUE = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


def constant_predictor(request):
    """uniform policy over the legal actions, the same relative values everywhere"""
    mask = np.frombuffer(request[N_OBSERVATION:], dtype=np.uint8).astype(np.float32)
    return (mask / mask.sum()).astype(np.float32).tobytes() + RELATIVE_VALUE.tobytes()


def valid_sample(seed=0, weight=1.):
    rng = np.random.default_rng(seed)
    env = SplendorRLEnv()
    env.reset(seed)
    obs = np.frombuffer(env.observation_bytes(), dtype=np.uint8).astype(np.int32)
    mask = env.legal_mask()
    probs = np.where(mask, rng.random(N_ACTIONS), 0.).astype(np.float32)
    probs /= probs.sum()
    values = np.asarray([0., 1., 0., 0.], dtype=np.float32) if weight else np.zeros(N_PLAYERS, dtype=np.float32)
    return obs, probs, mask, values, weight


class TruncationTest(unittest.TestCase):
    def test_truncated_state_keeps_legal_actions_and_has_no_value_target(self):
        env = SplendorRLEnv(max_turns=1)
        env.reset(3)
        env.step(env.legal_indices()[0])
        self.assertTrue(env.terminated and env.truncated)
        self.assertEqual(env.end_reason, 'truncated')
        self.assertTrue(env.legal_indices())          # the network can still evaluate it
        rewards, weight = env.value_target()
        self.assertEqual(weight, 0.)
        self.assertFalse(rewards.any())

    def test_game_ended_by_the_rules_has_a_value_target(self):
        env = SplendorRLEnv()
        env.reset(5)
        g = env.game
        g.end_reason, g.winners, g.phase = 'draw', [0, 1, 2, 3], 'terminal'
        rewards, weight = env.value_target()
        self.assertEqual(weight, 1.)
        np.testing.assert_allclose(rewards, [.25] * 4)
        self.assertEqual(env.legal_indices(), [])

    def test_search_bootstraps_truncated_leaves_with_the_network(self):
        env = SplendorRLEnv(max_turns=1)        # every root action ends the turn and truncates the game
        env.reset(11)
        mcts = SplendorMCTS(N_ACTIONS, constant_predictor, n_simulation=32, max_num_considered_actions=8,
                            seed=1, node_impl='python')
        mcts.simulate(env)
        visited = mcts.root_visits > 0
        self.assertTrue(visited.any())
        # child seat = root seat + 1: the network's value of the child, seen from the root player
        expected = to_absolute(RELATIVE_VALUE, (env.acting_player + 1) % 4)[env.acting_player]
        np.testing.assert_allclose(mcts.root_q[visited], expected, rtol=1e-6)

    def test_rollout_hitting_max_turns_falls_back_to_the_network(self):
        env = SplendorRLEnv(max_turns=2)
        env.reset(13)
        mcts = SplendorMCTS(N_ACTIONS, constant_predictor, n_simulation=16, max_num_considered_actions=8,
                            rollout_lambda=1., seed=2, node_impl='python')
        self.assertIsNone(mcts._rollout(env.copy_for_search()))
        mcts.simulate(env)   # leaf values come from v_θ, no exception


class RolloutDiscardTest(unittest.TestCase):
    def test_rollout_plays_the_discard_phase(self):
        env = SplendorRLEnv()
        env.reset(21)
        g = env.game
        p = g.players[0]
        for color, n in enumerate((2, 2, 2, 2, 1, 0)):   # 9 tokens, taking three enters the discard phase
            g.bank[color] -= n
            p.tokens[color] = n
        g.step(Action('take', colors=(0, 1, 2)), random.Random(0))
        self.assertEqual(g.phase, 'discard')
        mcts = SplendorMCTS(N_ACTIONS, constant_predictor, rollout_lambda=1., seed=3, node_impl='python')
        for seed in range(5):
            mcts.rng = random.Random(seed)
            outcome = mcts._rollout(env.copy_for_search())
            self.assertIsNotNone(outcome)
            self.assertAlmostEqual(sum(outcome), 1.)


class DataFileTest(unittest.TestCase):
    def test_format_parse_round_trip(self):
        sample = valid_sample(1)
        parsed = parse_sample(format_sample(*sample))
        self.assertIsNone(check_sample(*parsed))
        np.testing.assert_array_equal(parsed[0], sample[0])
        np.testing.assert_allclose(parsed[1], sample[1], atol=6e-6)
        np.testing.assert_array_equal(parsed[2], sample[2])
        np.testing.assert_allclose(parsed[3], sample[3])
        self.assertEqual(parsed[4], 1.)

    def test_invalid_samples_are_rejected(self):
        obs, probs, mask, values, weight = valid_sample(2)
        self.assertIsNone(check_sample(obs, probs, mask, values, weight))
        bad_obs = obs.copy(); bad_obs[0] = 99
        self.assertIn('vocab', check_sample(bad_obs, probs, mask, values, weight))
        bad_probs = probs.copy(); bad_probs[np.flatnonzero(~mask)[0]] = 0.5
        self.assertIsNotNone(check_sample(obs, bad_probs, mask, values, weight))
        self.assertIsNotNone(check_sample(obs, probs, mask, np.asarray([.5, .2, 0, 0], np.float32), 1.))
        self.assertIsNotNone(check_sample(obs, probs, mask, values, 0.))       # weight 0 needs zero values
        self.assertIsNone(check_sample(*valid_sample(3, weight=0.)))
        self.assertIsNotNone(check_sample(obs[:-1], probs, mask, values, weight))

    def test_header_versions_are_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'data.txt')
            writer = TrainDataWriter(path)
            self.assertTrue(writer.write(*valid_sample(4)))
            self.assertFalse(writer.write(*valid_sample(4)[:3], np.asarray([2., 0, 0, 0], np.float32), 1.))
            writer.close()
            samples, stats = read_samples(path)
            self.assertEqual((len(samples), stats['invalid']), (1, 0))
            with self.assertRaises(ValueError):
                TrainDataWriter(path, discard_over_limit=False)     # another rule variant
            with open(path, 'r') as f:
                lines = f.readlines()
            with open(path, 'w') as f:
                f.write(lines[0].replace(f'"rules": {version_info()["rules"]}', '"rules": 1'))
                f.writelines(lines[1:])
            with self.assertRaises(ValueError):
                read_samples(path)
            old = os.path.join(d, 'old.txt')
            with open(old, 'w') as f:
                f.write('1,2,3\n0.5,0.5\n1,1\n1,0,0,0\n\n')
            with self.assertRaises(ValueError):
                read_samples(old)

    def test_resume_point_partial_line_and_last(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'data.txt')
            writer = TrainDataWriter(path)
            for k in range(5):
                writer.write(*valid_sample(k))
            size = writer.flush()
            for k in range(5, 8):
                writer.write(*valid_sample(k))
            writer.close()
            self.assertEqual(len(read_samples(path)[0]), 8)
            self.assertEqual(len(read_samples(path, max_bytes=size)[0]), 5)
            self.assertEqual(len(read_samples(path, last=3)[0]), 3)
            with open(path, 'a') as f:
                f.write(format_sample(*valid_sample(9))[:40])   # killed while writing
            self.assertEqual(len(read_samples(path)[0]), 8)
            self.assertTrue(truncate_data_file(path, size))
            self.assertEqual(len(read_samples(path)[0]), 5)
            with open(path) as f:
                self.assertTrue(f.readline().startswith(DATA_HEADER_PREFIX))


class TailReadTest(unittest.TestCase):
    def test_tail_only_reads_the_same_last_samples(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'data.txt')
            writer = TrainDataWriter(path)
            for k in range(40):
                writer.write(*valid_sample(k))
            writer.close()
            with open(path, 'ab') as f:
                f.write(b'1,2,3')            # a partly written last line
            for last in (1, 7, 40, 100):
                full, _ = read_samples(path, last=last)
                tail, stats = read_samples(path, last=last, tail_only=True)
                self.assertEqual(len(full), len(tail))
                self.assertEqual(stats['read'], len(tail))
                for a, b in zip(full, tail):
                    for x, y in zip(a, b):
                        np.testing.assert_array_equal(x, y)


class SeedTest(unittest.TestCase):
    def test_derived_seeds_are_deterministic_and_distinct(self):
        self.assertEqual(derive_seed(1, 2, 3), derive_seed(1, 2, 3))
        seeds = {derive_seed(20261105, 13, game, decision) for game in range(50) for decision in range(50)}
        self.assertEqual(len(seeds), 2500)
        self.assertNotEqual(derive_seed(1, 2), derive_seed(2, 1))
        self.assertLess(derive_seed(7), 2 ** 64)


if __name__ == '__main__':
    unittest.main()
