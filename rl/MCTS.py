import math
import random
import logging
import numpy as np

import torch.nn
from tools import interrupt
from queue import Empty
from env.constants import *


class MCTS:
    def __init__(self, n_actions, is_root, player_name_predict_in_queue_dict, predict_out_queue, apply_dirichlet_noice, workflow_lock, workflow_signal_queue, workflow_ack_signal_queue, n_simulation=2000, c_puct=1., tau=1, dirichlet_noice_epsilon=0.25, pid=None, thread_name=None):
        self.n_actions = n_actions
        self.is_root = is_root
        self.player_name_predict_in_queue_dict = player_name_predict_in_queue_dict
        self.predict_out_queue = predict_out_queue
        self.apply_dirichlet_noice = apply_dirichlet_noice
        self.workflow_lock = workflow_lock
        self.workflow_signal_queue = workflow_signal_queue
        self.workflow_ack_signal_queue = workflow_ack_signal_queue
        self.n_simulation = n_simulation
        self.c_puct = c_puct
        self.tau = tau
        self.dirichlet_noice_epsilon = dirichlet_noice_epsilon  # 用来保证s0即根节点的探索性
        self.pid = pid
        self.thread_name = thread_name

        self.children = None

        num_bins_for_raise_call = self.n_actions - 2
        self.num_small_range_bins = math.ceil(num_bins_for_raise_call / 2)
        self.num_big_range_bins = num_bins_for_raise_call - self.num_small_range_bins
        self.big_range = 1 / (self.num_big_range_bins + 1)
        self.small_range = self.big_range / self.num_small_range_bins

        self.default_action_probs = np.ones(self.n_actions, dtype=np.float32) / self.n_actions
        self.default_player_result_value = 0.

        self.children_w_array = None
        self.children_n_array = None
        self.children_q_array = None

    def simulate(self, observation, env):
        self.children = list()
        for _ in range(self.n_actions):
            self.children.append(MCTS(self.n_actions, is_root=False, player_name_predict_in_queue_dict=self.player_name_predict_in_queue_dict, predict_out_queue=self.predict_out_queue, apply_dirichlet_noice=False, workflow_lock=self.workflow_lock, workflow_signal_queue=self.workflow_signal_queue, workflow_ack_signal_queue=self.workflow_ack_signal_queue, n_simulation=self.n_simulation, c_puct=self.c_puct, tau=self.tau, dirichlet_noice_epsilon=self.dirichlet_noice_epsilon, pid=self.pid, thread_name=self.thread_name))
        self.children_w_array = np.zeros(self.n_actions)
        self.children_n_array = np.zeros(self.n_actions)
        self.children_q_array = np.zeros(self.n_actions)

        acting_player_name = env._acting_player_name
        p_array, _ = self.predict(observation, acting_player_name)
        if self.is_root and self.apply_dirichlet_noice:
            dirichlet_noise = np.random.dirichlet(p_array)
            p_array = (1 - self.dirichlet_noice_epsilon) * p_array + self.dirichlet_noice_epsilon * dirichlet_noise

        for i in range(self.n_simulation):
            if interrupt.interrupt_callback():
                logging.info("MCTS.simulate detect interrupt")
                return None

            new_env = env.new_random()

            action_bin, action = self._choose_action(p_array)
            observation_, reward_dict, terminated, info = new_env.step(action)
            if not terminated:
                reward_dict = self.children[action_bin].expand(observation_, new_env)
            player_action_reward = self.get_player_action_reward(reward_dict, acting_player_name)

            self.children_w_array[action_bin] += player_action_reward
            self.children_n_array[action_bin] += 1
            self.children_q_array[action_bin] = self.children_w_array[action_bin] / self.children_n_array[action_bin]

        return self._cal_action_probs(self.tau)

    # πa ∝ pow(N(s, a), 1 / τ)
    def _cal_action_probs(self, tau):
        # tau <= 0是没有意义的，退化为不考虑tau参数
        if tau <= 0:
            pow_N_to_taus = self.children_n_array
        else:
            tau_array = np.ones(self.n_actions, dtype=np.float32) / tau
            pow_N_to_taus = np.power(self.children_n_array, tau_array)
        sum_N_to_taus = sum(pow_N_to_taus)
        action_probs = pow_N_to_taus / sum_N_to_taus
        return action_probs

    def get_action(self, action_probs, use_argmax=False):
        if use_argmax:
            action_idx = int(np.argmax(action_probs).tolist())
            return self.map_model_action_to_actual_action_and_value(action_idx)
        else:
            random_num = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(action_probs):
                cumulative_prob += prob
                if random_num <= cumulative_prob:
                    return self.map_model_action_to_actual_action_and_value(i)
            raise ValueError(f"action_probs should be a probability distribution, but={action_probs}")

    def get_player_action_reward(self, reward, acting_player_name):
        # 注意此处要取到的值是：对于玩家的相对价值得失
        # win/lose/draw(1/-1/0), win_value/value_game_start[0, 1]
        player_result_value, reward_value, net_win_value = reward[acting_player_name]
        return reward_value

    def _choose_action(self, p_array):
        sum_N = sum(self.children_n_array)
        # 注意此处原生的蒙特卡洛搜索树要求U(s, a) ∝ P(s, a)/(1 + N(s, a))
        # 此处增加了sqrt(sum_N)项作为分子，因此在第一次求解时需要满足U(s, a) ∝ P(s, a)
        if sum_N == 0:
            action_bin = int(np.argmax(p_array).tolist())
        else:
            # sqrt_sum_N_of_b_array = np.sqrt(sum_N - self.children_n_array)
            sqrt_sum_N_of_b_array = np.sqrt(sum_N)
            N_term_array = sqrt_sum_N_of_b_array / (1 + self.children_n_array)
            U_array = self.c_puct * p_array * N_term_array
            R_array = U_array + self.children_q_array

            action_bin = int(np.argmax(R_array).tolist())

        action, action_value = self.map_model_action_to_actual_action_and_value(action_bin)
        return action_bin, (action, action_value)

    def map_model_action_to_actual_action_and_value(self, action_bin):
        def generate_value_by_bin_number(bin_num):
            if bin_num < self.num_small_range_bins:
                bin_start = self.small_range * bin_num
                bin_thres = self.small_range
            else:
                big_bin_num = bin_num - self.num_small_range_bins + 1
                bin_start = self.big_range * big_bin_num
                bin_thres = self.big_range
            return random.random() * bin_thres + bin_start

        if action_bin == 0:
            return PlayerActions.FOLD.value, 0.
        elif action_bin == 1:
            return PlayerActions.CHECK.value, 0.
        elif 2 <= action_bin < self.n_actions:
            return PlayerActions.RAISE.value, generate_value_by_bin_number(action_bin - 2)
        else:
            raise ValueError(f"action should be in [0, {self.n_actions}), which is {action_bin}")

    def expand(self, observation, env):
        if self.children is None:
            self.children = list()
            for _ in range(self.n_actions):
                self.children.append(MCTS(self.n_actions, is_root=False, player_name_predict_in_queue_dict=self.player_name_predict_in_queue_dict, predict_out_queue=self.predict_out_queue, apply_dirichlet_noice=False, workflow_lock=self.workflow_lock, workflow_signal_queue=self.workflow_signal_queue, workflow_ack_signal_queue=self.workflow_ack_signal_queue, n_simulation=self.n_simulation, c_puct=self.c_puct, tau=self.tau, dirichlet_noice_epsilon=self.dirichlet_noice_epsilon, pid=self.pid, thread_name=self.thread_name))
            self.children_w_array = np.zeros(self.n_actions)
            self.children_n_array = np.zeros(self.n_actions)
            self.children_q_array = np.zeros(self.n_actions)

        acting_player_name = env._acting_player_name
        p_array, _ = self.predict(observation, acting_player_name)
        action_bin, action = self._choose_action(p_array)

        observation_, reward_dict, terminated, info = env.step(action)
        if not terminated:
            reward_dict = self.children[action_bin].expand(observation_, env)
        player_action_reward = self.get_player_action_reward(reward_dict, acting_player_name)

        self.children_w_array[action_bin] += player_action_reward
        self.children_n_array[action_bin] += 1
        self.children_q_array[action_bin] = self.children_w_array[action_bin] / self.children_n_array[action_bin]

        return reward_dict

    def predict(self, observation, acting_player_name):
        # 仅支持多线程下的批量预测
        p_array = self.default_action_probs
        player_result_value = self.default_player_result_value
        # 这个锁用于控制workflow的状态切换
        if self.workflow_lock is not None and self.workflow_signal_queue is not None and self.workflow_ack_signal_queue is not None:
            try:
                workflow_status = self.workflow_signal_queue.get(block=False)
                self.workflow_ack_signal_queue.put(workflow_status)

                with self.workflow_lock:
                    self.workflow_lock.wait()
            except Empty:
                pass

        self.player_name_predict_in_queue_dict[acting_player_name].put((observation, self.pid))
        while True:
            if interrupt.interrupt_callback():
                logging.info(f"MCTS.predict{self.pid} detect interrupt")
                break

            try:
                p_array, player_result_value = self.predict_out_queue.get(block=True, timeout=0.001)
                break
            except Empty:
                continue
        return p_array, player_result_value
