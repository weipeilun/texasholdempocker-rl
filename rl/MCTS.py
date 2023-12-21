import math
import random
import logging
import numpy as np

import torch.nn
from tools import interrupt
from queue import Empty
from env.constants import *


class MCTS:
    def __init__(self, n_actions, is_root, player_name_predict_in_queue_dict, predict_out_queue, apply_dirichlet_noice, workflow_lock, workflow_signal_queue, workflow_ack_signal_queue, n_simulation=2000, c_puct=1., tau=1, dirichlet_noice_epsilon=0.25, model_Q_epsilon=0.5, init_root_n_simulation=10, log_to_file=False, pid=None, thread_name=None):
        self.n_actions = n_actions
        self.is_root = is_root
        self.player_name_predict_in_queue_dict = player_name_predict_in_queue_dict
        self.predict_out_queue = predict_out_queue
        self.apply_dirichlet_noice = apply_dirichlet_noice
        self.workflow_lock = workflow_lock
        self.workflow_signal_queue = workflow_signal_queue
        self.workflow_ack_signal_queue = workflow_ack_signal_queue
        self.n_simulation = n_simulation
        self.c_puct = c_puct    # 模型预测的高值大概率是更好的，要更多的探索。这是模型的p值到env的Q值的乘数，注意将U和Q保持在相同数量级上。todo：动态调整c_puct
        self.tau = tau
        self.dirichlet_noice_epsilon = dirichlet_noice_epsilon  # 用来保证s0即根节点的探索性
        self.model_Q_epsilon = model_Q_epsilon  # 用来平衡模型的价值和env的价值
        self.pid = pid
        self.thread_name = thread_name
        self.log_to_file = log_to_file

        self.file_writer_n = None
        self.file_writer_q = None
        self.file_writer_u = None
        self.file_writer_r = None
        self.file_writer_n_term = None
        self.file_writer_choice = None

        self.children = None

        num_bins_for_raise_call = self.n_actions - 3
        self.num_small_range_bins = math.ceil(num_bins_for_raise_call / 2)
        self.num_big_range_bins = num_bins_for_raise_call - self.num_small_range_bins
        self.big_range = 1 / (self.num_big_range_bins + 1)
        self.small_range = self.big_range / self.num_small_range_bins

        self.init_root_n_simulation = init_root_n_simulation
        self.max_init_root_num_simulation = int(self.init_root_n_simulation) * self.n_actions

        self.default_action_probs = np.ones(self.n_actions, dtype=np.float32) / self.n_actions
        self.default_action_Qs = np.zeros(self.n_actions, dtype=np.float32)

        self.children_w_array = None
        self.children_n_array = None
        self.children_q_array = None

    def simulate(self, observation, env):
        self.children = [None] * self.n_actions
        self.children_w_array = np.zeros(self.n_actions)
        self.children_n_array = np.zeros(self.n_actions)
        self.children_q_array = np.zeros(self.n_actions)

        if self.log_to_file and self.pid == 0:
            self.file_writer_n = open(f"log/n.csv", "w", encoding='UTF-8')
            self.file_writer_q = open(f"log/q.csv", "w", encoding='UTF-8')
            self.file_writer_u = open(f"log/u.csv", "w", encoding='UTF-8')
            self.file_writer_r = open(f"log/r.csv", "w", encoding='UTF-8')
            self.file_writer_n_term = open(f"log/n_term.csv", "w", encoding='UTF-8')
            self.file_writer_choice = open(f"log/choice.csv", "w", encoding='UTF-8')

        acting_player_name = env._acting_player_name
        action_prob, action_Qs = self.predict(observation, acting_player_name)
        if self.is_root and self.apply_dirichlet_noice:
            dirichlet_noise = np.random.dirichlet(action_prob)
            action_prob = (1 - self.dirichlet_noice_epsilon) * action_prob + self.dirichlet_noice_epsilon * dirichlet_noise

        for i in range(self.n_simulation):
            if interrupt.interrupt_callback():
                logging.info("MCTS.simulate detect interrupt")
                return None, None

            new_env = env.new_random()

            action_bin, action = self._choose_action(action_prob, num_simulation=i, do_log=True)
            if self.file_writer_choice is not None:
                self.file_writer_choice.write(f'{i},{action_bin}\n')
            observation_, reward_dict, terminated, info = new_env.step(action)
            if not terminated:
                if self.children[action_bin] is None:
                    self.children[action_bin] = MCTS(self.n_actions,
                                                     is_root=False,
                                                     player_name_predict_in_queue_dict=self.player_name_predict_in_queue_dict,
                                                     predict_out_queue=self.predict_out_queue,
                                                     apply_dirichlet_noice=False,
                                                     workflow_lock=self.workflow_lock,
                                                     workflow_signal_queue=self.workflow_signal_queue,
                                                     workflow_ack_signal_queue=self.workflow_ack_signal_queue,
                                                     n_simulation=self.n_simulation,
                                                     c_puct=self.c_puct,
                                                     tau=self.tau,
                                                     dirichlet_noice_epsilon=self.dirichlet_noice_epsilon,
                                                     model_Q_epsilon=self.model_Q_epsilon,
                                                     init_root_n_simulation=self.init_root_n_simulation,
                                                     log_to_file=self.log_to_file,
                                                     pid=self.pid,
                                                     thread_name=self.thread_name)
                reward_dict = self.children[action_bin].expand(observation_, new_env)
            player_action_reward = self.get_player_action_reward(reward_dict, acting_player_name)

            self.children_w_array[action_bin] += self.model_Q_epsilon * action_Qs[action_bin] + (1 - self.model_Q_epsilon) * player_action_reward
            self.children_n_array[action_bin] += 1
            self.children_q_array[action_bin] = self.children_w_array[action_bin] / self.children_n_array[action_bin]

            if self.file_writer_n is not None:
                self.file_writer_n.write(','.join('%d' % i for i in self.children_n_array) + '\n')
            if self.file_writer_q is not None:
                self.file_writer_q.write(','.join('%.3f' % i for i in self.children_q_array) + '\n')
        if self.log_to_file and self.pid == 0:
            self.file_writer_n.close()
            self.file_writer_q.close()
            self.file_writer_u.close()
            self.file_writer_r.close()
            self.file_writer_n_term.close()
            self.file_writer_choice.close()
        return self._cal_action_probs(self.tau), self.children_q_array

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

    def _choose_action(self, p_array, num_simulation=None, do_log=False):
        if num_simulation is not None and num_simulation < self.max_init_root_num_simulation:
            # init root node
            action_bin = num_simulation % self.n_actions
        else:
            # 这是核心的蒙特卡洛搜索树算法
            # 注意此处原生的蒙特卡洛搜索树要求U(s, a) ∝ sqrt(ln(sum_N) / N(s, a))
            # 在没有遍历完子树的所有节点的情况下，随机选择一个没有遍历过的节点
            # AlphaGo Zero修改U(s, a) ∝ P(s, a) * sqrt(sum_N) / (1 + N(s, a))
            # 所以在没有遍历完子树的所有节点的情况下，要选择一个没遍历过的P(s, a)最大的节点
            # todo: 非完全信息博弈场景，每次模拟都面向一个新的随机隐藏信息，所以每次模拟的q都和历史模拟的q不一样。所以本质上这的MCTS在物理意义上是不成立的（exploitation-exploration过程中exploitation是不严格成立的）。但在应用中可能有效的假设在于：对历史隐藏信息统计的q在当前隐藏信息下仍然是有效的。
            sum_N = sum(self.children_n_array)
            if sum_N < self.n_actions:
                action_bin = -1
                tmp_max_p = -1
                for i, p in enumerate(p_array):
                    if self.children_n_array[i] == 0:
                        if action_bin == -1 or p > tmp_max_p:
                            action_bin = i
                            tmp_max_p = p
            else:
                # sqrt_sum_N_of_b_array = np.sqrt(sum_N - self.children_n_array)
                sqrt_sum_N_of_b_array = np.sqrt(sum_N)
                N_term_array = sqrt_sum_N_of_b_array / (1 + self.children_n_array)
                # this p should be proportional to n, as an adjust factor for q if N is big enough
                # this p will lead to a inefficient MCTS if the whole process does work
                # powered_p_array = np.power(p_array, np.ones(self.n_actions, dtype=np.float32) * tau)
                U_array = self.c_puct * p_array * N_term_array
                R_array = U_array + self.children_q_array

                if do_log:
                    if self.file_writer_n_term is not None:
                        self.file_writer_n_term.write(','.join('%.3f' % i for i in N_term_array) + '\n')
                    if self.file_writer_u is not None:
                        self.file_writer_u.write(','.join('%.3f' % i for i in U_array) + '\n')
                    if self.file_writer_r is not None:
                        self.file_writer_r.write(','.join('%.3f' % i for i in R_array) + '\n')

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
        elif 2 <= action_bin < self.n_actions - 1:
            return PlayerActions.RAISE.value, generate_value_by_bin_number(action_bin - 2)
        elif action_bin == self.n_actions - 1:
            return PlayerActions.RAISE.value, 1.
        else:
            raise ValueError(f"action should be in [0, {self.n_actions}), which is {action_bin}")

    def expand(self, observation, env):
        if self.children is None:
            self.children = [None] * self.n_actions
            self.children_w_array = np.zeros(self.n_actions)
            self.children_n_array = np.zeros(self.n_actions)
            self.children_q_array = np.zeros(self.n_actions)

        acting_player_name = env._acting_player_name
        action_prob, action_Qs = self.predict(observation, acting_player_name)
        action_bin, action = self._choose_action(action_prob)

        observation_, reward_dict, terminated, info = env.step(action)
        if not terminated:
            if self.children[action_bin] is None:
                self.children[action_bin] = MCTS(self.n_actions,
                                                 is_root=False,
                                                 player_name_predict_in_queue_dict=self.player_name_predict_in_queue_dict,
                                                 predict_out_queue=self.predict_out_queue,
                                                 apply_dirichlet_noice=False,
                                                 workflow_lock=self.workflow_lock,
                                                 workflow_signal_queue=self.workflow_signal_queue,
                                                 workflow_ack_signal_queue=self.workflow_ack_signal_queue,
                                                 n_simulation=self.n_simulation,
                                                 c_puct=self.c_puct,
                                                 tau=self.tau,
                                                 dirichlet_noice_epsilon=self.dirichlet_noice_epsilon,
                                                 model_Q_epsilon=self.model_Q_epsilon,
                                                 init_root_n_simulation=self.init_root_n_simulation,
                                                 log_to_file=self.log_to_file,
                                                 pid=self.pid,
                                                 thread_name=self.thread_name)
            reward_dict = self.children[action_bin].expand(observation_, env)
        player_action_reward = self.get_player_action_reward(reward_dict, acting_player_name)

        self.children_w_array[action_bin] += self.model_Q_epsilon * action_Qs[action_bin] + (1 - self.model_Q_epsilon) * player_action_reward
        self.children_n_array[action_bin] += 1
        self.children_q_array[action_bin] = self.children_w_array[action_bin] / self.children_n_array[action_bin]

        return reward_dict

    def predict(self, observation, acting_player_name):
        # 仅支持多线程下的批量预测
        action_prob = self.default_action_probs
        action_Qs = self.default_action_Qs
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
                action_prob, action_Qs = self.predict_out_queue.get(block=True, timeout=0.001)
                break
            except Empty:
                continue
        return action_prob, action_Qs
