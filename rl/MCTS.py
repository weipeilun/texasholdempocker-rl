import math
import time
import logging
import numpy as np
import torch
from env.constants import *
from utils.math_utils import *
from tools import interrupt
from queue import Empty
from utils.workflow_utils import map_action_bin_to_actual_action_and_value_v2


class MCTS:
    def __init__(self, n_actions, is_root, predict_in_queue, predict_out_queue, apply_dirichlet_noice, workflow_lock, workflow_signal_queue, workflow_ack_signal_queue, small_blind, n_simulation=2000, c_puct=1., tau=1, dirichlet_noice_epsilon=0.25, model_Q_epsilon=0.5, init_root_n_simulation=4, choice_method=ChoiceMethod.ARGMAX, log_to_file=False, pid=None, thread_name=None):
        self.n_actions = n_actions
        self.is_root = is_root
        self.predict_in_queue = predict_in_queue
        self.predict_out_queue = predict_out_queue
        self.apply_dirichlet_noice = apply_dirichlet_noice
        self.workflow_lock = workflow_lock
        self.workflow_signal_queue = workflow_signal_queue
        self.workflow_ack_signal_queue = workflow_ack_signal_queue
        self.small_blind = small_blind
        self.n_simulation = n_simulation
        self.c_puct = c_puct    # 模型预测的高值大概率是更好的，要更多的探索。这是模型的p值到env的Q值的乘数，注意将U和Q保持在相同数量级上。todo：动态调整c_puct
        self.tau = tau
        self.dirichlet_noice_epsilon = dirichlet_noice_epsilon  # 用来保证s0即根节点的探索性
        self.model_Q_epsilon = model_Q_epsilon  # 用来平衡模型的价值和env的价值
        self.choice_method = ChoiceMethod(choice_method)
        self.pid = pid
        self.thread_name = thread_name
        self.log_to_file = log_to_file

        self.file_writer_n = None
        self.file_writer_q = None
        self.file_writer_u = None
        self.file_writer_r = None
        self.file_writer_n_term = None
        self.file_writer_choice = None
        # 最后一步的R很重要，要拉倒很接近的数值上才能说明整个MCTS是有效的。因此强制输出最后一步的R到日志。
        self.file_writer_final_status = None

        self.children = None

        num_bins_for_raise_call = self.n_actions - 3
        self.num_small_range_bins = math.ceil(num_bins_for_raise_call / 2)
        self.num_big_range_bins = num_bins_for_raise_call - self.num_small_range_bins
        self.big_range = 1 / (self.num_big_range_bins + 1)
        self.small_range = self.big_range / self.num_small_range_bins

        self.init_root_n_simulation = init_root_n_simulation
        assert self.init_root_n_simulation >= 1, ValueError(f'init_root_n_simulation must be greater or equal than 1 to init root')

        self.default_action_probs = np.ones(self.n_actions, dtype=np.float32) / self.n_actions
        self.default_estimate_reward_value = np.zeros(1, dtype=np.float32)
        self.default_winning_prob = np.zeros(1, dtype=np.float32)

        self.children_w_array = None
        self.children_n_array = None
        self.children_q_array = None

    def new_child(self):
        return MCTS(self.n_actions,
                    is_root=False,
                    predict_in_queue=self.predict_in_queue,
                    predict_out_queue=self.predict_out_queue,
                    apply_dirichlet_noice=False,
                    workflow_lock=self.workflow_lock,
                    workflow_signal_queue=self.workflow_signal_queue,
                    workflow_ack_signal_queue=self.workflow_ack_signal_queue,
                    small_blind=self.small_blind,
                    n_simulation=self.n_simulation,
                    c_puct=self.c_puct,
                    tau=self.tau,
                    dirichlet_noice_epsilon=self.dirichlet_noice_epsilon,
                    model_Q_epsilon=self.model_Q_epsilon,
                    init_root_n_simulation=self.init_root_n_simulation,
                    choice_method=self.choice_method,
                    log_to_file=self.log_to_file,
                    pid=self.pid,
                    thread_name=self.thread_name)

    def simulate(self, observation, env):
        self.children = [None] * self.n_actions
        self.children_w_array = np.zeros(self.n_actions)
        self.children_n_array = np.zeros(self.n_actions)
        self.children_q_array = np.zeros(self.n_actions)

        self.file_writer_final_status = open("log/final_status.csv", "a", encoding='UTF-8')
        if self.log_to_file and self.pid == 0:
            self.file_writer_n = open(f"log/n.csv", "w", encoding='UTF-8')
            self.file_writer_q = open(f"log/q.csv", "w", encoding='UTF-8')
            self.file_writer_u = open(f"log/u.csv", "w", encoding='UTF-8')
            self.file_writer_r = open(f"log/r.csv", "w", encoding='UTF-8')
            self.file_writer_n_term = open(f"log/n_term.csv", "w", encoding='UTF-8')
            self.file_writer_choice = open(f"log/choice.csv", "w", encoding='UTF-8')

        acting_player_name = env.acting_player_name
        action_prob, estimate_reward_value, _ = self.predict(observation)
        if self.is_root and self.apply_dirichlet_noice:
            dirichlet_noise = np.random.dirichlet(action_prob)
            action_prob = (1 - self.dirichlet_noice_epsilon) * action_prob + self.dirichlet_noice_epsilon * dirichlet_noise
        self.file_writer_final_status.write(','.join('%.3f' % i for i in action_prob) + '\n')

        for i in range(self.n_simulation):
            if interrupt.interrupt_callback():
                logging.info("MCTS.simulate detect interrupt")
                return None, None

            new_env = env.new_random()

            action_bin, action = self._choose_action(action_prob, new_env, num_simulation=i, do_log=True)

            if self.file_writer_choice is not None:
                self.file_writer_choice.write(f'{i},{action_bin}\n')

            observation_, reward_dict, terminated, info = new_env.step(action)
            if not terminated:
                if self.children[action_bin] is None:
                    self.children[action_bin] = self.new_child()
                reward_dict = self.children[action_bin].expand(observation_, new_env)
            player_action_reward = self.get_player_action_reward(reward_dict, acting_player_name)

            # according to AlphaGo (<Mastering_the_game_of_Go_with_deep_neural_networks_and_tree_search>), passage '4 Searching with Policy and Value Networks': V(sL) = (1 −λ)vθ(sL) + λzL
            self.children_w_array[action_bin] += self.model_Q_epsilon * estimate_reward_value + (1 - self.model_Q_epsilon) * player_action_reward
            self.children_n_array[action_bin] += 1
            self.children_q_array[action_bin] = self.children_w_array[action_bin] / self.children_n_array[action_bin]

            if self.file_writer_n is not None:
                self.file_writer_n.write(','.join('%d' % i for i in self.children_n_array) + '\n')
            if self.file_writer_q is not None:
                self.file_writer_q.write(','.join('%.3f' % i for i in self.children_q_array) + '\n')

        self.file_writer_final_status.close()
        if self.log_to_file and self.pid == 0:
            self.file_writer_n.close()
            self.file_writer_q.close()
            self.file_writer_u.close()
            self.file_writer_r.close()
            self.file_writer_n_term.close()
            self.file_writer_choice.close()
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

    def get_action(self, action_probs, env, choice_method):
        action_mask_list, action_value_or_ranges_list, acting_player_value_left, current_round_acting_player_historical_value, delta_min_value_to_raise = env.get_valid_action_info_v2()
        # 目前get_action都是跟在simulate之后的，认为此处的归一化是多余步骤，但为get_action方法独立的正确性仍然保留归一化
        valid_action_probs = np.copy(action_probs)
        valid_action_probs[action_mask_list] = 0

        action_mask_int_list = [int(not mask) for mask in action_mask_list]

        choice_idx = choose_idx_by_array(valid_action_probs, choice_method)
        return map_action_bin_to_actual_action_and_value_v2(choice_idx, action_value_or_ranges_list, acting_player_value_left, current_round_acting_player_historical_value, delta_min_value_to_raise, self.small_blind), action_mask_int_list

    def get_player_action_reward(self, reward, acting_player_name):
        # 注意此处要取到的值是：对于玩家的相对价值得失
        # win/lose/draw(1/-1/0), win_value/value_game_start[0, 1]
        player_result_value, reward_value, net_win_value = reward[acting_player_name]
        return reward_value

    def _choose_action(self, p_array, env, num_simulation=None, do_log=False):
        action_mask_list, action_value_or_ranges_list, acting_player_value_left, current_round_acting_player_historical_value, delta_min_value_to_raise = env.get_valid_action_info_v2()
        num_valid_actions = len(action_mask_list) - sum(action_mask_list)
        if num_simulation is not None and num_simulation < (self.init_root_n_simulation * num_valid_actions):
            # init root node
            target_valid_bin = num_simulation % num_valid_actions
            action_bin = 0
            tmp_bin = 0
            for action_mask in action_mask_list:
                if not action_mask:
                    if tmp_bin == target_valid_bin:
                        break
                    else:
                        tmp_bin += 1
                action_bin += 1
        else:
            # 这是核心的蒙特卡洛搜索树算法
            # 注意此处原生的蒙特卡洛搜索树要求U(s, a) ∝ sqrt(ln(sum_N) / N(s, a))
            # 在没有遍历完子树的所有节点的情况下，随机选择一个没有遍历过的节点 todo:这点重新检查
            # AlphaGo Zero修改U(s, a) ∝ P(s, a) * sqrt(sum_N) / (1 + N(s, a))
            # 所以在没有遍历完子树的所有节点的情况下，要选择一个没遍历过的P(s, a)最大的节点
            # todo: 非完全信息博弈场景，每次模拟都面向一个新的随机隐藏信息，所以每次模拟的q都和历史模拟的q不一样。所以本质上这的MCTS在物理意义上是不成立的（exploitation-exploration过程中exploitation是不严格成立的）。但在应用中可能有效的假设在于：对历史隐藏信息统计的q在当前隐藏信息下仍然是有效的。
            sum_N = sum(self.children_n_array)
            if sum_N == 0:
                valid_p_array = np.copy(p_array)
                valid_p_array[action_mask_list] = 0
                action_bin = choose_idx_by_array(valid_p_array, self.choice_method)
            else:
                # sqrt_sum_N_of_b_array = np.sqrt(sum_N - self.children_n_array)
                sqrt_sum_N_of_b_array = np.sqrt(sum_N)
                N_term_array = sqrt_sum_N_of_b_array / (1 + self.children_n_array)
                # this p should be proportional to n, as an adjust factor for q if N is big enough
                # this p will lead to a inefficient MCTS if the whole process does work
                # powered_p_array = np.power(p_array, np.ones(self.n_actions, dtype=np.float32) * tau)
                # since poker's tree depth is much more shallow than chess, and has more flexibility action choice to chess, we delete p_array to use a default setting of MCTS, making it less act to the rule 'stick to the right moves'
                # U_array = self.c_puct * p_array * N_term_array
                U_array = self.c_puct * N_term_array
                R_array = U_array + self.children_q_array

                if do_log:
                    if self.file_writer_n_term is not None:
                        self.file_writer_n_term.write(','.join('%.3f' % i for i in N_term_array) + '\n')
                    if self.file_writer_u is not None:
                        self.file_writer_u.write(','.join('%.3f' % i for i in U_array) + '\n')
                    if self.file_writer_r is not None:
                        self.file_writer_r.write(','.join('%.3f' % i for i in R_array) + '\n')

                valid_R_array = np.copy(R_array)
                # + 0.01 to make a difference between invalid value and min value in log
                valid_R_array -= min(valid_R_array) - 0.01
                valid_R_array[action_mask_list] = 0

                # 如果R最大的是遍历过的桶，直接取
                # 如果R最大的是没遍历过的桶，取所有没遍历过的桶，选取原则遵循self.choice_method
                max_valid_R_bin = int(np.argmax(valid_R_array).tolist())
                if self.children_n_array[max_valid_R_bin] > 0:
                    action_bin = max_valid_R_bin
                else:
                    mask_bin_list = []
                    for idx, (children_n, action_mask) in enumerate(zip(self.children_n_array, action_mask_list)):
                        if children_n > 0 or action_mask:
                            mask_bin_list.append(idx)
                    valid_p_array = np.copy(p_array)
                    valid_p_array[mask_bin_list] = 0
                    action_bin = choose_idx_by_array(valid_p_array, self.choice_method)

                # force log to file
                if num_simulation is not None and num_simulation == self.n_simulation - 1:
                    self.file_writer_final_status.write(','.join('%.3f' % i for i in U_array) + '\n')
                    self.file_writer_final_status.write(','.join('%.3f' % i for i in valid_R_array) + '\n')
                    self.file_writer_final_status.write(','.join('%d' % i for i in self.children_n_array) + '\n\n')

        action, action_value = map_action_bin_to_actual_action_and_value_v2(action_bin, action_value_or_ranges_list, acting_player_value_left, current_round_acting_player_historical_value, delta_min_value_to_raise, self.small_blind)
        return action_bin, (action, action_value)

    def expand(self, observation, env):
        if self.children is None:
            self.children = [None] * self.n_actions
            self.children_w_array = np.zeros(self.n_actions)
            self.children_n_array = np.zeros(self.n_actions)
            self.children_q_array = np.zeros(self.n_actions)

        acting_player_name = env.acting_player_name
        action_prob, estimate_reward_value, _ = self.predict(observation)
        action_bin, action = self._choose_action(action_prob, env)

        observation_, reward_dict, terminated, info = env.step(action)
        if not terminated:
            if self.children[action_bin] is None:
                self.children[action_bin] = self.new_child()
            reward_dict = self.children[action_bin].expand(observation_, env)
        player_action_reward = self.get_player_action_reward(reward_dict, acting_player_name)

        self.children_w_array[action_bin] += self.model_Q_epsilon * estimate_reward_value + (1 - self.model_Q_epsilon) * player_action_reward
        self.children_n_array[action_bin] += 1
        self.children_q_array[action_bin] = self.children_w_array[action_bin] / self.children_n_array[action_bin]

        return reward_dict

    def predict(self, observation):
        # 仅支持多线程下的批量预测
        action_prob = self.default_action_probs
        estimate_reward_value = self.default_estimate_reward_value
        winning_prob = self.default_winning_prob
        # 这个锁用于控制workflow的状态切换
        if self.workflow_lock is not None and self.workflow_signal_queue is not None and self.workflow_ack_signal_queue is not None:
            try:
                workflow_status = self.workflow_signal_queue.get(block=False)
                logging.info(f"MCTS.predict{self.pid} received workflow_status: {workflow_status.name}")
                self.workflow_ack_signal_queue.put(workflow_status)

                with self.workflow_lock:
                    self.workflow_lock.wait()
            except Empty:
                pass

        self.predict_in_queue.put((self.pid, np.asarray(observation)))
        begin_time = time.time()
        log_interval = 10
        next_log_time = begin_time + log_interval
        while True:
            if interrupt.interrupt_callback():
                logging.info(f"MCTS.predict{self.pid} detect interrupt")
                break

            if time.time() > next_log_time:
                now = time.time()
                next_log_time = now + log_interval
                logging.warning(f"MCTS.predict{self.pid} waited predict_out_queue for %.2fs" % (now - begin_time))

            try:
                action_prob, estimate_reward_value, winning_prob = self.predict_out_queue.get(block=True, timeout=0.01)
                break
            except Empty:
                continue
        return action_prob, estimate_reward_value, winning_prob


class SingleThreadMCTS(MCTS):
    def __init__(self, n_actions, is_root, apply_dirichlet_noice, small_blind, model, n_simulation=2000, c_puct=1.,
                 tau=1, dirichlet_noice_epsilon=0.25, model_Q_epsilon=0.5, init_root_n_simulation=4, choice_method=ChoiceMethod.ARGMAX, log_to_file=False, pid=None):
        self.n_actions = n_actions
        self.is_root = is_root
        self.apply_dirichlet_noice = apply_dirichlet_noice
        self.small_blind = small_blind
        self.model = model
        self.n_simulation = n_simulation
        self.c_puct = c_puct  # 模型预测的高值大概率是更好的，要更多的探索。这是模型的p值到env的Q值的乘数，注意将U和Q保持在相同数量级上。todo：动态调整c_puct
        self.tau = tau
        self.dirichlet_noice_epsilon = dirichlet_noice_epsilon  # 用来保证s0即根节点的探索性
        self.model_Q_epsilon = model_Q_epsilon  # 用来平衡模型的价值和env的价值
        self.choice_method = choice_method
        self.log_to_file = log_to_file
        self.pid = pid

        super().__init__(n_actions=self.n_actions,
                         is_root=self.is_root,
                         predict_in_queue=None,
                         predict_out_queue=None,
                         apply_dirichlet_noice=self.apply_dirichlet_noice,
                         workflow_lock=None,
                         workflow_signal_queue=None,
                         workflow_ack_signal_queue=None,
                         small_blind=self.small_blind,
                         n_simulation=self.n_simulation,
                         c_puct=self.c_puct,
                         tau=self.tau,
                         dirichlet_noice_epsilon=self.dirichlet_noice_epsilon,
                         model_Q_epsilon=self.model_Q_epsilon,
                         init_root_n_simulation=init_root_n_simulation,
                         choice_method=choice_method,
                         log_to_file=log_to_file,
                         pid=self.pid,
                         thread_name=None
                         )

    def new_child(self):
        return SingleThreadMCTS(n_actions=self.n_actions,
                                is_root=self.is_root,
                                apply_dirichlet_noice=self.apply_dirichlet_noice,
                                small_blind=self.small_blind,
                                model=self.model,
                                n_simulation=self.n_simulation,
                                c_puct=self.c_puct,
                                tau=self.tau,
                                dirichlet_noice_epsilon=self.dirichlet_noice_epsilon,
                                model_Q_epsilon=self.model_Q_epsilon,
                                choice_method=self.choice_method,
                                log_to_file=self.log_to_file,
                                pid=self.pid,
                                )

    def predict(self, observation):
        # 单线程预测
        with torch.no_grad():
            observation_array = np.array([observation])
            observation_tensor = torch.tensor(observation_array, dtype=torch.int32, device=self.model.device, requires_grad=False)
            action_probs_logits_tensor, estimate_reward_value_tensor, winning_prob_tensor = self.model(observation_tensor)
            action_probs_tensor = torch.softmax(action_probs_logits_tensor, dim=1)
            action_prob = action_probs_tensor.cpu().numpy()[0]
            estimate_reward_value = estimate_reward_value_tensor.cpu().numpy()[0]
            winning_prob = winning_prob_tensor.cpu().numpy()[0]
        return action_prob, estimate_reward_value, winning_prob
