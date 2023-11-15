import logging
import os
from threading import Thread
from multiprocessing import Manager, Process
from rl.AlphaGoZero import AlphaGoZero
from env.workflow import *
from tools import counter
from queue import Queue


if __name__ == '__main__':
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 用户所有连续特征分桶embedding
    num_bins = 11
    # 模型输出值分桶数
    num_value_bins_for_raise_call = 19
    num_model_output_class = num_value_bins_for_raise_call + 2
    model = AlphaGoZero(n_observation=5,
                 n_actions=4,
                 num_output_class=num_model_output_class,
                 # device='cuda:0',
                 device='cpu',
                 embedding_dim=128,
                 positional_embedding_dim=64,
                 num_layers=4,
                 historical_action_sequence_length=16,
                 epsilon=0.05,
                 epsilon_max=0.8,
                 gamma=0.9,
                 batch_size=16,
                 learning_rate=3e-4,
                 num_bins=num_bins,
                 save_train_data=True
                 )
    model_path = 'models/deep_q_network.pth_cp'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logging.info(f'model loaded from {model_path}')

    # init generate winning probability calculating data processes
    num_gen_winning_prob_cal_data_processes = 4
    env_info_dict = dict()
    winning_probability_generating_task_queue = Manager().Queue()
    game_result_out_queue = Manager().Queue()
    for pid in range(num_gen_winning_prob_cal_data_processes):
        Process(target=simulate_processes, args=(winning_probability_generating_task_queue, game_result_out_queue, pid, log_level), daemon=True).start()

    # reward receiving thread
    Thread(target=receive_game_result_thread, args=(game_result_out_queue, env_info_dict), daemon=True).start()

    # game simulation thread
    # 保持线程数=batch_size * 3，可以保证瓶颈计算资源打满
    predict_batch_size = 1
    num_game_loop_thread = int(predict_batch_size * 2)
    num_mcts_simulation_per_step = 3000
    mcts_c_puct = 10.
    mcts_tau = 1.
    is_mcts_use_batch = False
    game_id_counter = counter.Counter()
    game_train_data_queue_dict = dict()
    game_finished_signal_queue = Queue()
    game_id_signal_queue = Queue()
    model_predict_batch_in_queue = Queue()
    model_predict_batch_out_queue_list = list()
    for tid in range(num_game_loop_thread):
        model_predict_batch_out_queue = Queue()
        model_predict_batch_out_queue_list.append(model_predict_batch_out_queue)
        Thread(target=game_loop_thread, args=(
        game_id_signal_queue, model, game_train_data_queue_dict, game_finished_signal_queue,
        winning_probability_generating_task_queue, model_predict_batch_in_queue, model_predict_batch_out_queue,
        num_bins, env_info_dict, num_mcts_simulation_per_step, is_mcts_use_batch, mcts_c_puct, mcts_tau, tid, tid), daemon=True).start()

    # control game simulation thread, to prevent excess task produced by producer
    game_finalized_signal_queue = Queue()
    Thread(target=train_game_control_thread, args=(game_id_signal_queue, game_finalized_signal_queue, num_game_loop_thread), daemon=True).start()

    finished_game_id = None
    while True:
        if interrupt.interrupt_callback():
            logging.info("main loop detect interrupt")
            break

        while True:
            try:
                if interrupt.interrupt_callback():
                    logging.info("main loop (game_finished_signal_queue) detect interrupt")
                    break

                finished_game_id = game_finished_signal_queue.get(block=True, timeout=1)
                break
            except Empty:
                continue

        game_train_data_queue = game_train_data_queue_dict.pop(finished_game_id)
        game_info_dict = env_info_dict[finished_game_id]

        is_need_to_break = False
        while not game_train_data_queue.empty():
            train_data_list, step_info = game_train_data_queue.get()

            round_num = step_info[KEY_ROUND_NUM]
            player_name = step_info[KEY_ACTED_PLAYER_NAME]
            while player_name not in game_info_dict:
                if interrupt.interrupt_callback():
                    logging.info("main loop (game_info_dict[player_name]) detect interrupt")
                    is_need_to_break = True
                    break
                time.sleep(1)
            if is_need_to_break:
                break

            player_round_info_dict = game_info_dict[player_name]
            while round_num not in player_round_info_dict:
                if interrupt.interrupt_callback():
                    logging.info("main loop (player_round_info_dict[round_num]) detect interrupt")
                    is_need_to_break = True
                    break
                time.sleep(1)
            if is_need_to_break:
                break

            observation, estimate_value_probs = train_data_list
            estimate_winning_prob = player_round_info_dict[round_num]

            model_value_probs, model_winning_prob = model.cal_reward([observation])

            estimate_value_probs_str = ','.join('%.2f' % value for value in estimate_value_probs)
            model_value_probs_str = ','.join('%.2f' % value for value in model_value_probs)
            logging.info(f'observation={observation}\nestimate_value_probs={estimate_value_probs_str}\nmodel_value_probs={model_value_probs_str}\nestimate_winning_prob={estimate_winning_prob}\nmodel_winning_prob={model_winning_prob}\n')

        # 清掉这局的数据cache
        del env_info_dict[finished_game_id]
        game_finalized_signal_queue.put(finished_game_id)

