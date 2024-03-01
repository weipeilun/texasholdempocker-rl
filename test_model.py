from env.workflow import *
from tools.param_parser import *
from tools import counter
from queue import Queue
from torch.multiprocessing import Manager, Process, Condition


if __name__ == '__main__':
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    args, params = parse_params()

    torch.multiprocessing.set_start_method('spawn')

    num_predict_batch_process = params['num_predict_batch_process']
    update_model_param_queue_list = [Manager().Queue() for _ in range(num_predict_batch_process)]
    workflow_queue_list = [Manager().Queue() for _ in range(num_predict_batch_process)]
    workflow_ack_queue_list = [Manager().Queue() for _ in range(num_predict_batch_process)]

    # 游戏设置
    small_blind = params['small_blind']
    big_blind = params['big_blind']
    update_model_bb_per_100_thres = params['update_model_bb_per_100_thres']

    # 用户所有连续特征分桶embedding
    num_bins = params['num_action_bins']
    model_param_dict = params['model_param_dict']
    model_init_checkpoint_path = params['model_init_checkpoint_path']
    model = AlphaGoZero(**model_param_dict)

    # init generate winning probability calculating data processes
    num_gen_winning_prob_cal_data_processes = params['num_gen_winning_prob_cal_data_processes']
    simulation_recurrent_param_dict = params['simulation_recurrent_params']
    env_info_dict = dict()
    winning_probability_generating_task_queue = Manager().Queue()
    game_result_out_queue = Manager().Queue()
    for pid in range(num_gen_winning_prob_cal_data_processes):
        Process(target=simulate_processes, args=(winning_probability_generating_task_queue, game_result_out_queue, simulation_recurrent_param_dict, pid, log_level), daemon=True).start()

    # reward receiving thread
    Thread(target=receive_game_result_thread, args=(game_result_out_queue, env_info_dict), daemon=True).start()

    # game simulation threads (train and evaluate)
    # 必须保证线程数 > batch_size * 2，以在train中gpu基本打满，在eval中不死锁
    predict_batch_size = params['predict_batch_size']
    num_game_loop_thread = int(predict_batch_size * params['game_loop_thread_multiple'])
    assert num_game_loop_thread % 2 == 0, 'num_game_loop_thread must be even'
    # train_eval进程数
    num_train_eval_process = params['num_train_eval_process']
    num_train_eval_thread = num_game_loop_thread * num_predict_batch_process
    assert num_train_eval_thread % num_train_eval_process == 0, 'num_game_loop_thread must be multiple of num_train_eval_process'
    num_game_loop_thread_per_process = num_train_eval_thread // num_train_eval_process
    # game params
    num_mcts_simulation_per_step = params['num_mcts_simulation_per_step']
    mcts_c_puct = params['mcts_c_puct']
    mcts_tau = params['mcts_tau']
    mcts_dirichlet_noice_epsilon = params['mcts_dirichlet_noice_epsilon']
    mcts_model_Q_epsilon = params['mcts_model_Q_epsilon']
    mcts_log_to_file = params['mcts_log_to_file']
    mcts_choice_method = params['mcts_choice_method']
    game_id_counter = counter.Counter()
    workflow_lock = Condition()
    workflow_game_loop_signal_queue_list = list()
    workflow_game_loop_ack_signal_queue_list = list()
    # train thread参数
    game_train_data_queue = Manager().Queue()
    train_game_finished_signal_queue = Manager().Queue()
    train_game_id_signal_queue = Manager().Queue()
    # in_queue, (out_queue_list, out_queue_map_dict_train, out_queue_map_dict_eval)
    model_predict_batch_queue_info_list = [(Manager().Queue(), (list(), dict(), dict())) for _ in range(num_predict_batch_process)]
    # 相同thread_id，分train和eval线程
    train_thread_param_list = list()
    for pid in range(num_predict_batch_process):
        for tid in range(num_game_loop_thread):
            thread_id = pid * num_game_loop_thread + tid
            thread_name = f'{pid}_{tid}'
            batch_out_queue = Manager().Queue()

            workflow_game_loop_signal_queue = Manager().Queue()
            workflow_game_loop_signal_queue_list.append(workflow_game_loop_signal_queue)
            workflow_game_loop_ack_signal_queue = Manager().Queue()
            workflow_game_loop_ack_signal_queue_list.append(workflow_game_loop_ack_signal_queue)

            batch_queue_info_train = map_train_thread_to_queue_info_train(thread_id, model_predict_batch_queue_info_list)
            map_batch_predict_process_to_out_queue(thread_id, batch_out_queue, batch_queue_info_train[1][0], batch_queue_info_train[1][1], batch_queue_info_train[1][2])
            train_thread_param = (train_game_id_signal_queue, model.num_output_class, game_train_data_queue, train_game_finished_signal_queue, winning_probability_generating_task_queue, batch_queue_info_train[0], batch_out_queue, num_bins, small_blind, big_blind, num_mcts_simulation_per_step, mcts_c_puct, mcts_tau, mcts_dirichlet_noice_epsilon, mcts_model_Q_epsilon, workflow_lock, workflow_game_loop_signal_queue, workflow_game_loop_ack_signal_queue, mcts_log_to_file, mcts_choice_method, thread_id, thread_name)

            train_thread_param_list.append((train_thread_param, None))
            logging.info(f'Finished init train thread {thread_id}')

    is_init_train_thread = True
    is_init_eval_thread = False
    for pid in range(num_train_eval_process):
        Process(target=train_eval_process, args=(train_thread_param_list[pid * num_game_loop_thread_per_process: (pid + 1) * num_game_loop_thread_per_process], is_init_train_thread, is_init_eval_thread, pid, log_level), daemon=True).start()
    logging.info('All train_eval_process inited.')

    # batch predict process：接收一个in_queue的输入，从out_queue_list中选择一个输出，选择规则遵从map_dict
    for pid, (workflow_queue, workflow_ack_queue, update_model_param_queue, (model_predict_batch_in_queue, (model_predict_batch_out_queue_list, model_predict_batch_out_map_dict_train, model_predict_batch_out_map_dict_eval))) in enumerate(zip(workflow_queue_list, workflow_ack_queue_list, update_model_param_queue_list, model_predict_batch_queue_info_list)):
        Process(target=predict_batch_process, args=(model_predict_batch_in_queue, model_predict_batch_out_queue_list, model_predict_batch_out_map_dict_train, model_predict_batch_out_map_dict_eval, predict_batch_size, model_param_dict, update_model_param_queue, workflow_queue, workflow_ack_queue, None, None, pid, log_level), daemon=True).start()
    logging.info('All predict_batch_process inited.')

    # load model and synchronize to all predict_batch_process
    load_model_and_synchronize(model, model_init_checkpoint_path, update_model_param_queue_list, workflow_ack_queue_list)

    # control game simulation thread, to prevent excess task produced by producer
    game_finalized_signal_queue = Queue()
    Thread(target=train_game_control_thread, args=(train_game_id_signal_queue, game_finalized_signal_queue, env_info_dict, num_game_loop_thread * num_predict_batch_process), daemon=True).start()

    finished_game_id = None
    finished_game_id_dict = dict()
    game_train_data_list_dict = dict()
    while True:
        if interrupt.interrupt_callback():
            logging.info("main loop detect interrupt")
            break

        while True:
            try:
                if interrupt.interrupt_callback():
                    logging.info("train_gather_result_thread detect interrupt")
                    break

                game_id, finished_game_info = game_train_data_queue.get(block=False)
                if game_id in game_train_data_list_dict:
                    game_train_data_list = game_train_data_list_dict[game_id]
                else:
                    game_train_data_list = list()
                    game_train_data_list_dict[game_id] = game_train_data_list
                game_train_data_list.append(finished_game_info)
            except Empty:
                break

        while True:
            try:
                if interrupt.interrupt_callback():
                    logging.info("train_gather_result_thread detect interrupt")
                    break

                signal_finished_game_id = train_game_finished_signal_queue.get(block=False)
                finished_game_id_dict[signal_finished_game_id] = len(game_train_data_list_dict[signal_finished_game_id])
            except Empty:
                break

        finalized_game_id_set = set()
        for finished_game_id, num_total_games in finished_game_id_dict.items():
            if finished_game_id in game_train_data_list_dict and finished_game_id in env_info_dict:
                game_train_data_list = game_train_data_list_dict[finished_game_id]
                game_info_dict = env_info_dict[finished_game_id]

                game_train_data_not_finished_list = list()
                for game_train_data in game_train_data_list:
                    train_data_list, step_info = game_train_data

                    round_num = step_info[KEY_ROUND_NUM]
                    player_name = step_info[KEY_ACTED_PLAYER_NAME]
                    if player_name in game_info_dict and round_num in game_info_dict[player_name]:
                        observation, estimate_value_probs, estimate_value_Qs = train_data_list
                        estimate_winning_prob = game_info_dict[player_name][round_num]
                        model_value_probs, action_Q_loss = model.cal_reward([observation])
                        estimate_value_probs_str = ','.join('%.2f' % value for value in estimate_value_probs)
                        model_value_probs_str = ','.join('%.2f' % value for value in model_value_probs)
                        # todo: model input observation
                        estimation_action_Q_str = ','.join('%.2f' % value for value in estimate_value_Qs)
                        action_Q_str = ','.join('%.2f' % value for value in action_Q_loss)
                        logging.info(f'observation={observation}\nestimate_value_probs={estimate_value_probs_str}\nmodel_value_probs={model_value_probs_str}\nestimation_action_Q_str={estimation_action_Q_str}\naction_Qs={action_Q_str}\n')
                    else:
                        game_train_data_not_finished_list.append(game_train_data)

                if len(game_train_data_not_finished_list) == 0:
                    finalized_game_id_set.add(finished_game_id)
                    logging.info('Game %s finished, generated %d data' % (finished_game_id, num_total_games))
                else:
                    game_train_data_list_dict[finished_game_id] = game_train_data_not_finished_list

        # 清掉这局的数据cache
        for finalized_game_id in finalized_game_id_set:
            # 清掉这局的数据cache
            del game_train_data_list_dict[finalized_game_id]
            del env_info_dict[finalized_game_id]
            del finished_game_id_dict[finalized_game_id]
            game_finalized_signal_queue.put(finalized_game_id)

        time.sleep(1.)

