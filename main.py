import logging
from env.workflow import *
from tools.param_parser import *
from tools.high_performance_queue import Many2OneQueue, One2ManyQueue
from tools import counter
from queue import Queue
from torch.multiprocessing import Manager, Process, Condition, Queue as MPQueue


if __name__ == '__main__':
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    logging.info(f"main process pid {os.getpid()} started")

    args, params = parse_params()

    batch_predict_model_type = ModelType(params['batch_predict_model_type'])
    if batch_predict_model_type == ModelType.PYTORCH:
        torch.multiprocessing.set_start_method('spawn')

    num_predict_batch_process = params['num_predict_batch_process']
    assert num_predict_batch_process % 2 == 0, 'num_predict_batch_process must be even'
    update_model_param_queue_list = [MPQueue() for _ in range(num_predict_batch_process)]
    logging.info(f'Finished init {num_predict_batch_process} update_model_param_queue_list.')
    workflow_queue_list = [MPQueue() for _ in range(num_predict_batch_process)]
    logging.info(f'Finished init {num_predict_batch_process} workflow_queue_list.')
    workflow_ack_queue_list = [MPQueue() for _ in range(num_predict_batch_process)]
    logging.info(f'Finished init {num_predict_batch_process} workflow_ack_queue_list.')

    # 游戏设置
    small_blind = params['small_blind']
    big_blind = params['big_blind']
    update_model_bb_per_100_thres = params['update_model_bb_per_100_thres']

    # 用户所有连续特征分桶embedding
    num_bins = params['num_bins']
    model_param_dict = params['model_param_dict']
    model_last_checkpoint_path = params['model_last_checkpoint_path']
    model_init_checkpoint_path = params['model_init_checkpoint_path']

    # init generate winning probability calculating data processes
    num_gen_winning_prob_cal_data_processes = params['num_gen_winning_prob_cal_data_processes']
    simulation_recurrent_param_dict = params['simulation_recurrent_params']
    env_info_dict = dict()
    winning_probability_generating_task_queue = Manager().Queue()
    game_result_out_queue = MPQueue()
    for pid in range(num_gen_winning_prob_cal_data_processes):
        Process(target=simulate_processes, args=(winning_probability_generating_task_queue, game_result_out_queue, simulation_recurrent_param_dict, pid, log_level), daemon=True).start()
    logging.info(f'Finished init {num_gen_winning_prob_cal_data_processes} simulate_processes.')

    # reward receiving thread
    Thread(target=receive_game_result_thread, args=(game_result_out_queue, env_info_dict), daemon=True).start()

    # game simulation threads (train and evaluate)
    # 必须保证线程数 > batch_size * 2，以在train中gpu基本打满，在eval中不死锁
    predict_batch_size = params['predict_batch_size']
    num_game_loop_thread = int(predict_batch_size * params['game_loop_thread_multiple'])
    # train_eval进程数
    num_train_eval_process = params['num_train_eval_process']
    num_train_eval_thread = num_game_loop_thread * num_predict_batch_process
    assert num_train_eval_thread % num_train_eval_process == 0, 'num_game_loop_thread must be multiple of num_train_eval_process'
    assert num_train_eval_process % num_predict_batch_process == 0, 'num_train_eval_process must be multiple of num_predict_batch_process'
    num_game_loop_thread_per_process = num_train_eval_thread // num_train_eval_process
    # game params
    num_mcts_simulation_per_step = params['num_mcts_simulation_per_step']
    mcts_c_puct = params['mcts_c_puct']
    mcts_tau = params['mcts_tau']
    mcts_log_to_file = params['mcts_log_to_file']
    mcts_dirichlet_noice_epsilon = params['mcts_dirichlet_noice_epsilon']
    mcts_model_Q_epsilon = params['mcts_model_Q_epsilon']
    mcts_choice_method = params['mcts_choice_method']
    workflow_lock = Condition()
    workflow_game_loop_signal_queue_list = [MPQueue() for _ in range(num_train_eval_process)]
    workflow_game_loop_ack_signal_queue = MPQueue()
    logging.info(f'Finished init {num_train_eval_process} workflow_game_loop_signal_queue_list.')
    # train thread参数
    game_train_data_queue = MPQueue()
    train_game_finished_signal_queue = MPQueue()
    train_game_id_signal_queue = MPQueue()
    # eval thread参数
    eval_game_finished_reward_queue = MPQueue()
    eval_game_id_signal_queue = MPQueue()
    eval_workflow_signal_queue_list = [MPQueue() for _ in range(num_train_eval_process)]
    eval_workflow_ack_signal_queue = MPQueue()
    logging.info(f'Finished init {num_train_eval_process} eval_workflow_signal_queue_list.')
    # in_queue, train_tid_pid_map, eval_tid_pid_map
    predict_feature_size_list = params['predict_feature_size_list']
    # spawn进程模式下不支持SharedMemory内存共享，会导致对拷失败
    if batch_predict_model_type == ModelType.TENSORRT:
        predict_batch_in_queue_info_list = [(Many2OneQueue(predict_feature_size_list, np.int32(), n_producers_over_process=num_train_eval_process, n_producers_in_process=num_game_loop_thread_per_process, max_queue_size=num_train_eval_thread), dict(), dict()) for _ in range(num_predict_batch_process)]
        logging.info(f'Finished init {num_predict_batch_process} predict_batch_in_queue_info_list.')

        # data_out_queue
        predict_batch_out_queue_list = [One2ManyQueue(predict_feature_size_list, np.int32(), n_consumers_over_process=num_train_eval_process, n_consumers_in_process=num_game_loop_thread_per_process, max_queue_size=num_train_eval_thread) for _ in range(num_predict_batch_process)]
        logging.info(f'Finished init {num_train_eval_process} predict_batch_out_queues_list.')
    else:
        raise ValueError(f'Only support TensorRT for Queue mode, but get {batch_predict_model_type}')
    # tid_process_tid_map(tid: in_train_eval_process_tid)
    predict_batch_out_info_list = [dict() for _ in range(num_train_eval_process)]
    # tid_train_eval_pid_map(tid: train_eval_process_id)
    tid_train_eval_pid_dict = dict()
    # 相同thread_id，分train和eval线程
    train_eval_thread_param_list = list()
    for pid in range(num_predict_batch_process):
        for tid in range(num_game_loop_thread):
            thread_id = pid * num_game_loop_thread + tid
            thread_name = f'{pid}_{tid}'
            train_eval_process_pid = thread_id // num_game_loop_thread_per_process
            tid_train_eval_pid_dict[thread_id] = train_eval_process_pid

            predict_batch_in_queue_info = get_train_info(train_eval_process_pid, predict_batch_in_queue_info_list)
            predict_batch_in_queue_info[1][thread_id] = train_eval_process_pid

            predict_batch_in_best_queue_info, predict_batch_in_new_queue_info = get_eval_info(thread_id, predict_batch_in_queue_info_list)
            predict_batch_in_best_queue_info[2][thread_id] = train_eval_process_pid
            predict_batch_in_new_queue_info[2][thread_id] = train_eval_process_pid

            predict_batch_out_queue_info = predict_batch_out_info_list[train_eval_process_pid]
            predict_batch_out_queue_info[thread_id] = len(predict_batch_out_queue_info)

            # train_thread_param = (train_game_id_signal_queue, model_param_dict['num_output_class'], game_train_data_queue, train_game_finished_signal_queue, winning_probability_generating_task_queue, predict_batch_in_queue_info[0].producer_list[thread_id], num_bins, small_blind, big_blind, num_mcts_simulation_per_step, mcts_c_puct, mcts_tau, mcts_dirichlet_noice_epsilon, mcts_model_Q_epsilon, workflow_lock, workflow_game_loop_ack_signal_queue, mcts_log_to_file, mcts_choice_method, thread_id, thread_name)

            # eval_thread_param = (eval_game_id_signal_queue, model_param_dict['num_output_class'], eval_game_finished_reward_queue, eval_workflow_ack_signal_queue, predict_batch_in_best_queue_info[0].producer_list[thread_id], predict_batch_in_new_queue_info[0].producer_list[thread_id], num_bins, small_blind, big_blind, num_mcts_simulation_per_step, mcts_c_puct, mcts_model_Q_epsilon, mcts_choice_method, thread_id, thread_name)

            train_thread_param = (train_game_id_signal_queue, model_param_dict['num_output_class'], game_train_data_queue, train_game_finished_signal_queue, winning_probability_generating_task_queue, num_bins, small_blind, big_blind, num_mcts_simulation_per_step, mcts_c_puct, mcts_tau, mcts_dirichlet_noice_epsilon, mcts_model_Q_epsilon, workflow_lock, workflow_game_loop_ack_signal_queue, mcts_log_to_file, mcts_choice_method, thread_id, thread_name)

            eval_thread_param = (eval_game_id_signal_queue, model_param_dict['num_output_class'], eval_game_finished_reward_queue, eval_workflow_ack_signal_queue, num_bins, small_blind, big_blind, num_mcts_simulation_per_step, mcts_c_puct, mcts_model_Q_epsilon, mcts_choice_method, thread_id, thread_name)

            train_eval_thread_param_list.append((train_thread_param, eval_thread_param))

    is_init_train_thread = True
    is_init_eval_thread = True
    for pid, (tid_process_tid_map, workflow_game_loop_signal_queue, eval_workflow_signal_queue) in enumerate(zip(predict_batch_out_info_list, workflow_game_loop_signal_queue_list, eval_workflow_signal_queue_list)):
        data_queue_idx = get_train_info_for_process(pid, predict_batch_in_queue_info_list)
        data_best_queue_idx, data_new_queue_idx = get_eval_info_for_process(pid, predict_batch_in_queue_info_list)

        data_in_queue_list = [predict_batch_in_queue_info[0].producer_list for predict_batch_in_queue_info in predict_batch_in_queue_info_list]
        data_out_queue_list = [predict_batch_out_queue.consumer_list for predict_batch_out_queue in predict_batch_out_queue_list]

        Process(target=train_eval_process, args=(train_eval_thread_param_list[pid * num_game_loop_thread_per_process: (pid + 1) * num_game_loop_thread_per_process], is_init_train_thread, is_init_eval_thread, data_in_queue_list, data_out_queue_list, data_queue_idx, data_best_queue_idx, data_new_queue_idx, workflow_game_loop_signal_queue, eval_workflow_signal_queue, tid_process_tid_map, pid, log_level), daemon=True).start()
    logging.info('All train_eval_process inited.')

    # 训练中更新模型参数
    train_update_model_queue_list = [MPQueue() for _ in range(num_predict_batch_process)]
    train_update_model_ack_queue = MPQueue()
    train_hold_signal_queue_list = [MPQueue() for _ in range(num_predict_batch_process)]
    train_hold_signal_ack_queue = MPQueue()
    # batch predict process：接收一个in_queue的输入，从out_queue_list中选择一个输出，选择规则遵从map_dict
    for pid, ((predict_batch_in_queue, model_predict_batch_out_map_dict_train, model_predict_batch_out_map_dict_eval), predict_batch_out_queue, workflow_queue, workflow_ack_queue, update_model_param_queue, train_update_model_queue, train_hold_signal_queue) in enumerate(zip(predict_batch_in_queue_info_list, predict_batch_out_queue_list, workflow_queue_list, workflow_ack_queue_list, update_model_param_queue_list, train_update_model_queue_list, train_hold_signal_queue_list)):
        Process(target=predict_batch_process, args=(predict_batch_in_queue, model_predict_batch_out_map_dict_train, model_predict_batch_out_map_dict_eval, batch_predict_model_type, params, update_model_param_queue, workflow_queue, workflow_ack_queue, train_update_model_queue, train_update_model_ack_queue, train_hold_signal_queue, train_hold_signal_ack_queue, predict_batch_out_queue, pid, log_level), daemon=True).start()
    logging.info('All predict_batch_process inited.')

    # load model and synchronize to all predict_batch_process
    model = AlphaGoZero(**model_param_dict)
    load_model_and_synchronize(model, model_init_checkpoint_path, update_model_param_queue_list, workflow_ack_queue_list, batch_predict_model_type)

    # control game simulation thread, to prevent excess task produced by producer
    game_finalized_signal_queue = Queue()
    game_id_counter = counter.Counter()
    seed_counter = counter.Counter()
    Thread(target=train_game_control_thread, args=(train_game_id_signal_queue, game_finalized_signal_queue, env_info_dict, game_id_counter, seed_counter), daemon=True).start()

    # gather train result and save to buffer
    step_counter = counter.Counter()
    Thread(target=train_gather_result_thread, args=(game_train_data_queue, train_game_finished_signal_queue, game_finalized_signal_queue, env_info_dict, model, step_counter), daemon=True).start()

    # init training thread to separate cpu/gpu time
    is_save_model = True
    eval_model_queue = Queue()
    train_update_model_signal_queue = Queue()
    first_train_data_step = params['first_train_data_step']
    train_per_step = params['train_per_step']
    update_model_per_train_step = params['update_model_per_train_step']
    eval_model_per_step = params['eval_model_per_step']
    log_step_num = params['log_step_num']
    historical_data_filename = params['historical_data_filename']
    Thread(target=training_thread, args=(model, model_last_checkpoint_path, step_counter, is_save_model, eval_model_queue, first_train_data_step, train_per_step, update_model_per_train_step, eval_model_per_step, log_step_num, historical_data_filename, game_id_counter, seed_counter, env_info_dict, train_game_id_signal_queue, num_train_eval_thread, train_update_model_signal_queue, train_hold_signal_queue_list, train_hold_signal_ack_queue), daemon=True).start()

    # to monitor performance
    Thread(target=performance_monitor_thread, args=(winning_probability_generating_task_queue,), daemon=True).start()

    workflow_status = WorkflowStatus.DEFAULT
    best_model_trt_filename = model_init_checkpoint_path.replace('.pth', '.trt')
    new_model_trt_filename = None
    tmp_checkpoint_path = None
    best_model_state_dict = get_state_dict_from_model(model)
    new_model_state_dict = None
    new_optimizer_state_dict = None
    best_model_update_state_queue_list, new_model_update_state_queue_list = get_best_new_queues_for_eval(update_model_param_queue_list)
    eval_game_seed = int(1e9)
    eval_task_num_games = params['eval_task_num_games']
    eval_task_id = 0
    new_model_net_win_value_sum = 0
    is_update_old_model = False
    model_eval_snapshot_path_format = params['model_eval_snapshot_path_format']
    model_best_checkpoint_path = params['model_best_checkpoint_path']
    model_workflow_tmp_checkpoint_path = params['model_workflow_tmp_checkpoint_path']
    model_param_dict_for_save = model_param_dict.copy()
    model_param_dict_for_save['device'] = 'cpu'
    model_for_save = AlphaGoZero(**model_param_dict_for_save)
    while True:
        if interrupt.interrupt_callback():
            logging.info("main loop detect interrupt")
            if tmp_checkpoint_path is not None and os.path.exists(tmp_checkpoint_path):
                os.remove(tmp_checkpoint_path)
            break

        if workflow_status == WorkflowStatus.TRAINING:
            # 接收新的eval任务，停掉train任务中的MCTS模拟过程
            try:
                new_model_state_dict, new_optimizer_state_dict = eval_model_queue.get(block=False)
                eval_task_id += 1

                logging.info(f"Main thread start to switched workflow to {workflow_status.name}")
                workflow_status = switch_workflow_default(WorkflowStatus.TRAIN_FINISH_WAIT, workflow_queue_list, workflow_ack_queue_list)
                logging.info(f"Main thread switched workflow to {workflow_status.name}")

                new_model_trt_filename = save_model_by_state_dict(new_model_state_dict, new_optimizer_state_dict, model_eval_snapshot_path_format % eval_task_id, model_for_save, batch_predict_model_type, params)
            except Empty:
                # 先切换队列状态，如果队列状态不变，只在train任务中同步模型
                step_num, train_model_state_dict = None, None
                while True:
                    try:
                        step_num, train_model_state_dict = train_update_model_signal_queue.get(block=False)
                    except Empty:
                        break
                if step_num is not None and train_model_state_dict is not None:
                    if batch_predict_model_type == ModelType.TENSORRT:
                        tmp_checkpoint_path = save_model_by_state_dict(train_model_state_dict, None, model_workflow_tmp_checkpoint_path, model_for_save, batch_predict_model_type, params)
                    for train_update_model_queue in train_update_model_queue_list:
                        if batch_predict_model_type == ModelType.PYTORCH:
                            train_update_model_queue.put((step_num, train_model_state_dict))
                        elif batch_predict_model_type == ModelType.TENSORRT:
                            train_update_model_queue.put((step_num, tmp_checkpoint_path))
                    logging.info(f'Finished sending update model signals.')
                    if receive_and_check_ack_from_queue(workflow_status, train_update_model_ack_queue, len(train_update_model_queue_list)):
                        logging.info(f"Model state updating workflow finished at training step {step_num}.")
            finally:
                time.sleep(0.1)
        elif workflow_status == WorkflowStatus.TRAIN_FINISH_WAIT:
            # 清空batch predict进程中的任务队列，结束train任务
            if not switch_workflow_for_predict_process_default(workflow_status, workflow_game_loop_signal_queue_list, workflow_game_loop_ack_signal_queue, tid_train_eval_pid_dict):
                exit(-1)

            workflow_status = switch_workflow_default(WorkflowStatus.REGISTERING_EVAL_MODEL, workflow_queue_list, workflow_ack_queue_list)
            logging.info(f"Main thread switched workflow to {workflow_status.name}")
        elif workflow_status == WorkflowStatus.REGISTERING_EVAL_MODEL:
            # 负责新模型推理的batch predict进程注册新模型
            for new_model_update_state_queue in new_model_update_state_queue_list:
                if batch_predict_model_type == ModelType.PYTORCH:
                    new_model_update_state_queue.put(new_model_state_dict)
                elif batch_predict_model_type == ModelType.TENSORRT:
                    new_model_update_state_queue.put(new_model_trt_filename)
            for best_model_update_state_queue in best_model_update_state_queue_list:
                if batch_predict_model_type == ModelType.PYTORCH:
                    best_model_update_state_queue.put(best_model_state_dict)
                elif batch_predict_model_type == ModelType.TENSORRT:
                    best_model_update_state_queue.put(best_model_trt_filename)
            if not receive_and_check_all_ack(workflow_status, workflow_ack_queue_list):
                exit(-1)

            workflow_status = switch_workflow_default(WorkflowStatus.EVALUATING, workflow_queue_list, workflow_ack_queue_list)
            logging.info(f"Main thread switched workflow to {workflow_status.name}")
        elif workflow_status == WorkflowStatus.EVALUATING:
            # eval任务开始推理
            eval_game_seed = eval_task_begin(eval_game_seed, eval_game_id_signal_queue, eval_task_num_games=eval_task_num_games)

            while not eval_game_id_signal_queue.empty():
                time.sleep(1.)

            workflow_status = switch_workflow_default(WorkflowStatus.EVAL_FINISH_WAIT, workflow_queue_list, workflow_ack_queue_list)
            logging.info(f"Main thread switched workflow to {workflow_status.name}")
        elif workflow_status == WorkflowStatus.EVAL_FINISH_WAIT:
            # 等待所有eval任务结束，接收eval结果
            new_model_net_win_value_sum = 0
            new_model_player_name = get_new_model_player_name()
            finished_eval_game_id_set = set()
            while len(finished_eval_game_id_set) != eval_task_num_games:
                try:
                    if interrupt.interrupt_callback():
                        logging.info("main loop detect interrupt")
                        break

                    finished_eval_game_id, eval_reward_dict = eval_game_finished_reward_queue.get(block=True, timeout=1.)
                    assert finished_eval_game_id not in finished_eval_game_id_set, f'game_id:{finished_eval_game_id} repeated for finished evaluation task'
                    finished_eval_game_id_set.add(finished_eval_game_id)

                    player_result_value, reward_value, net_win_value = eval_reward_dict[new_model_player_name]
                    new_model_net_win_value_sum += net_win_value
                except Empty:
                    continue

            # eval流程收到中断处理
            if len(finished_eval_game_id_set) == eval_task_num_games:
                # 计算bb/100
                num_statics_round = eval_task_num_games / 100
                new_model_bb_per_100 = new_model_net_win_value_sum / num_statics_round / big_blind
                is_update_old_model = new_model_bb_per_100 > update_model_bb_per_100_thres

                logging.info(f'Evaluation finished for task_id={eval_task_id}, new_model_bb_per_100={new_model_bb_per_100}')

                workflow_status = switch_workflow_default(WorkflowStatus.REGISTERING_TRAIN_MODEL, workflow_queue_list, workflow_ack_queue_list)
                # 重置eval线程中的env
                if not switch_workflow_for_predict_process_default(workflow_status, eval_workflow_signal_queue_list, eval_workflow_ack_signal_queue, tid_train_eval_pid_dict):
                    exit(-1)
                logging.info(f"Main thread switched workflow to {workflow_status.name}")
        elif workflow_status == WorkflowStatus.REGISTERING_TRAIN_MODEL:
            # batch predict切换模型，继续开始train任务
            if is_update_old_model:
                # 在此处切换新旧模型pointer
                best_model_trt_filename = save_model_by_state_dict(new_model_state_dict, new_optimizer_state_dict, model_best_checkpoint_path, model_for_save, batch_predict_model_type, params)
                best_model_state_dict = new_model_state_dict

            # 永远使用最佳模型训练，依据AlphaGO Zero论文，Method章Self-play节：
            # The best current player αθ∗, as selected by the evaluator, is used to generate data.
            # todo: Also consider the 'league training' method introduced in <Grandmaster level in StarCraft II using multi-agent reinforcement learning>, since the algrithm may chase cycles (A defeats B, and B defeats C, but A loses to C) in poker games.
            for update_state_queue in update_model_param_queue_list:
                if batch_predict_model_type == ModelType.PYTORCH:
                    update_state_queue.put(best_model_state_dict)
                elif batch_predict_model_type == ModelType.TENSORRT:
                    update_state_queue.put(best_model_trt_filename)
            if not receive_and_check_all_ack(workflow_status, workflow_ack_queue_list):
                exit(-1)

            workflow_status = switch_workflow_default(WorkflowStatus.TRAINING, workflow_queue_list, workflow_ack_queue_list)
            logging.info(f"Main thread switched workflow to {workflow_status.name}")

            with workflow_lock:
                workflow_lock.notify_all()
