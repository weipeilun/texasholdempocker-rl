import signal
import time

from env.workflow import *
from tools.param_parser import *
from tools.data_loader import *
from torch.multiprocessing import Manager, Process, Condition


def train_process(params, n_loop, log_level):
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    num_predict_batch_process = params['num_predict_batch_process']
    assert num_predict_batch_process % 2 == 0, 'num_predict_batch_process must be even'
    update_model_param_queue_list = [Manager().Queue() for _ in range(num_predict_batch_process)]
    workflow_queue_list = [Manager().Queue() for _ in range(num_predict_batch_process)]
    workflow_ack_queue_list = [Manager().Queue() for _ in range(num_predict_batch_process)]

    # 游戏设置
    small_blind = params['small_blind']
    big_blind = params['big_blind']
    update_model_bb_per_100_thres = params['update_model_bb_per_100_thres']

    # 用户所有连续特征分桶embedding
    num_bins = params['num_bins']
    model_param_dict = params['model_param_dict']
    model = AlphaGoZero(**model_param_dict)

    # game simulation threads (train and evaluate)
    # 必须保证线程数 > batch_size * 2，以在train中gpu基本打满，在eval中不死锁
    predict_batch_size = params['predict_batch_size']
    num_game_loop_thread = int(predict_batch_size * params['game_loop_thread_multiple'])
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
    workflow_lock = Condition()
    # eval thread参数
    eval_game_finished_reward_queue = Manager().Queue()
    eval_game_id_signal_queue = Manager().Queue()
    eval_workflow_signal_queue_list = list()
    eval_workflow_ack_signal_queue_list = list()
    # in_queue, (out_queue_list, out_queue_map_dict_train, out_queue_map_dict_eval)
    model_predict_batch_queue_info_list = [(Manager().Queue(), (list(), dict(), dict())) for _ in range(num_predict_batch_process)]
    # 相同thread_id，分train和eval线程
    train_eval_thread_param_list = list()
    for pid in range(num_predict_batch_process):
        for tid in range(num_game_loop_thread):
            thread_id = pid * num_game_loop_thread + tid
            thread_name = f'{pid}_{tid}'
            batch_out_queue = Manager().Queue()

            eval_workflow_signal_queue = Manager().Queue()
            eval_workflow_ack_signal_queue = Manager().Queue()
            eval_workflow_signal_queue_list.append(eval_workflow_signal_queue)
            eval_workflow_ack_signal_queue_list.append(eval_workflow_ack_signal_queue)

            batch_queue_info_eval_best, batch_queue_info_eval_new = map_train_thread_to_queue_info_eval(thread_id, model_predict_batch_queue_info_list)
            map_batch_predict_process_to_out_queue(thread_id, batch_out_queue, batch_queue_info_eval_best[1][0], batch_queue_info_eval_best[1][2], batch_queue_info_eval_best[1][1])
            map_batch_predict_process_to_out_queue(thread_id, batch_out_queue, batch_queue_info_eval_new[1][0], batch_queue_info_eval_new[1][2], batch_queue_info_eval_new[1][1])
            eval_thread_param = (eval_game_id_signal_queue, model.num_output_class, eval_game_finished_reward_queue, eval_workflow_signal_queue, eval_workflow_ack_signal_queue, batch_queue_info_eval_best[0], batch_queue_info_eval_new[0], batch_out_queue, num_bins, small_blind, big_blind, num_mcts_simulation_per_step, mcts_c_puct, mcts_model_Q_epsilon, thread_id, thread_name)

            train_eval_thread_param_list.append((None, eval_thread_param))
            logging.info(f'Finished init params for train and eval thread {thread_id}')

    is_init_train_thread = False
    is_init_eval_thread = True
    for pid in range(num_train_eval_process):
        Process(target=train_eval_process, args=(train_eval_thread_param_list[pid * num_game_loop_thread_per_process: (pid + 1) * num_game_loop_thread_per_process], is_init_train_thread, is_init_eval_thread, pid, log_level), daemon=True).start()
    logging.info('All train_eval_process inited.')

    # batch predict process：接收一个in_queue的输入，从out_queue_list中选择一个输出，选择规则遵从map_dict
    for pid, (workflow_queue, workflow_ack_queue, update_model_param_queue, (model_predict_batch_in_queue, (model_predict_batch_out_queue_list, model_predict_batch_out_map_dict_train, model_predict_batch_out_map_dict_eval))) in enumerate(zip(workflow_queue_list, workflow_ack_queue_list, update_model_param_queue_list, model_predict_batch_queue_info_list)):
        Process(target=predict_batch_process, args=(model_predict_batch_in_queue, model_predict_batch_out_queue_list, model_predict_batch_out_map_dict_train, model_predict_batch_out_map_dict_eval, predict_batch_size, model_param_dict, update_model_param_queue, workflow_queue, workflow_ack_queue, pid, log_level), daemon=True).start()
    logging.info('All predict_batch_process inited.')

    # synchronize model to all predict_batch_process
    original_model_param = get_state_dict_from_model(model)
    for update_model_param_queue in update_model_param_queue_list:
        update_model_param_queue.put(original_model_param)
    for workflow_ack_queue in workflow_ack_queue_list:
        workflow_ack_queue.get(block=True, timeout=None)
    model_save_checkpoint_path = params['model_save_checkpoint_path_format'] % n_loop
    save_model(model, model_save_checkpoint_path)

    train_data_path = params['train_data_path']
    log_step_num = params['log_step_num']
    predict_step_num = params['predict_step_num']
    num_train_steps = params['num_train_steps']
    batch_size = params['model_param_dict']['batch_size']
    device = params['model_param_dict']['device']
    train_step_num = 0
    train_batch_gen = data_batch_generator(train_data_path, batch_size=batch_size, epoch=-1)
    for observation_list, action_probs_list, action_Q_list in train_batch_gen:
        logging.info(f'get data for train_step {train_step_num}')

        observation_array = np.array(observation_list)
        action_probs_array = np.array(action_probs_list)
        action_Q_array = np.array(action_Q_list)
        observation_tensor = torch.tensor(observation_array, dtype=torch.int32, device=device, requires_grad=False)
        action_probs_tensor = torch.tensor(action_probs_array, dtype=torch.float32, device=device, requires_grad=False)
        action_Q_tensor = torch.tensor(action_Q_array, dtype=torch.float32, device=device, requires_grad=False)
        try:
            action_probs_loss, action_Q_loss = model.learn(observation_tensor, action_probs_tensor, action_Q_tensor)
        except ValueError as e:
            logging.error(f'ValueError in model.learn: {e}')
            signal.alarm(0)
            time.sleep(5)
            exit(-1)
        train_step_num += 1

        if train_step_num % log_step_num == 0:
            logging.info(f'train_step {train_step_num}, action_probs_loss={action_probs_loss}, action_Q_loss={action_Q_loss}')

        if train_step_num % predict_step_num == 0:
            logging.info(f'start predict for train_step {train_step_num}')
            with torch.no_grad():
                action_probs_tensor, player_result_value_tensor = model(observation_tensor)
                logging.info(f'finished predict for train_step {train_step_num}')
                predict_action_probs_list = action_probs_tensor.cpu().numpy()
                predict_player_result_value_list = player_result_value_tensor.cpu().numpy()
            for action_probs, action_Q, predict_action_probs, predict_player_result_value in zip(action_probs_list, action_Q_list, predict_action_probs_list, predict_player_result_value_list):
                logging.info(f'action_probs={",".join(["%.4f" % item for item in action_probs])}\npredict_action_probs={",".join(["%.4f" % item for item in predict_action_probs.tolist()])}\naction_Q={",".join(["%.4f" % item for item in action_Q])}\npredict_winning_prob={",".join(["%.4f" % item for item in predict_player_result_value.tolist()])}\n')

        if train_step_num >= num_train_steps:
            break
    logging.info('Finish training.')

    # eval
    eval_task_num_games = params['eval_task_num_games']
    model_eval_snapshot_path = params['model_eval_snapshot_path_format'] % n_loop
    new_state_dict = get_state_dict_from_model(model)
    save_model(model, model_eval_snapshot_path)

    best_model_update_state_queue_list, new_model_update_state_queue_list = get_best_new_queues_for_eval(update_model_param_queue_list)
    best_model_workflow_queue_list, new_model_workflow_queue_list = get_best_new_queues_for_eval(workflow_queue_list)
    best_model_workflow_ack_queue_list, new_model_workflow_ack_queue_list = get_best_new_queues_for_eval(workflow_ack_queue_list)

    workflow_status = WorkflowStatus.TRAINING
    # 负责新模型推理的batch predict进程注册新模型
    for new_model_update_state_queue in new_model_update_state_queue_list:
        new_model_update_state_queue.put(new_state_dict)
    if not receive_and_check_all_ack(workflow_status, new_model_workflow_ack_queue_list):
        exit(-1)

    workflow_status = switch_workflow_default(WorkflowStatus.EVALUATING, workflow_queue_list, workflow_ack_queue_list)
    logging.info(f"Main thread switched workflow to {workflow_status.name}")

    # eval任务开始推理
    eval_game_seed = eval_task_begin(int(1e9), eval_game_id_signal_queue, eval_task_num_games=eval_task_num_games)

    while not eval_game_id_signal_queue.empty():
        time.sleep(1.)

    workflow_status = switch_workflow_default(WorkflowStatus.EVAL_FINISH_WAIT, workflow_queue_list, workflow_ack_queue_list)
    logging.info(f"Main thread switched workflow to {workflow_status.name}")

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

        logging.info(f'Evaluation finished for loop={n_loop}, new_model_bb_per_100={new_model_bb_per_100}')

    signal.alarm(0)
    train_batch_gen.close()
    time.sleep(5)
    exit(0)


def validate_params(new_params, params_buffer):
    for historical_params in params_buffer:
        if historical_params == new_params:
            return False
    return True


if __name__ == '__main__':
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    default_params, train_params = parse_test_train_params()

    torch.multiprocessing.set_start_method('spawn')

    num_choices = train_params['num_choices']
    params_buffer = list()
    for i in range(num_choices):
        logging.info(f'start train and eval loop {i}')

        chosen_task_params = choose_params_concurrent(train_params)
        while not validate_params(chosen_task_params, params_buffer):
            chosen_task_params = choose_params_concurrent(train_params)
        params_buffer.append(chosen_task_params)
        params = update_concurrent(default_params.copy(), chosen_task_params.copy())

        p = Process(target=train_process, args=(params, i, log_level))
        p.start()
        p.join()
        logging.info(f'end train and eval loop {i}')
