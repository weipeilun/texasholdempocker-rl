import logging
import time
import traceback
from threading import Thread
from env.env import *
from tools.data_loader import *
from rl.AlphaGoZero import AlphaGoZero
from queue import Empty, Queue
from rl.MCTS import MCTS
from utils.torch_utils import *
from utils.workflow_utils import *
from utils.tensorrt_utils import *
from utils.math_utils import *

try:
    from cuda import cudart
except:
    pass


def save_model(model, path):
    models_dict = dict()
    models_dict['model'] = model.state_dict()
    models_dict['optimizer'] = model.optimizer.state_dict()
    torch.save(models_dict, path)
    logging.info(f'model saved to {path}')


def save_model_by_state_dict(model_state_dict, optimizer_state_dict, path, model, batch_predict_model_type, params):
    models_dict = dict()
    models_dict['model'] = model_state_dict
    if optimizer_state_dict is not None:
        models_dict['optimizer'] = optimizer_state_dict
    torch.save(models_dict, path)

    if batch_predict_model_type == ModelType.TENSORRT:
        onnx_path = path.replace('.pth', '.onnx')
        model.load_state_dict(model_state_dict, strict=False)
        input_names = ['input']
        dynamic_axes = {'input': {0: 'batch_size'}}
        torch.onnx.export(model, torch.zeros((params['predict_batch_size_max'], *params['predict_feature_size_list']), dtype=torch.int32).to(model.device), onnx_path, opset_version=14, input_names=input_names, dynamic_axes=dynamic_axes)

        # onnxsim.simplify在未知原因的特定情况下会不抛异常导致进程直接崩溃：Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)。干掉这个步骤
        # trt_read_path = tmp_onnx_path
        # try:
        #     onnx_checkpoint = onnx.load(tmp_onnx_path)
        #     model_simple, is_simplify_success = onnxsim.simplify(onnx_checkpoint)
        #     assert is_simplify_success, ValueError(f'is_simplify_success={is_simplify_success}')
        #     onnx.save(model_simple, onnx_path)
        #     trt_read_path = onnx_path
        # except Exception as e:
        #     logging.warning(f'onnxsim failed, {e}')

        trt_path = path.replace('.pth', '.trt')
        trt_model_engine = build_engine(onnx_path, params['predict_batch_size_min'], params['predict_batch_size'], params['predict_batch_size_max'], params['predict_feature_size_list'])
        with open(trt_path, "wb") as f:
            f.write(trt_model_engine)
        logging.info(f'finished writing TensorRT model file {trt_path}')

        logging.info(f'model saved to {path}, {onnx_path}, {trt_path}')
        return trt_path
    else:
        logging.info(f'model saved to {path}')
        return None


def load_model_and_synchronize(model, model_path, update_model_param_queue_list, workflow_ack_queue_list, batch_predict_model_type):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model_param = checkpoint['model']
        model.load_state_dict(model_param, strict=False)
        if 'optimizer' in checkpoint:
            model.optimizer.load_state_dict(checkpoint['optimizer'])

        if update_model_param_queue_list is not None and len(update_model_param_queue_list) > 0 and workflow_ack_queue_list is not None and len(workflow_ack_queue_list) > 0:
            if batch_predict_model_type == ModelType.PYTORCH:
                for update_model_param_queue in update_model_param_queue_list:
                    update_model_param_queue.put(model_param)
                for workflow_ack_queue in workflow_ack_queue_list:
                    workflow_ack_queue.get(block=True, timeout=None)
                logging.info(f'model loaded from {model_path}')
            elif batch_predict_model_type == ModelType.TENSORRT:
                tensorrt_path = model_path.replace('.pth', '.trt')
                if os.path.exists(tensorrt_path):
                    for update_model_param_queue in update_model_param_queue_list:
                        update_model_param_queue.put(tensorrt_path)
                    for workflow_ack_queue in workflow_ack_queue_list:
                        workflow_ack_queue.get(block=True, timeout=None)
                    logging.info(f'model loaded from {model_path}')


# 接生成模拟数据的任务，生成模拟数据，计算胜率（子进程）
def simulate_processes(in_queue, out_queue, simulation_recurrent_param_dict, pid, log_level):
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    logging.info(f"simulate_processes {pid} pid {os.getpid()} started")

    # current_round: [(current_round_gen_num, num_random_generates, is_segment_end)]
    # 第一轮只遍历flop轮，其他随机，控制在10w次模拟(19600 * 5)
    # 第二轮只遍历turn轮，其他随机，控制在10w次模拟(2162 * 45)
    # 第三轮全部遍历，模拟次数(990 * 46)
    # 第四轮全部遍历，模拟次数(990)
    # simulation_recurrent_param_dict = {
    #     0: ([(1, 0, False), (1, 0, False), (1, 0, True), (4, 5, True)], 98000),
    #     1: ([(1, 0, True), (1, 0, True), (2, 45, True)], 97290),
    #     2: ([(1, 0, True), (1, 0, False), (1, 0, True)], 45540),
    #     3: ([(1, 0, False), (1, 0, True)], 990)
    # }

    def cal_unordered_combination(num_all, num_choice):
        result = 1.
        for i in range(num_choice):
            result *= (num_all - i)
            result /= (i + 1)
        return int(result)

    # 计算牌局结果
    def cal_game_result(cards_generated_list):
        player_hand_card = sorted(cards_generated_list[0: 2])
        flop_cards = sorted(cards_generated_list[2: 5])
        turn_cards = sorted(cards_generated_list[5: 6])
        river_cards = sorted(cards_generated_list[6: 7])
        current_other_player_hand_cards = [sorted(cards_generated_list[7: 9])]

        player_hand_card_dict = dict()
        player_hand_card_dict[0] = player_hand_card
        for i, hand_card in enumerate(current_other_player_hand_cards):
            player_hand_card_dict[i + 1] = hand_card

        winner_player = GameEnv.get_winner_set_v2(flop_cards, turn_cards, river_cards, player_hand_card_dict)

        player_game_result_dict = {0: GamePlayerResult.LOSE}
        GameEnv.generate_game_result(winner_player, player_game_result_dict)
        game_result = player_game_result_dict[0].value

        return game_result

    def generate_and_calculate_recurrent(generate_recurrent_and_randomly_num_list, exist_card_list, current_round_deck_idx):
        game_result_sum = 0.
        num_generated = 0
        current_round_gen_num, num_random_generates, is_segment_end = generate_recurrent_and_randomly_num_list.pop(0)
        num_cards_to_generate_all_other_round = 0
        if not is_segment_end:
            for round_gen_num, _, segment_end in generate_recurrent_and_randomly_num_list:
                num_cards_to_generate_all_other_round += round_gen_num
                if segment_end:
                    break

        is_last_round = len(generate_recurrent_and_randomly_num_list) == 0
        exist_card_set = set(exist_card_list)
        if current_round_gen_num == 1:
            for deck_idx in range(current_round_deck_idx, NUM_CARDS):
                current_card = deck[deck_idx]
                if current_card not in exist_card_set:
                    new_exist_card_list = exist_card_list.copy()
                    new_exist_card_list.append(current_card)
                    if is_last_round:
                        game_result_sum += cal_game_result(new_exist_card_list)
                        num_generated += 1
                    elif num_cards_to_generate_all_other_round + deck_idx < NUM_CARDS:
                        new_round_gen_num_list = generate_recurrent_and_randomly_num_list.copy()
                        if is_segment_end:
                            new_current_round_deck_idx = 0
                        else:
                            new_current_round_deck_idx = deck_idx + 1
                        new_game_result, new_num_generated = generate_and_calculate_recurrent(new_round_gen_num_list, new_exist_card_list, new_current_round_deck_idx)
                        game_result_sum += new_game_result
                        num_generated += new_num_generated
        else:
            all_cards_not_exist_list = list()
            for deck_idx in range(0, NUM_CARDS):
                current_card = deck[deck_idx]
                if current_card not in exist_card_set:
                    all_cards_not_exist_list.append(current_card)
            # 可随机生成的无序组合数大于本回合应生成的牌局数才有意义
            num_cards_not_exist_combination = cal_unordered_combination(len(all_cards_not_exist_list), current_round_gen_num)
            if num_cards_not_exist_combination >= num_random_generates:
                for _ in range(num_random_generates):
                    generated_cards = exist_card_list.copy()
                    cards_choice = random.sample(all_cards_not_exist_list, current_round_gen_num)
                    generated_cards.extend(cards_choice)
                    game_result_sum += cal_game_result(generated_cards)
                    num_generated += 1
            else:
                raise ValueError(f'Card combination generator error: can not generate {num_random_generates} cards randomly if card choice is {len(all_cards_not_exist_list)}')

        return game_result_sum, num_generated
    
    while True:
        try:
            if interrupt.interrupt_callback():
                logging.info(f"data_generator_process {pid} detect interrupt")
                return

            game_id, player_name, current_round, flop_cards, turn_cards, river_cards, player_hand_card = in_queue.get(block=True, timeout=0.1)

            generate_recurrent_and_randomly_num_list, num_estimation = simulation_recurrent_param_dict[current_round]
            generate_recurrent_and_randomly_num_list = generate_recurrent_and_randomly_num_list.copy()

            exist_card_list = player_hand_card.copy()
            if current_round > 0:
                for card in flop_cards:
                    exist_card_list.append(card)
            if current_round > 1:
                for card in turn_cards:
                    exist_card_list.append(card)
            if current_round > 2:
                for card in river_cards:
                    exist_card_list.append(card)
            game_result_sum, num_data_generated = generate_and_calculate_recurrent(generate_recurrent_and_randomly_num_list, exist_card_list, 0)
            assert num_data_generated == num_estimation, f'Actual number of data generated:{num_data_generated} not equals to num_estimation:{num_estimation}, multiprocess calculation will miscalculate.'

            out_queue.put((game_id, player_name, current_round, num_estimation, game_result_sum))
        except Empty:
            continue


# 聚合模拟数据reward结果（主进程）
def receive_game_result_thread(in_queue, env_info_dict):
    while True:
        if interrupt.interrupt_callback():
            logging.info("receive_reward_thread detect interrupt")
            break

        try:
            game_id, player_name, current_round, num_current_task_estimation, game_result_sum = in_queue.get(block=True, timeout=0.1)

            player_result_value = game_result_sum / num_current_task_estimation
            # GamePlayerResult的取值范围是是[-1, 1]，归一到[0, 1]
            winning_probability = player_result_value * 0.5 + 0.5

            if game_id in env_info_dict:
                game_info_dict = env_info_dict[game_id]
                if player_name in game_info_dict:
                    player_round_info_dict = game_info_dict[player_name]
                else:
                    player_round_info_dict = dict()
                    game_info_dict[player_name] = player_round_info_dict

                player_round_info_dict[current_round] = winning_probability
            else:
                logging.error(f'game_id {game_id} not found in env_info_dict')
        except Empty:
            continue
        except:
            error_info = str(traceback.format_exc())
            logging.error('receive_reward_thread error: trace=%s' % error_info)
            break


# batch预测（子进程）
def predict_batch_process(in_queue, out_queue_map_dict_train, out_queue_map_dict_eval, batch_predict_model_type, params, update_model_param_queue, workflow_queue, workflow_ack_queue, train_update_model_queue, train_update_model_ack_queue, train_hold_signal_queue,train_hold_signal_ack_queue, predict_batch_out_queue_list, pid, log_level):
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    logging.info(f"predict_batch_process {pid} pid {os.getpid()} started")

    # 初始化tensorRT batch buffer
    def init_tensorrt_instances(trt_filename, batch_size_list, feature_size_list, stream, batch_info_dict, tensorrt_min_infer_batch_size):
        engine = load_engine(trt_filename)
        context = engine.create_execution_context()
        context.set_optimization_profile_async(0, stream)
        if batch_info_dict is None or tensorrt_min_infer_batch_size is None:
            # 这个参数的意思是：WAIT_FINISH状态下，数据持续流入没有中断时的最小预测batch_size
            tensorrt_min_infer_batch_size = 16
            batch_info_dict = dict()
            for batch_size in batch_size_list:
                input_shape = (batch_size, *feature_size_list)
                input_dim = np.prod(input_shape)
                inputs, outputs, bindings = allocate_buffers(engine, context, 0, 'input', input_shape)
                batch_info_dict[batch_size] = (input_shape, input_dim, inputs, outputs, bindings)
        return engine, context, batch_info_dict, tensorrt_min_infer_batch_size

    if batch_predict_model_type == ModelType.PYTORCH:
        # PyTorch在GPU不可用时使用
        batch_size = params['predict_batch_size']
        model_param_dict = params['model_param_dict']
        model = AlphaGoZero(**model_param_dict)
        model.eval()
    elif batch_predict_model_type == ModelType.TENSORRT:
        # 使用tensorRT加速推理
        stream = cuda_call(cudart.cudaStreamCreate())
        engine, context = None, None
        batch_size_list = sorted(params['predict_batch_size_list'], reverse=True)
        feature_size_list = params['predict_feature_size_list']
        batch_info_dict, tensorrt_min_infer_batch_size = None, None
    else:
        raise ValueError(f'Invalid batch_predict_model_type: {batch_predict_model_type.name}')

    def send_thread(send_in_queue, send_out_queue_list, send_out_queue_map_dict_train, send_out_queue_map_dict_eval, logging_queue, pid):
        while True:
            if interrupt.interrupt_callback():
                logging.info(f"predict_batch_thread_{pid}.send_thread detect interrupt")
                break

            try:
                action_probs_list, reward_value_list, winning_prob_list, send_pid_list, send_workflow_status = send_in_queue.get(block=True, timeout=0.01)

                for action_probs, reward_value, winning_prob, send_pid in zip(action_probs_list, reward_value_list, winning_prob_list, send_pid_list):
                    if send_workflow_status == WorkflowStatus.TRAINING or send_workflow_status == WorkflowStatus.TRAIN_FINISH_WAIT:
                        logging.info(f'send_out_queue_map_dict_train={send_out_queue_map_dict_train}')
                        send_out_queue_list[send_out_queue_map_dict_train[send_pid]].put((send_pid, (action_probs, reward_value, winning_prob)))
                    elif send_workflow_status == WorkflowStatus.EVALUATING or send_workflow_status == WorkflowStatus.EVAL_FINISH_WAIT:
                        send_out_queue_list[send_out_queue_map_dict_eval[send_pid]].put((send_pid, (action_probs, reward_value, winning_prob)))
                    else:
                        raise ValueError(f'Invalid workflow_status: {send_workflow_status.name} when batch predicting.')

                logging_queue.put(len(action_probs_list))
            except Empty:
                continue

    def queue_monitor_thread(in_queue, send_in_queue, out_queue_list, logging_queue, pid):
        last_log_time = time.time()
        while True:
            if interrupt.interrupt_callback():
                logging.info("performance_monitor_thread detect interrupt")
                break

            out_queue_str_list = list()
            for idx, out_queue in enumerate(out_queue_list):
                out_queue_str = f'out_queue{idx} qsize:{out_queue.qsize()}'
                out_queue_str_list.append(out_queue_str)

            num_data_predicted = 0
            while True:
                try:
                    num_data_predicted += logging_queue.get(block=False)
                except Empty:
                    break

            now = time.time()
            time_elapsed = now - last_log_time
            last_log_time = now
            data_processed_per_second = num_data_predicted / time_elapsed
            logging.info(f'pid={pid}, data_processed_per_second:{data_processed_per_second:.2f}, in_queue qsize:{in_queue.qsize()}, send_in_queue qsize:{send_in_queue.qsize()}, {", ".join(out_queue_str_list)}')
            time.sleep(120)

    # 另开一个线程，cpu和gpu任务分开，防止阻塞predict_batch_process
    send_in_queue = Queue()
    # 用于输出性能指标
    logging_queue = Queue()
    Thread(target=send_thread, args=(send_in_queue, predict_batch_out_queue_list, out_queue_map_dict_train, out_queue_map_dict_eval, logging_queue, pid), daemon=True).start()
    Thread(target=queue_monitor_thread, args=(in_queue, send_in_queue, predict_batch_out_queue_list, logging_queue, pid), daemon=True).start()

    def predict_and_send(predict_send_model, predict_send_batch_list, predict_send_pid_list, predict_send_send_in_queue, predict_send_workflow_status):
        with torch.no_grad():
            observation_array = np.array(predict_send_batch_list)
            observation_tensor = torch.tensor(observation_array, dtype=torch.int32, device=model.device, requires_grad=False)
            action_probs_logits_tensor, reward_value_tensor, winning_prob_tensor = predict_send_model(observation_tensor)
            action_probs_tensor = torch.softmax(action_probs_logits_tensor, dim=1)
            action_probs_list = action_probs_tensor.cpu().numpy()
            reward_value_list = reward_value_tensor.cpu().numpy()
            winning_prob_list = winning_prob_tensor.cpu().numpy()

        predict_send_send_in_queue.put((action_probs_list, reward_value_list, winning_prob_list, predict_send_pid_list, predict_send_workflow_status))

    def predict_and_send_trt(data_list, data_pid_list, predict_send_send_in_queue, predict_send_workflow_status, context, bindings, batch_size, input_shape, input_dim, inputs, outputs, stream):
        context.set_input_shape("input", input_shape)
        data_np = np.asarray(data_list).reshape(input_dim)
        np.copyto(inputs[0].host, data_np)
        action_prob_logits, reward_value, winning_prob = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        action_prob_logits = np.copy(action_prob_logits).reshape((batch_size, -1))
        reward_value = np.copy(reward_value)
        winning_prob = np.copy(winning_prob)
        action_probs = softmax_np(action_prob_logits)
        predict_send_send_in_queue.put((action_probs, reward_value, winning_prob, data_pid_list, predict_send_workflow_status))

    batch_list = list()
    pid_list = list()
    workflow_status = WorkflowStatus.DEFAULT
    while True:
        if interrupt.interrupt_callback():
            logging.info(f"predict_batch_process_{pid} detect interrupt")
            if batch_predict_model_type == ModelType.TENSORRT and batch_info_dict is not None:
                for _, _, inputs, outputs, _ in batch_info_dict.values():
                    free_buffers(inputs, outputs, stream)
            break

        # 切换进程状态
        try:
            workflow_status = workflow_queue.get(block=False)
            workflow_ack_queue.put(workflow_status)
        except Empty:
            pass
        # 切换进程状态结束

        try:
            msg = update_model_param_queue.get(block=False)
            if batch_predict_model_type == ModelType.PYTORCH:
                model_dict = msg
                model.load_state_dict(model_dict)
                logging.info(f'predict_batch_process_{pid} model updated')
            elif batch_predict_model_type == ModelType.TENSORRT:
                trt_filename = msg
                engine, context, batch_info_dict, tensorrt_min_infer_batch_size = init_tensorrt_instances(trt_filename, batch_size_list, feature_size_list, stream, batch_info_dict, tensorrt_min_infer_batch_size)
                logging.info(f'predict_batch_process_{pid} model updated: {trt_filename}')
            workflow_ack_queue.put(workflow_status)
        except Empty:
            pass

        try:
            if batch_predict_model_type == ModelType.PYTORCH:
                data_pid, data = in_queue.get(block=True, timeout=0.01)
                batch_list.append(data)
                pid_list.append(data_pid)

                if batch_size == len(batch_list):
                    # 切换进程状态
                    try:
                        workflow_status = workflow_queue.get(block=False)
                        workflow_ack_queue.put(workflow_status)
                    except Empty:
                        pass
                    # 切换进程状态结束

                    # 等待训练流程，更新模型
                    try:
                        if train_update_model_queue is not None and train_update_model_ack_queue is not None and train_hold_signal_queue is not None and train_hold_signal_ack_queue is not None:
                            hold_train_step = train_hold_signal_queue.get(block=False)
                            logging.info(f'predict_batch_process_{pid} received hold signal at training step {hold_train_step}')
                            train_hold_signal_ack_queue.put(TrainHoldStatus.WAITING_MODEL)

                            while True:
                                try:
                                    step_num, train_model_dict = train_update_model_queue.get(block=True, timeout=0.1)
                                    if train_model_dict is not None and workflow_status == WorkflowStatus.TRAINING:
                                        model.load_state_dict(train_model_dict)
                                        logging.info(f'predict_batch_process_{pid} model training updated')
                                    else:
                                        logging.info(f'predict_batch_process_{pid} model training updated skipped')
                                    train_update_model_ack_queue.put(WorkflowStatus.TRAINING)
                                    train_hold_signal_ack_queue.put((TrainHoldStatus.FINISHED_UPDATING, step_num))
                                    break
                                except Empty:
                                    pass
                    except Empty:
                        pass
                    # 等待训练流程结束

                    predict_and_send(model, batch_list, pid_list, send_in_queue, workflow_status)
                    batch_list = list()
                    pid_list = list()
            elif batch_predict_model_type == ModelType.TENSORRT:
                # 取空队列，永远按最大batch做预测
                while True:
                    # 一直取数，不要sleep，以提高显卡利用率
                    data_pid, data = in_queue.get(block=True, timeout=0.001)
                    logging.info(f'data_pid={data_pid}, data={data}')
                    batch_list.append(data)
                    pid_list.append(data_pid)
                    # 达到min_infer_batch_size，做预测
                    # 这主要在WAIT_FINISH状态下，数据持续流入没有中断情况下，可以保证预测以一个最小的batch_size进行
                    if len(batch_list) >= tensorrt_min_infer_batch_size:
                        for batch_size, (input_shape, input_dim, inputs, outputs, bindings) in batch_info_dict.items():
                            if batch_size <= len(batch_list):
                                # 切换进程状态
                                try:
                                    workflow_status = workflow_queue.get(block=False)
                                    workflow_ack_queue.put(workflow_status)
                                except Empty:
                                    pass
                                # 切换进程状态结束

                                # 等待训练流程，更新模型
                                try:
                                    if train_update_model_queue is not None and train_update_model_ack_queue is not None and train_hold_signal_queue is not None and train_hold_signal_ack_queue is not None:
                                        hold_train_step = train_hold_signal_queue.get(block=False)
                                        logging.info(f'predict_batch_process_{pid} received hold signal at training step {hold_train_step}')
                                        train_hold_signal_ack_queue.put(TrainHoldStatus.WAITING_MODEL)

                                        while True:
                                            try:
                                                step_num, train_trt_filename = train_update_model_queue.get(block=True, timeout=0.1)
                                                if train_trt_filename is not None and workflow_status == WorkflowStatus.TRAINING:
                                                    engine, context, batch_info_dict, tensorrt_min_infer_batch_size = init_tensorrt_instances(train_trt_filename, batch_size_list, feature_size_list, stream, batch_info_dict, tensorrt_min_infer_batch_size)
                                                    logging.info(f'predict_batch_process_{pid} model training updated: {train_trt_filename} at step {step_num}')
                                                else:
                                                    logging.info(f'predict_batch_process_{pid} model training updated skipped at step {step_num}')
                                                train_update_model_ack_queue.put(WorkflowStatus.TRAINING)
                                                logging.info(f'predict_batch_process_{pid} sent ack to train_update_model_ack_queue at step {step_num}')
                                                train_hold_signal_ack_queue.put((TrainHoldStatus.FINISHED_UPDATING, step_num))
                                                logging.info(f'predict_batch_process_{pid} sent ack to train_hold_signal_ack_queue at step {step_num}')
                                                break
                                            except Empty:
                                                pass
                                except Empty:
                                    pass
                                # 等待训练流程结束

                                # predict and send
                                predict_and_send_trt(batch_list[:batch_size], pid_list[:batch_size], send_in_queue, workflow_status, context, bindings, batch_size, input_shape, input_dim, inputs, outputs, stream)
                                batch_list = batch_list[batch_size:].copy()
                                pid_list = pid_list[batch_size:].copy()
                                break
                            if workflow_status == WorkflowStatus.TRAINING or workflow_status == WorkflowStatus.EVALUATING:
                                # 训练或评估中，只以最大的batch_size做预测，以提高显卡利用率
                                break
        except Empty:
            if batch_predict_model_type == ModelType.PYTORCH:
                # 小于一个batch_size也做预测，这样会降低性能。仅在任务即将结束，即可能产生小于一个batch的数据导致死锁时使用
                if (workflow_status == WorkflowStatus.TRAIN_FINISH_WAIT or workflow_status == WorkflowStatus.EVAL_FINISH_WAIT) and len(batch_list) > 0:
                    # 等待训练流程，更新模型
                    try:
                        if train_update_model_queue is not None and train_update_model_ack_queue is not None and train_hold_signal_queue is not None and train_hold_signal_ack_queue is not None:
                            hold_train_step = train_hold_signal_queue.get(block=False)
                            logging.info(f'predict_batch_process_{pid} received hold signal at training step {hold_train_step}')
                            train_hold_signal_ack_queue.put(TrainHoldStatus.WAITING_MODEL)

                            while True:
                                try:
                                    step_num, train_model_dict = train_update_model_queue.get(block=True, timeout=0.1)
                                    if train_model_dict is not None and workflow_status == WorkflowStatus.TRAINING:
                                        model.load_state_dict(train_model_dict)
                                        logging.info(f'predict_batch_process_{pid} model training updated')
                                    else:
                                        logging.info(f'predict_batch_process_{pid} model training updated skipped')
                                    train_update_model_ack_queue.put(WorkflowStatus.TRAINING)
                                    train_hold_signal_ack_queue.put((TrainHoldStatus.FINISHED_UPDATING, step_num))
                                    break
                                except Empty:
                                    pass
                    except Empty:
                        pass
                    # 等待训练流程结束

                    predict_and_send(model, batch_list, pid_list, send_in_queue, workflow_status)
                    batch_list = list()
                    pid_list = list()
            elif batch_predict_model_type == ModelType.TENSORRT:
                if len(batch_list) > 0:
                    for idx, (batch_size, (input_shape, input_dim, inputs, outputs, bindings)) in enumerate(batch_info_dict.items()):
                        if batch_size <= len(batch_list):
                            # 等待训练流程，更新模型
                            try:
                                if train_update_model_queue is not None and train_update_model_ack_queue is not None and train_hold_signal_queue is not None and train_hold_signal_ack_queue is not None:
                                    hold_train_step = train_hold_signal_queue.get(block=False)
                                    logging.info(
                                        f'predict_batch_process_{pid} received hold signal at training step {hold_train_step}')
                                    train_hold_signal_ack_queue.put(TrainHoldStatus.WAITING_MODEL)

                                    while True:
                                        try:
                                            step_num, train_trt_filename = train_update_model_queue.get(block=True, timeout=0.1)
                                            if train_trt_filename is not None and workflow_status == WorkflowStatus.TRAINING:
                                                engine, context, batch_info_dict, tensorrt_min_infer_batch_size = init_tensorrt_instances(train_trt_filename, batch_size_list, feature_size_list, stream, batch_info_dict, tensorrt_min_infer_batch_size)
                                                logging.info(f'predict_batch_process_{pid} model training updated: {train_trt_filename}')
                                            else:
                                                logging.info(f'predict_batch_process_{pid} model training updated skipped')
                                            train_update_model_ack_queue.put(WorkflowStatus.TRAINING)
                                            train_hold_signal_ack_queue.put((TrainHoldStatus.FINISHED_UPDATING, step_num))
                                            break
                                        except Empty:
                                            pass
                            except Empty:
                                pass
                            # 等待训练流程结束

                            predict_and_send_trt(batch_list[:batch_size], pid_list[:batch_size], send_in_queue, workflow_status, context, bindings, batch_size, input_shape, input_dim, inputs, outputs, stream)
                            batch_list = batch_list[batch_size:].copy()
                            pid_list = pid_list[batch_size:].copy()
                            break
                        if (workflow_status == WorkflowStatus.TRAINING or workflow_status == WorkflowStatus.EVALUATING) and idx > 0:
                            # 训练或评估中，最坏情况用top 2的batch_size做预测，以提高显卡利用率，否则等待数据
                            break
        except:
            error_info = str(traceback.format_exc())
            logging.error(f'predict_batch_process_{pid} error: trace=%s' % error_info)
            break


# 模型训练（主进程）
def training_thread(model, model_path, step_counter, is_save_model, eval_model_queue, first_train_data_step, train_per_step, update_model_per_train_step, eval_model_per_step, log_step_num, historical_data_filename, game_id_counter, seed_counter, env_info_dict, game_id_signal_queue, num_game_loop_thread, train_update_model_signal_queue, train_hold_signal_queue_list, train_hold_signal_ack_queue):
    assert train_per_step > 0, 'train_per_step must > 0.'

    next_train_step = first_train_data_step
    next_eval_step = eval_model_per_step
    train_step_num = 0

    if os.path.exists(historical_data_filename):
        logging.info(f'Found historical train data from {historical_data_filename}. Train with this first.')
        # load data and train
        data_generator = train_data_generator(historical_data_filename)
        for observation_list, action_probs_list, action_masks_list, reward_value_list, winning_prob_list in data_generator:
            model.store_transition(observation_list, action_probs_list, action_masks_list, reward_value_list, winning_prob_list, save_train_data=False)
            step_counter.increment()

            if step_counter.get_value() >= next_train_step:
                action_probs_loss, reward_value_loss, winning_prob_loss = model.learn()
                train_step_num += 1

                if train_step_num % log_step_num == 0:
                    logging.info(f'train_step {train_step_num}, action_probs_loss={action_probs_loss}, reward_value_loss={reward_value_loss}, winning_prob_loss={winning_prob_loss}')

                # 避免训练数据过多重复
                next_train_step += train_per_step

        # eval
        new_state_dict = get_state_dict_from_model(model)
        new_optimizer_state_dict = get_optimizer_state_dict_from_model(model)
        eval_model_queue.put((new_state_dict, new_optimizer_state_dict))
        next_eval_step = train_step_num + eval_model_per_step
        logging.info(f'Finished training historical data, train step: {train_step_num}, next eval step: {next_eval_step}')
    else:
        logging.info(f'No historical train data found, start training from scratch.')

    # 生成训练任务
    # 保持线程数=batch_size * 3，且线程数+2任务没有完成，可以保证瓶颈计算资源打满，同时有足够的任务进入模型batch队列
    # 如此保证整个流程不会阻塞
    # 这个方法放到这是为了解决训练和推理并行中RuntimeError: CUDA error: unspecified launch failure错误
    time.sleep(1)
    for _ in range(num_game_loop_thread + 2):
        game_id = game_id_counter.increment()
        seed = seed_counter.increment()

        # 此处保证game_id唯一，env_info_dict, game_train_data_queue_dict不用加锁
        game_info_dict = dict()
        env_info_dict[game_id] = game_info_dict

        game_id_signal_queue.put((game_id, seed))

    next_update_model_step = train_step_num + update_model_per_train_step
    while True:
        if interrupt.interrupt_callback():
            if is_save_model:
                save_model(model, model_path)
            logging.info("training_thread detect interrupt, model saved.")
            break

        if step_counter.get_value() >= next_train_step:
            train_step_num += 1

            # 训练、同步模型
            if train_step_num >= next_update_model_step:
                for train_hold_signal_queue in train_hold_signal_queue_list:
                    train_hold_signal_queue.put(train_step_num)
                logging.info(f'Train: train_step {train_step_num}, waiting for predict batch process to finish.')
                for _ in train_hold_signal_queue_list:
                    train_hold_status = train_hold_signal_ack_queue.get(block=True)
                    assert train_hold_status == TrainHoldStatus.WAITING_MODEL, f'train_hold_status should be {TrainHoldStatus.WAITING_MODEL.name}, but {train_hold_status.name}'
                logging.info(f'Train: train_step {train_step_num}, start to train.')

                action_probs_loss, reward_value_loss, winning_prob_loss = None, None, None
                for _ in range(update_model_per_train_step):
                    action_probs_loss, reward_value_loss, winning_prob_loss = model.learn()
                logging.info(f'train_step {train_step_num}, action_probs_loss={action_probs_loss}, reward_value_loss={reward_value_loss}, winning_prob_loss={winning_prob_loss}')

                new_state_dict = get_state_dict_from_model(model)
                train_update_model_signal_queue.put((train_step_num, new_state_dict))
                next_update_model_step += update_model_per_train_step
                
                for _ in train_hold_signal_queue_list:
                    train_hold_status, train_hold_step_num = train_hold_signal_ack_queue.get(block=True)
                    assert train_hold_status == TrainHoldStatus.FINISHED_UPDATING and train_hold_step_num == train_step_num, f'Train hold status check failed: train_hold_status={train_hold_status.name}, train_hold_step_num={train_hold_step_num}, train_step_num={train_step_num}'
                logging.info(f'Train: train_step {train_step_num}, finished training and updating.')

            # 触发eval任务
            if train_step_num >= next_eval_step:
                new_state_dict = get_state_dict_from_model(model)
                new_optimizer_state_dict = get_optimizer_state_dict_from_model(model)
                eval_model_queue.put((new_state_dict, new_optimizer_state_dict))
                next_eval_step += eval_model_per_step
                logging.info(f'Triggered eval task at train step: {train_step_num}, next eval step: {next_eval_step}')

            if is_save_model:
                if train_step_num % 200 == 0:
                    save_model(model, model_path)

            # 避免训练数据过多重复
            next_train_step += train_per_step
        else:
            time.sleep(0.1)


# 游戏对弈（train_eval_process进程）
def train_game_loop_thread(game_id_seed_signal_queue, n_actions, game_train_data_queue, game_finished_signal_queue, winning_probability_generating_task_queue, num_bins, small_blind, big_blind, num_mcts_simulation_per_step, mcts_c_puct, mcts_tau, mcts_dirichlet_noice_epsilon, mcts_model_Q_epsilon, workflow_lock, workflow_ack_signal_queue, mcts_log_to_file, mcts_choice_method, pid, thread_name, model_predict_out_queue, workflow_signal_queue, model_predict_in_queue):
    # 训练线程，使用RandomEnv，来生成更离散的数据，避免模型过拟合导致训练数据过于集中
    env = RandomEnv(winning_probability_generating_task_queue, num_bins, num_players=MAX_PLAYER_NUMBER, small_blind=small_blind, big_blind=big_blind)
    player_name_predict_in_queue_dict = get_player_name_model_dict(model_predict_in_queue, model_predict_in_queue)

    game_id = None
    seed = None
    while True:
        if interrupt.interrupt_callback():
            logging.info(f"train_game_loop_thread {thread_name} detect interrupt")
            break

        while True:
            try:
                if interrupt.interrupt_callback():
                    logging.info(f"train_game_loop_thread {thread_name} (game_id_signal_queue) detect interrupt")
                    break

                game_id, seed = game_id_seed_signal_queue.get(block=True, timeout=1.)
                logging.info(f"train_game_loop_thread {thread_name} start to simulate game {game_id}")
                break
            except Empty:
                continue

        observation = env.reset(game_id, seed=seed)
        logging.info(f"train_game_loop_thread {thread_name} game {game_id} env reset.")

        game_step_id = 0
        while True:
            if interrupt.interrupt_callback():
                logging.info(f"train_game_loop_thread {thread_name} detect interrupt")
                break

            t0 = time.time()
            logging.info(f'train_game_loop_thread {thread_name} game {game_id} start MCTS simulation step {game_step_id}')
            mcts = MCTS(n_actions, is_root=True, predict_in_queue=player_name_predict_in_queue_dict[env.acting_player_name], predict_out_queue=model_predict_out_queue, apply_dirichlet_noice=True, workflow_lock=workflow_lock, workflow_signal_queue=workflow_signal_queue, workflow_ack_signal_queue=workflow_ack_signal_queue, small_blind=small_blind, n_simulation=num_mcts_simulation_per_step, c_puct=mcts_c_puct, tau=mcts_tau, dirichlet_noice_epsilon=mcts_dirichlet_noice_epsilon, model_Q_epsilon=mcts_model_Q_epsilon, choice_method=mcts_choice_method, log_to_file=mcts_log_to_file, pid=pid, thread_name=thread_name)
            logging.info(f'train_game_loop_thread {thread_name} game {game_id} MCTS inited step {game_step_id}')
            action_probs = mcts.simulate(observation=observation, env=env)
            logging.info(f'train_game_loop_thread {thread_name} game {game_id} MCTS simulated step {game_step_id}')
            # 用于接收中断信号
            if action_probs is None:
                logging.info(f"train_game_loop_thread {thread_name} detect interrupt")
                break
            logging.info(f'train_game_loop_thread {thread_name} game {game_id} MCTS before take action step {game_step_id}')

            action, action_mask_idx = mcts.get_action(action_probs, env=env, choice_method=ChoiceMethod.PROBABILITY)
            t1 = time.time()
            logging.info(f'train_game_loop_thread {thread_name} game {game_id} MCTS took action:({action[0].name}, %d), cost:%.2fs, ' % (action[1], t1 - t0))

            observation_, reward, terminated, info = env.step(action)
            game_train_data_queue.put((game_id, ([observation, action_probs, action_mask_idx], info)))

            if not terminated:
                observation = observation_
            else:
                game_finished_signal_queue.put((game_id, reward, thread_name))
                break
            game_step_id += 1
    env.close()


# 游戏对弈（train_eval_process进程）
def eval_game_loop_thread(game_id_seed_signal_queue, n_actions, game_finished_reward_queue, eval_workflow_ack_signal_queue, num_bins, small_blind, big_blind, num_mcts_simulation_per_step, mcts_c_puct, mcts_model_Q_epsilon, mcts_choice_method, pid, thread_name, model_predict_out_queue, eval_workflow_signal_queue, model_predict_in_queue_best, model_predict_in_queue_new):
    env = Env(None, num_bins, num_players=MAX_PLAYER_NUMBER, ignore_all_async_tasks=True, small_blind=small_blind, big_blind=big_blind)
    player_name_predict_in_queue_dict = get_player_name_model_dict(model_predict_in_queue_best, model_predict_in_queue_new)

    game_id = None
    seed = None
    while True:
        if interrupt.interrupt_callback():
            logging.info(f"eval_game_loop_thread {thread_name} detect interrupt")
            break

        while True:
            try:
                if interrupt.interrupt_callback():
                    logging.info(f"eval_game_loop_thread {thread_name} (game_id_signal_queue) detect interrupt")
                    break

                # 一次eval任务结束需要重置env，否则导致players的初始资金在评估任务中不一致
                try:
                    workflow_status = eval_workflow_signal_queue.get(block=False)
                    env = Env(None, num_bins, num_players=MAX_PLAYER_NUMBER, ignore_all_async_tasks=True, small_blind=small_blind, big_blind=big_blind)

                    eval_workflow_ack_signal_queue.put(workflow_status)
                    logging.info(f"eval_game_loop_thread {thread_name} reset")
                except Empty:
                    pass

                game_id, seed = game_id_seed_signal_queue.get(block=True, timeout=1.)
                break
            except Empty:
                continue

        if game_id is not None and seed is not None:
            observation = env.reset(game_id, seed=seed)

            while True:
                if interrupt.interrupt_callback():
                    logging.info(f"eval_game_loop_thread {thread_name} detect interrupt")
                    break

                t0 = time.time()
                # 在eval时不使用dirichlet_noice以增加随机性，设置tau为0即在play步骤中丢弃随机性使用argmax
                mcts = MCTS(n_actions, is_root=True, predict_in_queue=player_name_predict_in_queue_dict[env.acting_player_name], predict_out_queue=model_predict_out_queue, small_blind=small_blind, apply_dirichlet_noice=False, workflow_lock=None, workflow_signal_queue=None, workflow_ack_signal_queue=None, n_simulation=num_mcts_simulation_per_step, c_puct=mcts_c_puct, tau=0, model_Q_epsilon=mcts_model_Q_epsilon, choice_method=mcts_choice_method, log_to_file=False, pid=pid, thread_name=thread_name)
                action_probs = mcts.simulate(observation=observation, env=env)
                # 用于接收中断信号
                if action_probs is None:
                    logging.info(f"eval_game_loop_thread {thread_name} detect interrupt")
                    break

                action, _ = mcts.get_action(action_probs, env=env, choice_method=ChoiceMethod.ARGMAX)
                t1 = time.time()
                # logging.info(f'MCTS took action:({action[0]}, %.4f), cost:%.2fs, eval_game_loop_thread id:{thread_name}' % (action[1], t1 - t0))

                observation_, reward, terminated, info = env.step(action)

                if not terminated:
                    observation = observation_
                else:
                    game_finished_reward_queue.put((game_id, reward))
                    logging.info(f'Evaluation finished for game {game_id}, eval_game_loop_thread id:{thread_name}')
                    break

            game_id = None
            seed = None
    env.close()


# train_eval_process进程，用于把cpu敏感任务分发到多个进程，避免单个进程打满
def train_eval_process(train_eval_thread_param_list, is_init_train_thread, is_init_eval_thread, data_in_queue_list, data_in_queue_idx, data_in_best_queue_idx, data_in_new_queue_idx, data_out_queue, workflow_game_loop_signal_queue, eval_workflow_signal_queue, tid_process_tid_map, pid, log_level):
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    logging.info(f"train_eval_process {pid} pid {os.getpid()} started")

    assert len(train_eval_thread_param_list) == len(tid_process_tid_map), f'Length of train_eval_thread_param_list:{len(train_eval_thread_param_list)} must equals to length of tid_process_tid_map:{len(tid_process_tid_map)}'

    # data_out_queue, workflow_signal_queue, eval_workflow_signal_queue
    data_map_queue_list = [Queue() for _ in tid_process_tid_map]
    workflow_signal_map_queue_list = [Queue() for _ in tid_process_tid_map]
    eval_workflow_signal_map_queue_list = [Queue() for _ in tid_process_tid_map]

    for tid, (train_eval_thread_param, data_map_queue, workflow_signal_map_queue, eval_workflow_signal_map_queue) in enumerate(zip(train_eval_thread_param_list, data_map_queue_list, workflow_signal_map_queue_list, eval_workflow_signal_map_queue_list)):
        predict_batch_in_queue = data_in_queue_list[data_in_queue_idx][pid]
        predict_batch_in_best_queue = data_in_queue_list[data_in_best_queue_idx][pid]
        predict_batch_in_new_queue = data_in_queue_list[data_in_new_queue_idx][pid]

        predict_batch_in_queue.start_map_data_thread()
        predict_batch_in_best_queue.start_map_data_thread()
        predict_batch_in_new_queue.start_map_data_thread()

        predict_batch_in_producer_queue = predict_batch_in_queue.producer_list[tid]
        predict_batch_in_producer_best_queue = predict_batch_in_best_queue.producer_list[tid]
        predict_batch_in_producer_new_queue = predict_batch_in_new_queue.producer_list[tid]

        if is_init_train_thread:
            Thread(target=train_game_loop_thread, args=(*train_eval_thread_param[0], data_map_queue, workflow_signal_map_queue, predict_batch_in_producer_queue), daemon=True).start()
        if is_init_eval_thread:
            Thread(target=eval_game_loop_thread, args=(*train_eval_thread_param[1], data_map_queue, eval_workflow_signal_map_queue, predict_batch_in_producer_best_queue, predict_batch_in_producer_new_queue), daemon=True).start()

    def map_data_thread(data_queue, data_map_dict, tid_process_tid_map, timeout):
        while True:
            if interrupt.interrupt_callback():
                logging.info(f"train_eval_process_{pid}.map_data_thread detect interrupt")
                break

            try:
                data_tid, data = data_queue.get(block=True, timeout=timeout)
                data_map_dict[tid_process_tid_map[data_tid]].put(data)
            except Empty:
                pass

    Thread(target=map_data_thread, args=(data_out_queue, data_map_queue_list, tid_process_tid_map, 0.01), daemon=True).start()
    Thread(target=map_data_thread, args=(workflow_game_loop_signal_queue, workflow_signal_map_queue_list, tid_process_tid_map, 0.1), daemon=True).start()
    Thread(target=map_data_thread, args=(eval_workflow_signal_queue, eval_workflow_signal_map_queue_list, tid_process_tid_map, 0.1), daemon=True).start()

    while True:
        if interrupt.interrupt_callback():
            logging.info(f"train_eval_process_{pid} detect interrupt")
            break
        time.sleep(1.)


# 游戏流程控制（主进程），防止生产者生产过多任务，导致任务队列过长
def train_game_control_thread(game_id_signal_queue, game_finalized_signal_queue, env_info_dict, game_id_counter, seed_counter):
    while True:
        if interrupt.interrupt_callback():
            logging.info("game_controll_thread detect interrupt")
            break

        while True:
            try:
                if interrupt.interrupt_callback():
                    logging.info("main loop (game_finished_signal_queue) detect interrupt")
                    break

                _ = game_finalized_signal_queue.get(block=True, timeout=0.1)
                game_id = game_id_counter.increment()
                seed = seed_counter.increment()

                # 此处保证game_id唯一，env_info_dict, game_train_data_queue_dict不用加锁
                game_info_dict = dict()
                env_info_dict[game_id] = game_info_dict

                game_id_signal_queue.put((game_id, seed))
            except Empty:
                continue


def eval_task_begin(seed, game_id_signal_queue, eval_task_num_games=1000):
    game_id = 0
    for _ in range(eval_task_num_games):
        game_id += 1
        seed += 1
        game_id_signal_queue.put((game_id, seed))
    return seed


# train流程结尾，模拟数据存入buffer（主进程）
def train_gather_result_thread(game_train_data_queue, game_finished_signal_queue, game_finalized_signal_queue, env_info_dict, model, step_counter):
    finished_game_id_dict = dict()
    game_train_data_list_dict = dict()

    while True:
        if interrupt.interrupt_callback():
            logging.info("train_gather_result_thread detect interrupt")
            break

        game_finished_signal_list = list()
        while True:
            try:
                if interrupt.interrupt_callback():
                    logging.info("train_gather_result_thread detect interrupt")
                    break

                game_finished_signal_info = game_finished_signal_queue.get(block=False)
                game_finished_signal_list.append(game_finished_signal_info)
            except Empty:
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

        # 此处会有多进程下的数据幻觉问题，会导致取数异常。把从game_finished_signal_queue取数放到前边
        for signal_finished_game_id, signal_finished_reward_dict, thread_name in game_finished_signal_list:
            while signal_finished_game_id not in game_train_data_list_dict:
                logging.warning(f'signal_finished_game_id {signal_finished_game_id} not in game_train_data_list_dict')
                time.sleep(1.)
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
            finished_game_id_dict[signal_finished_game_id] = (len(game_train_data_list_dict[signal_finished_game_id]), thread_name, signal_finished_reward_dict)

        finalized_game_id_set = set()
        for finished_game_id, (num_total_games, thread_name, finished_reward_dict) in finished_game_id_dict.items():
            if finished_game_id in game_train_data_list_dict and finished_game_id in env_info_dict:
                game_train_data_list = game_train_data_list_dict[finished_game_id]
                game_info_dict = env_info_dict[finished_game_id]

                game_train_data_not_finished_list = list()
                for game_train_data in game_train_data_list:
                    train_data_list, step_info = game_train_data

                    round_num = step_info[KEY_ROUND_NUM]
                    player_name = step_info[KEY_ACTED_PLAYER_NAME]
                    if player_name in game_info_dict and round_num in game_info_dict[player_name]:
                        # reward_dict = env._get_final_reward()
                        reward_value = finished_reward_dict[player_name][1]
                        winning_prob = game_info_dict[player_name][round_num]
                        model.store_transition(*train_data_list, reward_value, winning_prob)
                        step_counter.increment()
                    else:
                        game_train_data_not_finished_list.append(game_train_data)

                if len(game_train_data_not_finished_list) == 0:
                    finalized_game_id_set.add(finished_game_id)
                    logging.info(f'Game {finished_game_id} finished by train_game_loop_thread {thread_name}, generated {num_total_games} data.')
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


# 监控cpu瓶颈队列长度（主进程）
# 此监控可以用来平衡CPU/GPU使用效率
def performance_monitor_thread(winning_probability_generating_task_queue):
    while True:
        if interrupt.interrupt_callback():
            logging.info("performance_monitor_thread detect interrupt")
            break

        logging.info(f'winning_rate estimation qsize:{winning_probability_generating_task_queue.qsize()}')
        time.sleep(120)
