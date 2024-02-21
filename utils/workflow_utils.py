import logging
import threading
from queue import Empty
from env.constants import *
from tools import interrupt
import torch
import random
import numpy as np


def get_train_info(train_eval_process_pid, queue_info_list):
    num_in_queue = len(queue_info_list)
    in_queue_idx = train_eval_process_pid % num_in_queue
    return queue_info_list[in_queue_idx]


def get_train_info_for_process(process_id, queue_info_list):
    num_in_queue = len(queue_info_list)
    in_queue_idx = process_id % num_in_queue
    return in_queue_idx


def get_eval_info(thread_id, queue_info_list):
    num_in_queue = len(queue_info_list)
    num_in_queue_group = num_in_queue // 2
    in_queue_group_idx = thread_id % num_in_queue_group
    return queue_info_list[in_queue_group_idx], queue_info_list[in_queue_group_idx + num_in_queue_group]


def get_eval_info_for_process(process_id, queue_info_list):
    num_in_queue = len(queue_info_list)
    num_in_queue_group = num_in_queue // 2
    in_queue_group_idx = process_id % num_in_queue_group
    return in_queue_group_idx, in_queue_group_idx + num_in_queue_group


def map_batch_predict_process_to_out_queue(thread_id, process_out_queue, process_out_queue_list, queue_map_dict_to_add, queue_map_dict_existed):
    if thread_id not in queue_map_dict_existed:
        out_queue_idx = len(process_out_queue_list)
        process_out_queue_list.append(process_out_queue)
        queue_map_dict_to_add[thread_id] = out_queue_idx
    else:
        queue_map_dict_to_add[thread_id] = queue_map_dict_existed[thread_id]


def get_best_new_queues_for_eval(queue_list):
    num_in_queue = len(queue_list)
    num_in_queue_group = num_in_queue // 2
    return queue_list[: num_in_queue_group], queue_list[num_in_queue_group:]


def send_signal_thread(signal_queue, signal, qid):
    try:
        logging.info(f"Send signal {signal} to queue {qid}.")
        signal_queue.put(signal, block=False)
    except Exception as e:
        logging.error(f"Failed to send signal {signal} to queue {qid}.")
        raise e


def receive_and_check_all_ack(target_ack_status, workflow_ack_queue_list):
    for idx, workflow_ack_queue in enumerate(workflow_ack_queue_list):
        while True:
            if interrupt.interrupt_callback():
                logging.info("main loop detect interrupt")
                exit(0)

            try:
                ack_status = workflow_ack_queue.get(block=True, timeout=0.1)
                if ack_status != target_ack_status:
                    logging.error(f"Workflow ack from queue {idx} error, expect {target_ack_status.name}, but get {ack_status.name}")
                    return False
                else:
                    logging.info(f"Workflow received ack {target_ack_status.name} from queue {idx}.")
                    break
            except Empty:
                continue
    return True


def switch_workflow_default(target_workflow_status, workflow_queue_list, workflow_ack_queue_list):
    logging.info(f"Start to switch to workflow status {target_workflow_status.name}")
    for qid, workflow_queue in enumerate(workflow_queue_list):
        threading.Thread(target=send_signal_thread, args=(workflow_queue, target_workflow_status, qid), daemon=True).start()
    if not receive_and_check_all_ack(target_workflow_status, workflow_ack_queue_list):
        exit(-1)
    logging.info(f"Finished to switch to workflow status {target_workflow_status.name}")
    return target_workflow_status


def receive_and_check_ack_from_queue(target_workflow_status, workflow_ack_queue, num_data_send):
    num_data_received = 0
    while True:
        if interrupt.interrupt_callback():
            logging.info("main loop detect interrupt")
            exit(0)

        try:
            ack_status = workflow_ack_queue.get(block=True, timeout=0.1)
            num_data_received += 1
            if ack_status != target_workflow_status:
                logging.error(f"Workflow ack idx {num_data_received} error, expect {target_workflow_status.name}, but get {ack_status.name}")
                return False
            else:
                logging.info(f"Workflow received ack {target_workflow_status.name} idx {num_data_received}.")
                if num_data_received == num_data_send:
                    break
        except Empty:
            continue
    return True


def switch_workflow_for_predict_process_default(target_workflow_status, workflow_queue_list, workflow_ack_queue, tid_train_eval_pid_dict):
    logging.info(f"Start to switch to workflow status {target_workflow_status.name}")
    num_data_send = len(tid_train_eval_pid_dict)
    for qid, (tid, pid) in enumerate(tid_train_eval_pid_dict.items()):
        threading.Thread(target=send_signal_thread, args=(workflow_queue_list[pid], (tid, target_workflow_status), qid), daemon=True).start()

    if not receive_and_check_ack_from_queue(target_workflow_status, workflow_ack_queue, num_data_send):
        exit(-1)
    logging.info(f"Finished to switch to workflow status {target_workflow_status.name}")
    return target_workflow_status


# 左右互搏的玩家区分
# 如果修改玩家数量要修改这块
def get_player_name_model_dict(best_model_entity, new_model_entity):
    player_name0 = GET_PLAYER_NAME(0)
    player_name1 = GET_PLAYER_NAME(1)
    return {player_name0: best_model_entity,
            player_name1: new_model_entity}


# 左右互搏的玩家区分
# 如果修改玩家数量要修改这块
def get_new_model_player_name():
    return GET_PLAYER_NAME(1)


# 左右互搏的玩家区分
# 如果修改玩家数量要修改这块
def map_action_bin_to_actual_action_and_value(action_bin, action_value_or_ranges_list, acting_player_value_left, current_round_acting_player_historical_value, small_blind):
    assert action_value_or_ranges_list[action_bin] is not None, ValueError(f'None action_bin choice:{action_bin}, while action_value_or_ranges_list={action_value_or_ranges_list}')
    action, action_value_or_range = action_value_or_ranges_list[action_bin]
    if isinstance(action_value_or_range, int):
        return action, action_value_or_range
    elif isinstance(action_value_or_range, tuple):
        # 重要：把随机切分的下注比例映射到small_blind的整数倍，int型
        range_start, range_end = action_value_or_range
        choice_proportion = range_start + (range_end - range_start) * random.random()
        delta_value_choice = acting_player_value_left * choice_proportion
        multiple_of_small_bind = round(delta_value_choice / small_blind)
        delta_action_value = multiple_of_small_bind * small_blind
        action_value = current_round_acting_player_historical_value + delta_action_value
        return action, action_value
    else:
        raise ValueError(f'Error action_bin choice:{action_bin}, while action_value_or_ranges_list={action_value_or_ranges_list}')


# 左右互搏的玩家区分
# 如果修改玩家数量要修改这块
def map_action_bin_to_actual_action_and_value_v2(action_bin, action_value_or_ranges_list, acting_player_value_left, current_round_acting_player_historical_value, delta_min_value_to_raise, small_blind):
    assert action_value_or_ranges_list[action_bin] is not None, ValueError(f'None action_bin choice:{action_bin}, while action_value_or_ranges_list={action_value_or_ranges_list}')
    action, action_value_or_range = action_value_or_ranges_list[action_bin]
    if isinstance(action_value_or_range, int):
        return action, action_value_or_range
    elif isinstance(action_value_or_range, tuple):
        # 重要：把随机切分的下注比例映射到small_blind的整数倍，int型
        range_start, range_end = action_value_or_range
        choice_proportion = range_start + (range_end - range_start) * random.random()
        delta_value_choice = (acting_player_value_left - delta_min_value_to_raise) * choice_proportion
        delta_action_value = GET_VALID_BET_VALUE(delta_value_choice, small_blind)
        action_value = delta_min_value_to_raise + delta_action_value + current_round_acting_player_historical_value
        return action, action_value
    else:
        raise ValueError(f'Error action_bin choice:{action_bin}, while action_value_or_ranges_list={action_value_or_ranges_list}')


def get_num_feature_bins_v2():
    num_bin = 0
    for player_action, (range_start, range_end) in ACTION_BINS_DICT:
        if player_action == PlayerActions.CHECK_CALL or (player_action == PlayerActions.RAISE and range_start < 1.):
            num_bin += 1
    return num_bin


def log_inference(observation_list, action_probs_list, action_Qs_list, winning_prob_list, model, num_data_print_per_inference):
    with torch.no_grad():
        observation_array = np.array(observation_list)
        observation_tensor = torch.tensor(observation_array, dtype=torch.int32, device=model.device, requires_grad=False)
        predict_action_probs_logits_tensor, predict_action_Qs_tensor, predict_winning_prob_tensor = model(observation_tensor)
        predict_action_probs_tensor = torch.softmax(predict_action_probs_logits_tensor, dim=1)
        predict_action_probs_list = predict_action_probs_tensor.cpu().numpy()
        predict_action_Qs_list = predict_action_Qs_tensor.cpu().numpy()
        predict_winning_prob_list = predict_winning_prob_tensor.cpu().numpy()
    for data_idx, (action_probs, action_Qs, winning_prob, predict_action_probs, predict_action_Qs, predict_winning_prob) in enumerate(zip(action_probs_list, action_Qs_list, winning_prob_list, predict_action_probs_list, predict_action_Qs_list, predict_winning_prob_list)):
        if data_idx >= num_data_print_per_inference:
            break
        logging.info(f'action_probs={",".join(["%.4f" % item for item in action_probs])}\npredict_action_probs={",".join(["%.4f" % item for item in predict_action_probs.tolist()])}\naction_Qs={",".join(["%.4f" % item for item in action_Qs])}\npredict_action_Qs={",".join(["%.4f" % item for item in predict_action_Qs.tolist()])}\nwinning_prob={winning_prob}\npredict_winning_prob={predict_winning_prob}\n')
