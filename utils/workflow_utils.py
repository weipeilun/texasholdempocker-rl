import logging
from queue import Empty
from env.constants import GET_PLAYER_NAME
from tools import interrupt
import torch
import numpy as np


def map_train_thread_to_queue_info_train(thread_id, queue_info_list):
    num_in_queue = len(queue_info_list)
    in_queue_idx = thread_id % num_in_queue
    return queue_info_list[in_queue_idx]


def map_train_thread_to_queue_info_eval(thread_id, queue_info_list):
    num_in_queue = len(queue_info_list)
    num_in_queue_group = num_in_queue // 2
    in_queue_group_idx = thread_id % num_in_queue_group
    return queue_info_list[in_queue_group_idx], queue_info_list[in_queue_group_idx + num_in_queue_group]


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
    for workflow_queue in workflow_queue_list:
        workflow_queue.put(target_workflow_status)
    if not receive_and_check_all_ack(target_workflow_status, workflow_ack_queue_list):
        exit(-1)
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
