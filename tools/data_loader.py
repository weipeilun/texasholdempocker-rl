from tools import interrupt
import logging
import numpy as np


def train_data_generator(filename, epoch=1):
    current_epoch = 1
    f = open(filename, 'r')
    while True:
        if interrupt.interrupt_callback():
            logging.info("train_data_generator detect interrupt")
            break

        feature_str = f.readline()
        target_probs_str = f.readline()
        target_mask_str = f.readline()
        target_reward_value_str = f.readline()
        target_value_str = f.readline()
        f.readline()

        if feature_str == '' or target_probs_str == '' or target_mask_str == '' or target_reward_value_str == '' or target_value_str == '':
            if epoch < 0 or current_epoch < epoch:
                f = open(filename, 'r')
                current_epoch += 1
                continue
            else:
                break

        modified_feature_list = np.asarray([int(feature) for feature in feature_str.rstrip().split(',')], dtype=np.int32)
        modified_target_probs_list = [float(prob_str) for prob_str in target_probs_str.split(',')]
        modified_target_mask_list = [int(prob_str) for prob_str in target_mask_str.split(',')]
        modified_target_reward_value = float(target_reward_value_str)
        modified_target_winning_prob = float(target_value_str)
        yield modified_feature_list, modified_target_probs_list, modified_target_mask_list, modified_target_reward_value, modified_target_winning_prob
    f.close()


def data_batch_generator(filename, batch_size=8, epoch=1):
    data_generator = train_data_generator(filename, epoch=epoch)

    feature_list_buffer = list()
    target_probs_list_buffer = list()
    target_mask_buffer = list()
    target_reward_value_buffer = list()
    target_winning_prob_buffer = list()
    while True:
        if interrupt.interrupt_callback():
            logging.info("data_batch_generator detect interrupt")
            break

        feature_list, target_probs_list, target_mask_list, target_reward_value, target_winning_prob = data_generator.__next__()
        feature_list_buffer.append(feature_list)
        target_probs_list_buffer.append(target_probs_list)
        target_mask_buffer.append(target_mask_list)
        target_reward_value_buffer.append(target_reward_value)
        target_winning_prob_buffer.append(target_winning_prob)
        if len(feature_list_buffer) == batch_size:
            yield feature_list_buffer, target_probs_list_buffer, target_mask_buffer, target_reward_value_buffer, target_winning_prob_buffer
            feature_list_buffer = list()
            target_probs_list_buffer = list()
            target_mask_buffer = list()
            target_reward_value_buffer = list()
            target_winning_prob_buffer = list()
    data_generator.close()
