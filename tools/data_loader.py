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
        target_Qs_str = f.readline()
        target_value_str = f.readline()
        f.readline()

        if feature_str == '' or target_probs_str == '' or target_Qs_str == '' or target_value_str == '':
            if epoch < 0 or current_epoch < epoch:
                f = open(filename, 'r')
                current_epoch += 1
                continue
            else:
                break

        modified_feature_list = np.asarray([int(feature) for feature in feature_str.rstrip().split(',')], dtype=np.int32)
        modified_target_probs_list = [float(prob_str) for prob_str in target_probs_str.split(',')]
        modified_target_Qs_list = [float(prob_str) for prob_str in target_Qs_str.split(',')]
        modified_target_winning_prob = float(target_value_str)
        yield modified_feature_list, modified_target_probs_list, modified_target_Qs_list, modified_target_winning_prob
    f.close()


def data_batch_generator(filename, batch_size=8, epoch=1):
    data_generator = train_data_generator(filename, epoch=epoch)

    feature_list_buffer = list()
    target_probs_list_buffer = list()
    target_Qs_buffer = list()
    target_winning_prob_buffer = list()
    while True:
        if interrupt.interrupt_callback():
            logging.info("data_batch_generator detect interrupt")
            break

        feature_list, target_probs_list, target_Qs_list, target_winning_prob = data_generator.__next__()
        feature_list_buffer.append(feature_list)
        target_probs_list_buffer.append(target_probs_list)
        target_Qs_buffer.append(target_Qs_list)
        target_winning_prob_buffer.append(target_winning_prob)
        if len(feature_list_buffer) == batch_size:
            yield feature_list_buffer, target_probs_list_buffer, target_Qs_buffer, target_winning_prob_buffer
            feature_list_buffer = list()
            target_probs_list_buffer = list()
            target_Qs_buffer = list()
            target_winning_prob_buffer = list()
    data_generator.close()
