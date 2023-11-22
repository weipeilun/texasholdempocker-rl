from env.workflow import *
from tools.param_parser import *
from tools import counter
from queue import Queue
from torch.multiprocessing import Manager, Process, Condition


def train_data_generator(filename):
    f = open(filename, 'r')
    while True:
        if interrupt.interrupt_callback():
            logging.info("train_data_generator detect interrupt")
            break

        feature_str = f.readline()
        target_probs_str = f.readline()
        target_value_str = f.readline()
        f.readline()

        if feature_str == '':
            f = open(filename, 'r')
            continue

        modified_feature_list = [int(feature) for feature in feature_str.rstrip().split(',')]
        modified_target_probs_list = [float(prob_str) for prob_str in target_probs_str.split(',')]
        modified_target_value = float(target_value_str)
        yield modified_feature_list, modified_target_probs_list, modified_target_value
    f.close()


def data_batch_generator(filename, batch_size=8):
    data_generator = train_data_generator(filename)

    feature_list_buffer = list()
    target_probs_list_buffer = list()
    target_value_buffer = list()
    while True:
        if interrupt.interrupt_callback():
            logging.info("data_batch_generator detect interrupt")
            break

        feature_list, target_probs_list, target_value = data_generator.__next__()
        feature_list_buffer.append(feature_list)
        target_probs_list_buffer.append(target_probs_list)
        target_value_buffer.append(target_value)
        if len(feature_list_buffer) == batch_size:
            feature_list_array = np.asarray(feature_list_buffer, dtype=np.int32)
            yield feature_list_array, target_probs_list_buffer, target_value_buffer
            feature_list_buffer = list()
            target_probs_list_buffer = list()
            target_value_buffer = list()


if __name__ == '__main__':
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    params = parse_params()

    # 用户所有连续特征分桶embedding
    num_bins = params['num_bins']
    model_param_dict = params['model_param_dict']
    model_test_checkpoint_path = params['model_test_checkpoint_path']
    model = AlphaGoZero(**model_param_dict)

    train_data_path = params['train_data_path']
    log_step_num = params['log_step_num']
    predict_step_num = params['predict_step_num']
    batch_size = params['model_param_dict']['batch_size']
    train_step_num = 0
    train_batch_gen = data_batch_generator(train_data_path, batch_size=batch_size)
    for observation_list, action_probs_list, winning_prob_list in train_batch_gen:
        action_probs_loss, winning_prob_loss = model.learn(observation_list, action_probs_list, winning_prob_list)
        train_step_num += 1

        if train_step_num % log_step_num == 0:
            logging.info(f'train_step {train_step_num}, action_probs_loss={action_probs_loss}, winning_prob_loss={winning_prob_loss}')

        if train_step_num % predict_step_num == 0:
            with torch.no_grad():
                action_probs_tensor, player_result_value_tensor = model(observation_list)
                predict_action_probs_list = action_probs_tensor.cpu().numpy()
                predict_player_result_value_list = player_result_value_tensor.cpu().numpy()
            for predict_action_probs, predict_player_result_value in zip(predict_action_probs_list, predict_player_result_value_list):
                logging.info(f'predic_action_probs={",".join(["%.4f" % item for item in predict_action_probs.tolist()])}, predic_winning_prob={predict_player_result_value.tolist()}')
