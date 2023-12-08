import logging
import random

import yaml
import argparse
import numpy as np


def update_concurrent(target_dict, new_dict):
    for key, value in new_dict.items():
        if key in target_dict:
            if isinstance(value, dict):
                update_concurrent(target_dict[key], new_dict[key])
            else:
                target_dict[key] = new_dict[key]
        else:
            target_dict[key] = value
    return target_dict


def choose_params_concurrent(param_dict):
    new_dict = dict()
    for param_key, param_value in param_dict.items():
        if isinstance(param_value, list):
            value_choice = random.choice(param_value)
            new_dict[param_key] = value_choice
            logging.info(f'chosen {param_key}: {value_choice}')
        elif isinstance(param_value, dict):
            new_dict[param_key] = choose_params_concurrent(param_value)
        else:
            new_dict[param_key] = param_value
    return new_dict


def parse_params():
    """
    从命令行解析参数并返回参数对象

    Args:
        无

    Returns:
        dict: 包含所有配置参数的字典对象

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'debug', 'test', 'test_train'], help='train/debug/test/test_train', required=True)
    args = parser.parse_args()
    task_param_path_dict = {
        'train': 'config/train.yml',
        'debug': 'config/debug.yml',
        'test': 'config/test.yml',
        'test_train': 'config/test_train.yml',
    }

    default_params = yaml.load(open('config/default.yml', encoding="UTF-8"), Loader=yaml.FullLoader)
    task_params = yaml.load(open(task_param_path_dict[args.mode], encoding="UTF-8"), Loader=yaml.FullLoader)
    params = update_concurrent(default_params.copy(), task_params.copy())
    return args, params


def parse_test_train_params():
    default_params = yaml.load(open('config/default.yml', encoding="UTF-8"), Loader=yaml.FullLoader)
    task_params = yaml.load(open('config/test_train.yml', encoding="UTF-8"), Loader=yaml.FullLoader)
    return default_params, task_params
