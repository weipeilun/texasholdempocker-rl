import yaml
import argparse


def _update_concurrent(target_dict, new_dict):
    for key, value in target_dict.items():
        if key in new_dict:
            if isinstance(value, dict):
                _update_concurrent(target_dict[key], new_dict[key])
            else:
                target_dict[key] = new_dict[key]
    return target_dict


def parse_params():
    """
    从命令行解析参数并返回参数对象

    Args:
        无

    Returns:
        dict: 包含所有配置参数的字典对象

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'debug'], help='train/debug', required=True)
    args = parser.parse_args()
    task_param_path_dict = {
        'train': 'config/train.yml',
        'debug': 'config/debug.yml',
    }

    default_params = yaml.load(open('config/default.yml', encoding="UTF-8"), Loader=yaml.FullLoader)
    task_params = yaml.load(open(task_param_path_dict[args.mode], encoding="UTF-8"), Loader=yaml.FullLoader)
    params = _update_concurrent(default_params.copy(), task_params.copy())
    return params
