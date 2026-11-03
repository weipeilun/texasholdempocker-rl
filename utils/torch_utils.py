import math
import torch


# 初始化时，一个 pre-norm FFN 子层写入残差流的幅度（RMS）。
# nn.TransformerEncoderLayer 默认 ReLU FFN，nn.Linear 默认初始化 U(±1/sqrt(fan_in))，即方差 1/(3·fan_in)：
#   u = LayerNorm(x)          每个元素方差 1
#   h = W1·u                  方差 d · 1/(3d) · 1 = 1/3（偏置贡献 1/(3d)，可忽略）
#   relu(h)                   E[relu(h)^2] = 1/6
#   W2·relu(h)                方差 ff · 1/(3ff) · 1/6 = 1/18
# 与 d_model、dim_feedforward 都无关，RMS = sqrt(1/18) ≈ 0.236。
PRE_NORM_FFN_WRITE_RMS = math.sqrt(1. / 18.)


def resolve_embedding_init_rms(init_rms):
    """'auto' -> PRE_NORM_FFN_WRITE_RMS；None -> None（保留 PyTorch 默认 N(0, 1)）；数值 -> 原值。

    embedding 是残差流的第一笔写入。按 'auto' 初始化时，embedding 的幅度与之后每个子层的单次写入同一量级，
    初始化时两者都不占主导。
    """
    if init_rms == 'auto':
        return PRE_NORM_FFN_WRITE_RMS
    if init_rms is None:
        return None
    return float(init_rms)


def _clone_state_dict(state_dict):
    if isinstance(state_dict, dict):
        new_state = dict()
        for key, state in state_dict.items():
            if isinstance(state, torch.Tensor):
                new_state[key] = state.clone().cpu()
            elif isinstance(state, (dict, list, tuple)):
                new_state[key] = _clone_state_dict(state)
            else:
                new_state[key] = state
    else:
        new_state = list()
        for state in state_dict:
            if isinstance(state, torch.Tensor):
                new_state.append(state.clone().cpu())
            elif isinstance(state, (dict, list, tuple)):
                new_state.append(_clone_state_dict(state))
            else:
                new_state.append(state)
    return new_state


def get_state_dict_from_model(model):
    state_dict = model.state_dict()
    return _clone_state_dict(state_dict)


def get_optimizer_state_dict_from_model(model):
    optimizer_state_dict = model.optimizer.state_dict()
    return _clone_state_dict(optimizer_state_dict)


def print_model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 12 + 'weight shape dtype' + ' ' * 12 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape) + ', ' + str(w_variable.dtype)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
