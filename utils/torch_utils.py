import torch


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
