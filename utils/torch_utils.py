
def _clone_state_dict(state_dict):
    new_state_dict = dict()
    for key, state_tensor in state_dict.items():
        new_state_dict[key] = state_tensor.clone().cpu()
    return new_state_dict


def get_state_dict_from_model(model):
    state_dict = model.state_dict()
    return _clone_state_dict(state_dict)


def get_optimizer_state_dict_from_model(model):
    optimizer_state_dict = model.optimizer.state_dict()
    return _clone_state_dict(optimizer_state_dict)
