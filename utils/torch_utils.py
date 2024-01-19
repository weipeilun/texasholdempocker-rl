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
