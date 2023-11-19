def get_state_dict_from_model(model):
    state_dict = model.state_dict()
    new_state_dict = dict()
    for key, state_tensor in state_dict.items():
        new_state_dict[key] = state_tensor.clone().cpu()
    return new_state_dict
