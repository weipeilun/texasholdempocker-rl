def get_state_dict_from_model(model):
    state_dict = model.state_dict()
    new_state_dict = dict()
    for key, state_tensor in state_dict.items():
        if not state_tensor.requires_grad:
            continue
        new_state_dict[key] = state_tensor.detach().cpu()
    return state_dict
