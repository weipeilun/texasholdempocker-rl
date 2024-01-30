import numpy as np
import random
from env.constants import ChoiceMethod


def choose_idx_by_array(array, method):
    if method == ChoiceMethod.ARGMAX:
        return int(np.argmax(array).tolist())
    elif method == ChoiceMethod.PROBABILITY:
        sum_probs = sum(array)

        random_num = random.random() * sum_probs
        cumulative_prob = 0
        for i, prob in enumerate(array):
            cumulative_prob += prob
            if random_num <= cumulative_prob:
                return i
        raise ValueError(f"array should be a probability distribution, but={array}")


def softmax_np(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)