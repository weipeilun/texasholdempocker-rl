import numpy as np
import random
from env.constants import ChoiceMethod


def choose_idx_by_array(array, method):
    new_array = np.array(array)
    if method == ChoiceMethod.ARGMAX:
        return int(np.argmax(new_array).tolist())
    elif method == ChoiceMethod.PROBABILITY:
        sum_probs = sum(new_array)
        new_array /= sum_probs

        random_num = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(new_array):
            cumulative_prob += prob
            if random_num <= cumulative_prob:
                return i
        raise ValueError(f"array should be a probability distribution, but={array}")
