import random
import math
import numpy as np
from tools.memories.abstract_memory import AbstractMemory
from tools.sumtree import SumTree


class PrioritizedMemory(AbstractMemory):
    def __init__(self, capacity, sample_length, alpha=0.8, beta=0.6, beta_increment_per_sampling=0.001, epsilon=0.001):
        super(PrioritizedMemory, self).__init__(capacity, sample_length)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon

        self.st = SumTree(capacity, sample_length)

        self.abs_err_upper = 1.0

    def store(self, transition):
        all_p = self.st.leaves_p
        if len(all_p) == 0:
            max_p = self.abs_err_upper
        else:
            max_p = np.max(all_p)
        self.st.add(max_p, transition)

    def sample(self, n):
        sample_idx_np = np.zeros((n, ), dtype=np.int64)
        weight_np = np.zeros((n, 1), dtype=np.float32)
        sample_list = [None for _ in range(n)]
        sample_length = self.st.total_p / n
        p_min = np.min(self.st.leaves_p)

        if self.beta_increment_per_sampling is not None:
            self.beta = min(self.beta + self.beta_increment_per_sampling, 1)

        for i in range(n):
            sample_p = i * sample_length + random.random() * sample_length
            data_idx, p, sample = self.st.get_leaf(sample_p)
            sample_idx_np[i] = data_idx
            try:
                weight_np[i, 0] = math.pow(p / p_min, -self.beta)
            except:
                print(p)
                print(p_min)
                print(self.st.leaves_p)
                print(self.st.total_p)
                exit(-1)
            sample_list[i] = sample

        return sample_idx_np, weight_np, sample_list

    def batch_update(self, tree_idx_list, abs_error_np):
        abs_error_np = np.abs(abs_error_np)
        abs_error_np += self.epsilon
        # todo: rethink this
        abs_error_np = np.minimum(abs_error_np, self.abs_err_upper)
        abs_error_np = np.power(abs_error_np, self.alpha)

        updated_idx_set = set()
        for tree_idx, abs_error in zip(tree_idx_list, abs_error_np):
            if tree_idx not in updated_idx_set:
                self.st.update(tree_idx, abs_error)
                updated_idx_set.add(tree_idx)
