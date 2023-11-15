import random
import math
import numpy as np
from tools.memories.abstract_memory import AbstractMemory


class SimpleMemory(AbstractMemory):
    def __init__(self, capacity, sample_length):
        super(SimpleMemory, self).__init__(capacity, sample_length)

        self.transition_buffer = [None for _ in range(capacity)]
        self.current_data_idx: int = 0
        self.memory_full = False

    def store(self, transition):
        self.transition_buffer[self.current_data_idx] = list(transition)

        if isinstance(transition[2], dict):
            reward_dict = transition[2]
            num_players = len(reward_dict)
            modified_players_set = set()
            modify_data_idx = self.current_data_idx
            while True:
                modify_data = self.transition_buffer[modify_data_idx]
                if modify_data is not None:
                    player_role = modify_data[0][1]
                    if player_role not in modified_players_set and player_role in reward_dict:
                        modify_data[2] = reward_dict[player_role]
                        modified_players_set.add(player_role)
                        if len(modified_players_set) >= num_players:
                            break

                modify_data_idx -= 1
                if modify_data_idx == self.current_data_idx:
                    break
                if modify_data_idx < 0:
                    modify_data_idx = self.capacity - 1

        self.current_data_idx += 1
        if self.current_data_idx >= self.capacity:
            self.current_data_idx = 0
            self.memory_full = True

    def sample(self, n, randomize=True):
        sample_list = [None for _ in range(n)]
        weight_np = np.ones((n, 1), dtype=np.float32)

        if not self.memory_full and self.current_data_idx < self.capacity:
            max_choose_from = self.current_data_idx
        else:
            max_choose_from = self.capacity

        if randomize:
            sample_idx_np = np.random.choice(max_choose_from, size=n)
        else:
            sample_idx_np = np.arange(0, max_choose_from)

        for idx, sample_idx in enumerate(sample_idx_np):
            sample_list[idx] = self.transition_buffer[sample_idx]

        sample_idx_np_exp = np.expand_dims(sample_idx_np, axis=1)
        return sample_idx_np_exp, weight_np, sample_list

    def batch_update(self, tree_idx_list, abs_error_np):
        pass
