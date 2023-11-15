import numpy as np


class SumTree(object):
    def __init__(self, capacity, sample_length):
        self.capacity = capacity
        # self.sample_length = sample_length

        self.node_buffer = [0 for _ in range(capacity * 2 - 1)]
        self.data_buffer = [None for _ in range(capacity)]
        # self.node_buffer = np.zeros((capacity * 2 - 1, ), dtype=np.float32)
        # self.data_buffer = np.zeros((capacity, sample_length), dtype=np.float32)
        self.current_data_idx = 0

        self.valid_data_num = 0

    def add(self, p, data):
        # if self.sample_length > 1:
        #     data = np.asarray(data, dtype=np.float32)
        self.data_buffer[self.current_data_idx] = list(data)
        self.update(self.current_data_idx, p)

        if isinstance(data[2], dict):
            reward_dict = data[2]
            num_players = len(reward_dict)
            modified_players_set = set()
            modify_data_idx = self.current_data_idx
            while True:
                modify_data = self.data_buffer[modify_data_idx]
                if modify_data is not None:
                    player_role = modify_data[0][1]
                    if player_role not in modified_players_set:
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

        if self.valid_data_num < self.capacity:
            self.valid_data_num += 1

    def update(self, data_idx, p):
        tree_idx = data_idx + self.capacity - 1
        p_gap = p - self.node_buffer[tree_idx]
        self.node_buffer[tree_idx] = p

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.node_buffer[tree_idx] += p_gap

    def get_leaf(self, v):
        node_idx = 1
        while node_idx < self.capacity - 1:
            if v < self.node_buffer[node_idx]:
                node_idx = node_idx * 2 + 1
            else:
                v -= self.node_buffer[node_idx]
                node_idx = (node_idx + 1) * 2 + 1
        if v >= self.node_buffer[node_idx]:
            node_idx += 1
        data_idx = node_idx - self.capacity + 1
        return data_idx, self.node_buffer[node_idx], self.data_buffer[data_idx]

    @property
    def total_p(self):
        return self.node_buffer[0]

    @property
    def leaves_p(self):
        return self.node_buffer[self.capacity - 1: self.capacity - 1 + self.valid_data_num]


if __name__ == '__main__':
    st = SumTree(8, 2)
    for i in range(1, 20):
        print(st.leaves_p)
        st.add(i, np.asarray((i, i)))
    print(st.leaves_p)
    print(st.total_p)
    print(st.get_leaf(0))
    print(st.get_leaf(123))
