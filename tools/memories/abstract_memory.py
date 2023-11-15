from abc import abstractmethod, ABC


class AbstractMemory(ABC):
    def __init__(self, capacity, sample_length):
        self.capacity = capacity
        self.sample_length = sample_length

    @abstractmethod
    def store(self, transition):
        pass

    @abstractmethod
    def sample(self, n):
        pass

    @abstractmethod
    def batch_update(self, tree_idx_list, abs_error_np):
        pass
