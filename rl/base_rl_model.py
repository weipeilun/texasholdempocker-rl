from abc import abstractmethod, ABC


class BaseRLModel(ABC):

    @abstractmethod
    def store_transition(self, *transition):
        pass

    @abstractmethod
    def learn(self):
        pass
