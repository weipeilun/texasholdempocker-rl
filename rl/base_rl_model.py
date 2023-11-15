from abc import abstractmethod, ABC


class BaseRLModel(ABC):

    @abstractmethod
    def choose_action(self, observation):
        pass

    @abstractmethod
    def store_transition(self, *transition):
        pass

    @abstractmethod
    def learn(self):
        pass
