from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self):
        self.state = None
        self.reward = None
        self.done = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def close(self):
        pass
