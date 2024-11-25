from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    @abstractmethod
    def preprocess(self, data):
        pass
    
    @abstractmethod
    def augment(self, data):
        pass 