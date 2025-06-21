# analyser_interface.py
from abc import ABC, abstractmethod

class AnalyserTrainer(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def split_data(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def validate_model(self):
        pass

    @abstractmethod
    def export_model(self, save_path: str):
        pass

    def run_pipeline(self, save_path: str):
        self.load_data()
        self.preprocess()
        self.train_model()
        self.export_model(save_path)
