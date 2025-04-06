import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class F1Predictor:
    def __init__(self):
        self.data = None
        self.model = None
    
    def load_data(self, filepath):
        """Load F1 race data"""
        self.data = pd.read_csv(filepath)
        return self
    
    def preprocess_data(self):
        """Prepare F1 data for modeling"""
        pass
    
    def train_model(self):
        """Train model to predict F1 race outcomes"""
        pass

if __name__ == "__main__":
    predictor = F1Predictor()