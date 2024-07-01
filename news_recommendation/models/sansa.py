import gc
import torch


class EASE:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        
    def fit(self, X):
        X = X.to(self.device)
        
        pass
    
    def predict(self, X):
        pass