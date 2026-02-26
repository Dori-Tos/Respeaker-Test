
class DOAEstimator:
    def __init__(self):
        self.frozen: bool = False
    
    def freeze(self):
        self.frozen = True
    
    def unfreeze(self):
        self.frozen = False
    
    @property
    def is_frozen(self):
        return self.frozen
    
    
class IterativeDOAEstimator(DOAEstimator):
    def __init__(self):
        super().__init__()