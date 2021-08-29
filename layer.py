# Base class
class Layer:
    def __init__(self, is_residualbranch = False):
        self.is_residualbranch = is_residualbranch

    def forward_propagation(self, input):
        raise NotImplementedError