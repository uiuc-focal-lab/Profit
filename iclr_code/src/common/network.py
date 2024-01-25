from enum import Enum
from copy import deepcopy

class Network(list):
    def __init__(self, net_name=None, input_name=None, input_shape=None, torch_net=None):
        super().__init__()
        self.net_name = net_name
        self.input_name = input_name
        self.input_shape = input_shape
        self.torch_net = torch_net


class Layer:
    def __init__(self, weight=None, bias=None, type=None):
        self.weight = deepcopy(weight)
        self.bias = deepcopy(bias)
        self.type = type


class LayerType(Enum):
    Conv2D = 1
    Linear = 2
    ReLU = 3
    Flatten = 4
    MaxPool1D = 5
    Normalization = 6
    NoOp = 7