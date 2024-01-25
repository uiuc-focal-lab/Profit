import torch
from src.AIDomains import *
import sys

def load_pynet_model(net_path):
    model = torch.load(net_path,  map_location=torch.device('cpu'))
    print(model)
    return model