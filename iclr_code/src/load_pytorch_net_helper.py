import torch
import torch.nn as nn
from torch.nn import Flatten

# CNN, relatively small 3-layer
# parameter in_ch: input image channel, 1 for MNIST and 3 for CIFAR
# parameter in_dim: input dimension, 28 for MNIST and 32 for CIFAR
# parameter kernel_size: convolution kernel size, 3 or 5
# parameter width: width multiplier
def model_cnn_3layer_fixed(in_ch, in_dim, kernel_size, width, linear_size = None):
    if linear_size is None:
        linear_size = width * 64
    if kernel_size == 5:
        h = (in_dim - 4) // 4
    elif kernel_size == 3:
        h = in_dim // 4
    else:
        raise ValueError("Unsupported kernel size")
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, kernel_size=4, stride=4, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*h*h, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def model_cnn_2layer(in_ch, in_dim, width, linear_size=128): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

def model_cnn_2layer_modified(in_ch, in_dim, width, linear_size=100):
    
    model = nn.Sequential(
        nn.Conv2d(in_ch, 16*2, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*2*(in_dim // 4)*(in_dim // 4), linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

def load_pth_model_modified(path):
    model_structure = model_cnn_2layer_modified(in_ch=1, in_dim=28, width=1, linear_size=100)
    dict_n = torch.load(path,  map_location=torch.device('cpu'))
    model_state_dict = dict_n['state_dict']
    model_structure.load_state_dict(model_state_dict)
    return model_structure

def load_pt_model_modified(path):
    model_structure = model_cnn_2layer_modified(in_ch=1, in_dim=28, width=1, linear_size=100)
    dict_n = torch.load(path,  map_location=torch.device('cpu'))
    model_structure.load_state_dict(dict_n)
    return model_structure


def load_pth_model(path, param_dict):
    if "in_ch" in param_dict.keys():
        in_channel = param_dict["in_ch"]
    else:
        raise ValueError("In channel is missing")
    if "in_dim" in param_dict.keys():
        in_dimension = param_dict["in_dim"]
    else:
        raise ValueError("In dimension is missing")
    if "linear_size" in param_dict.keys():
        linear_size = param_dict["linear_size"]
    else:
        raise ValueError("In kernel size is missing")
    if "width" in param_dict.keys():
        width = param_dict["width"]
    else:
        raise ValueError("In width is missing")
    model_structure = model_cnn_2layer(in_ch=in_channel, in_dim=in_dimension, width=width, linear_size=linear_size)
    dict_n = torch.load(path,  map_location=torch.device('cpu'))
    model_state_dict = dict_n['state_dict']
    model_structure.load_state_dict(model_state_dict)
    return model_structure


# if __name__ == '__main__':
#     net_name = 'cifar_cnn_2layer_width_2_best.pth'
#     path = dir_path + net_name
#     param_dict =   {"in_ch": 3, "in_dim": 32, "width": 2, "linear_size": 256}
#     model = load_pth_model(path, param_dict)
#     print(model)