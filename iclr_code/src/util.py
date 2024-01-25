import csv
import os
import resource

import onnx2pytorch

import src.common as common
from time import gmtime, strftime
import onnx
import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from src.load_pytorch_net_helper import load_pth_model, load_pth_model_modified, load_pt_model_modified
from src.load_pynet_model_helper import load_pynet_model

import src.parse as parse
import src.training.models as models
from src.common.dataset import Dataset
from src.common import Domain
from src.domains.box import BoxTransformer
from src.domains.deeppoly import DeeppolyTransformer
from src.domains.deepz import ZonoTransformer
from src.domains.lptransformer import LPTransformer
from src.networks import FullyConnected, Conv
from src.common.network import LayerType

rt.set_default_logger_severity(3)

def get_custom_pth_nets():
    _custom_pth_netwoks = ['convMedGRELU__PGDK_w_0.3.pyt']
    return _custom_pth_netwoks

def get_pth_model_formats():
    _pth_model_format = {}
    _pth_model_format["cifar_cnn_3layer_fixed_kernel_3_width_1_best"] = {"in_ch": 3, "in_dim": 32, "kernel_size": 3, "width": 8}
    _pth_model_format["mnist_cnn_3layer_fixed_kernel_3_width_1_best"] = {"in_ch": 1, "in_dim": 28, "kernel_size": 3, "width": 1}
    _pth_model_format["cifar_cnn_2layer_width_2_best"] = {"in_ch": 3, "in_dim": 32, "width": 2, "linear_size": 256}
    _pth_model_format["mnist_cnn_2layer_width_1_best"] = {"in_ch": 1, "in_dim": 28, "width": 1, "linear_size": 128}
    return _pth_model_format

def get_pth_model_parameter(net_name):
    model_param_dict = get_pth_model_formats()
    if net_name not in  model_param_dict.keys():
        raise ValueError("Format corresponding to net name not preset")
    return model_param_dict[net_name]


# Compute the intersection of 2d list.
def compute_2dlist_intersection(list1, list2):
    if list1 is None:
        return list2
    if len(list1) != len(list2):
        print("List 1", len(list1))
        print("List 2", len(list2))
        raise ValueError("Both the list should be of same dimesion")
    intersection_list = []
    for i in range(len(list1)):
        intersection_list.append([])
        for x in list1[i]:
            if x in list2[i]:
                intersection_list[i].append(x)
    return intersection_list


def get_linear_layers(net):
    linear_layers = []
    for layer in net:
        if layer.type is not LayerType.Linear:
            continue
        linear_layers.append(layer)
    return linear_layers


def get_torch_net(net_file, dataset, device='cpu'):
    net_name = net_file.split('/')[-1].split('.')[-2]
    if 'pth' in net_file:
        if 'modified' in net_file:
           model = load_pth_model_modified(net_file)
           return model
        param_dict = get_pth_model_parameter(net_name)
        model = load_pth_model(net_file, param_dict)
        return model
    
    if 'pt' in net_file:
        model = load_pt_model_modified(net_file)
        return model

    if 'cpt' in net_file:
        return get_torch_test_net(net_name, net_file)

    if dataset == Dataset.MNIST:
        model = models.Models[net_name](in_ch=1, in_dim=28)
    elif dataset == Dataset.CIFAR10:
        model = models.Models[net_name](in_ch=3, in_dim=32)
    else:
        raise ValueError("Unsupported dataset")

    if 'kw' in net_file:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'][0])
    elif 'eran' in net_file:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'][0])
    else:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'])

    return model


def get_torch_test_net(net_name, path, device='cpu', input_size=28):
    if net_name == 'fc1':
        net = FullyConnected(device, input_size, [50, 10]).to(device)
    elif net_name == 'fc2':
        net = FullyConnected(device, input_size, [100, 50, 10]).to(device)
    elif net_name == 'fc3':
        net = FullyConnected(device, input_size, [100, 100, 10]).to(device)
    elif net_name == 'fc4':
        net = FullyConnected(device, input_size, [100, 100, 50, 10]).to(device)
    elif net_name == 'fc5':
        net = FullyConnected(device, input_size, [100, 100, 100, 10]).to(device)
    elif net_name == 'fc6':
        net = FullyConnected(device, input_size, [100, 100, 100, 100, 10]).to(device)
    elif net_name == 'fc7':
        net = FullyConnected(device, input_size, [100, 100, 100, 100, 100, 10]).to(device)
    elif net_name == 'conv1':
        net = Conv(device, input_size, [(16, 3, 2, 1)], [100, 10], 10).to(device)
    elif net_name == 'conv2':
        net = Conv(device, input_size, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(device)
    elif net_name == 'conv3':
        net = Conv(device, input_size, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(device)
    else:
        assert False

    net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return net.layers


def parse_spec(spec):
    with open(spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(spec[:-4].split('/')[-1].split('_')[-1])

    return true_label, pixel_values, eps


def sample(net_name, ilb, iub):
    print('Sample some output points:')
    sess = rt.InferenceSession(net_name)
    input_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {input_name: ilb.numpy().reshape(1, -1)})
    print('onnx output:', pred_onnx)
    pred_onnx = sess.run(None, {input_name: iub.numpy().reshape(1, -1)})
    print('onnx output2:', pred_onnx)
    pred_onnx = sess.run(None, {input_name: ((iub + ilb) / 2).numpy().reshape(1, -1)})
    print('onnx output3:', pred_onnx)


def check_adversarial(adv_ex, net, prop):
    """
    returns true if adv_ex is an adversarial example if following conditions hold
    1. net does not classify adv_ex to true_label.
    2. adv_ex lies within the ilb and iub. i.e. ilb <= adv_ex <= iub

    if @param adv_label_to_check is not None then we only check if the adv_ex is adversarial for that particular label
    """
    if adv_ex is not None:
        # sanity check adv example
        adv_ex = torch.tensor(adv_ex)

        num_err = 1e-5
        assert torch.max(prop.input_lb - adv_ex.flatten() - num_err).item() <= 0
        assert torch.max(adv_ex.flatten() - prop.input_ub - num_err).item() <= 0

        adv_ex = reshape_input(adv_ex, prop.dataset)
        print(adv_ex.shape)
        adv_label, out = compute_output_tensor(adv_ex, net)

        if prop.is_local_robustness():
            true_label = prop.get_label()
            print(out[true_label] - out)
            print('True label ', true_label, '  adv Label: ', adv_label)

            adv_label_to_check = None
            if true_label != adv_label and (adv_label_to_check is None or adv_label_to_check == adv_label):
                return True
        else:
            # TODO: The general check if the adversarial is real for any output constraint is not implemented
            return False
    return False


def reshape_input(adv_ex, dataset):
    if dataset == Dataset.MNIST:
        adv_ex = adv_ex.reshape(1, 1, 28, 28)
    elif dataset == Dataset.CIFAR10:
        adv_ex = adv_ex.reshape(1, 3, 32, 32)
    elif dataset == Dataset.ACAS:
        adv_ex = adv_ex.reshape(5)
    else:
        raise ValueError("Unknown dataset!")
    return adv_ex


def compute_output_tensor(inp, net):
    if net.net_format == 'pt':
        out = net.torch_net(inp)
        adv_label = torch.argmax(out)
        out = out.flatten()
    elif net.net_format == 'onnx':
        sess = rt.InferenceSession(net.net_name)
        inp = inp.reshape(net.input_shape)
        out = sess.run(None, {net.input_name: inp.numpy()})
        out = np.array(out)
        out = torch.tensor(out).flatten()
        adv_label = torch.argmax(out).item()
    else:
        raise ValueError("We only support torch and onnx!")

    return adv_label, out


def prepare_data(dataset, train=False, batch_size=100):
    if dataset == Dataset.CIFAR10:
        transform_test = transforms.Compose([
            transforms.ToTensor(), ])

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)

        inputs, _ = next(iter(testloader))
    elif dataset == Dataset.MNIST:
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=train, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0,), (1,))
                                       ])),
            batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Unsupported Dataset")
    return testloader


def ger_property_from_id(imag_idx, eps_temp, dataset):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_test = torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                                              transform=transforms.Compose([transforms.ToTensor(), normalize]))

    x, y = cifar_test[imag_idx]
    x = x.unsqueeze(0)

    # first check the model is correct at the input
    # y_pred = torch.max(model(x)[0], 0)[1].item()

    ilb = (x - eps_temp).flatten()
    iub = (x + eps_temp).flatten()

    return ilb, iub, torch.tensor(y)


def PGD(net, spec, label, steps=10):
    x_old = torch.reshape((spec.ilb + spec.iub) / 2, (1, 1, 28, 28))
    x = x_old.requires_grad_()

    net = get_torch_net('training/IBP_MNIST4/cnn_4layer.pt', 'mnist')

    # What should this be?
    learning_rate = 1e-4

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=learning_rate)

    for st in range(steps):
        y = net(x).flatten()

        if (torch.argmax(y).item() != label):
            return True, x

        loss = torch.mean(y[label] - y)

        print('CE found at:', st)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x = torch.min(x, spec.iub.reshape(1, 1, 28, 28))
        x = torch.max(x, spec.ilb.reshape(1, 1, 28, 28))

    return False, None


def get_net_format(net_name):
    net_format = None
    if 'pt' in net_name:
        net_format = 'pt'
    if 'onnx' in net_name:
        net_format = 'onnx'
    if 'pynet' in net_name:
        net_format = 'pynet'
    return net_format


def is_lirpa_domain(domain):
    lirpa_domains = [Domain.LIRPA_IBP, Domain.LIRPA_CROWN, Domain.LIRPA_CROWN_IBP,
                      Domain.LIRPA_CROWN_OPT, Domain.LIRPA_CROWN_FORWARD]
    if domain in lirpa_domains:
        return True
    return False


def get_domain_builder(domain):
    if domain == Domain.DEEPPOLY:
        return DeeppolyTransformer
    if domain == Domain.DEEPZ:
        return ZonoTransformer
    if domain == Domain.BOX:
        return BoxTransformer
    if domain == Domain.LP:
        return LPTransformer
    raise ValueError("Unexpected domain!")


def get_domain(transformer):
    if type(transformer) == DeeppolyTransformer:
        return Domain.DEEPPOLY
    if type(transformer) == ZonoTransformer:
        return Domain.DEEPZ
    if type(transformer) == BoxTransformer:
        return Domain.BOX
    if type(transformer) == LPTransformer:
        return Domain.LP
    raise ValueError("Unexpected domain!")


def prune_last_layer(weight, indices):
    sz = weight.size()
    for ind in indices:
        if ind < sz[1]:
            with torch.no_grad():
                weight[:, ind] = 0
        else:
            raise ValueError("Inidices out of range")

def get_net(net_name, dataset):
    net_format = get_net_format(net_name)
    if net_format == 'pt':
        # Load the model
        net_torch = get_torch_net(net_name, dataset)
        net = parse.parse_torch_layers(net_torch)
    elif net_format == 'pynet':
        net = load_pynet_model(net_path=net_name)
        net = parse.parse_torch_layers(net=net[0])
    elif net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        net = parse.parse_onnx_layers(net_onnx)
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name
    net.net_format = net_format
    return net


def get_sparsification_indices(f_lb, f_ub, final_layer_wt,
                            const_mat):
    out_constraint_mat = const_mat.T
    final_wt = out_constraint_mat @ final_layer_wt
    final_wt = torch.abs(final_wt)
    wt_bounds = torch.max(final_wt, dim=0)
    wt_bounds = wt_bounds[0]    
    abs_feature = torch.maximum(torch.abs(f_lb), torch.abs(f_ub))
    greedy_features = torch.mul(abs_feature, wt_bounds)
    sorted_features = torch.sort(greedy_features)
    nonzero_count = torch.count_nonzero(sorted_features[0])
    zero_fetures_indices = sorted_features[1][:-nonzero_count]
    nonzero_fetures_indices = sorted_features[1][-nonzero_count:]
    return nonzero_count, zero_fetures_indices, nonzero_fetures_indices


def get_gradient_based_sparsification_index(final_coef_mat, zero_indices, 
                                            topk_priority_norm=common.FeaturePriorityNorm.L2):
    if topk_priority_norm is common.FeaturePriorityNorm.L2:
        norms = torch.norm(final_coef_mat, dim=1)
    elif topk_priority_norm is common.FeaturePriorityNorm.L1:
        norms = torch.linalg.norm(final_coef_mat, ord=1, dim=1)
    elif topk_priority_norm is common.FeaturePriorityNorm.Linf:
        norms = torch.linalg.norm(final_coef_mat, ord=float('inf'), dim=1)
    norms[zero_indices] = -100.0
    sorted_norms = torch.sort(norms)
    return sorted_norms[1]

def log_memory_usage():
    mu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mu /= (1024*1024)
    os.makedirs(common.RESULT_DIR, exist_ok=True)
    file_name = common.RESULT_DIR + 'memory_usage.csv'
    with open(file_name, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(['Memory Usage at', strftime("%Y-%m-%d %H:%M:%S", gmtime())])
        writer.writerow([str(mu) + 'MBs'])

def onnx2torch(onnx_model):
    # find the input shape from onnx_model generally
    # https://github.com/onnx/onnx/issues/2657
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    net_feed_input = [node for node in onnx_model.graph.input if node.name in net_feed_input]

    if len(net_feed_input) != 1:
        # in some rare case, we use the following way to find input shape but this is not always true (collins-rul-cnn)
        net_feed_input = [onnx_model.graph.input[0]]

    onnx_input_dims = net_feed_input[0].type.tensor_type.shape.dim
    onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])

    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=False, debug=True)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())

    return pytorch_model, onnx_shape