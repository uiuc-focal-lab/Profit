from src.training.models.resnet import model_resnet
from src.training.models.feedforward import *
from src.training.models.resnext import *
from src.training.models.resnext_imagenet64 import *
from src.training.models.densenet import *
from src.training.models.mobilenet import *
from src.training.models.densenet_no_bn import *
from src.training.models.densenet_imagenet import *
from src.training.models.wide_resnet_imagenet64 import *
from src.training.models.wide_resnet_cifar import *
from src.training.models.resnet18 import *
from src.training.models.vnncomp_resnet import resnet2b as vnncomp_resnet2b, resnet4b as vnncomp_resnet4b
from src.training.models.model_defs import mnist_6_100, mnist_conv_small, mnist_conv_big

Models = {
    'mlp_2layer': mlp_2layer,
    'mlp_3layer': mlp_3layer,
    'mlp_3layer_weight_perturb': mlp_3layer_weight_perturb,
    'mlp_5layer': mlp_5layer,
    'cnn_4layer': cnn_4layer,
    'cnn_6layer': cnn_6layer,
    'cnn_7layer': cnn_7layer,
    'cnn_7layer_bn': cnn_7layer_bn,
    'cnn_7layer_bn_imagenet': cnn_7layer_bn_imagenet,
    'resnet': model_resnet,
    'resnet18': ResNet18,
    'ResNeXt_cifar': ResNeXt_cifar,
    'ResNeXt_imagenet64': ResNeXt_imagenet64,
    'Densenet_cifar_32': Densenet_cifar_32,
    'Densenet_cifar_wobn': Densenet_cifar_wobn,
    'Densenet_imagenet': Densenet_imagenet,
    'MobileNet_cifar': MobileNetV2,
    'wide_resnet_cifar': wide_resnet_cifar,
    'wide_resnet_cifar_bn': wide_resnet_cifar_bn,
    'wide_resnet_cifar_bn_wo_pooling': wide_resnet_cifar_bn_wo_pooling,
    'wide_resnet_cifar_bn_wo_pooling_dropout': wide_resnet_cifar_bn_wo_pooling_dropout,
    'wide_resnet_imagenet64': wide_resnet_imagenet64,
    'wide_resnet_imagenet64_1000class': wide_resnet_imagenet64_1000class,
    'vnncomp_resnet2b': vnncomp_resnet2b,
    'vnncomp_resnet4b': vnncomp_resnet4b,
    'cifar_base_kw': cifar_base_kw,
    'mnist_6_100_nat': mnist_6_100,
    'mnist_conv_small_nat': mnist_conv_small,
    'mnist_conv_big_diffai': mnist_conv_big,
}
