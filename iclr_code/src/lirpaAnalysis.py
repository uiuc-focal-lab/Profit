import torch

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import sys
import time
sys.path.append('../')


from src.network_conversion_helper import get_pytorch_net
from src.common import Status
from src import util
from src.util import get_linear_layers, get_gradient_based_sparsification_index
from src.sparsification_util import *
from copy import deepcopy
from src.common import Domain, FeaturePriority
import random
from src.common.dataset import Dataset
from auto_LiRPA.operators import BoundLinear, BoundConv, BoundRelu


class LirpaTransformer:
    def __init__(self, domain, dataset):
        """"
        prop: Property for verification
        """
        self.domain = domain
        self.dataset = dataset

        if domain == Domain.LIRPA_IBP:
            self.method = 'CROWN-IBP'
        elif domain == Domain.LIRPA_CROWN_IBP:
            self.method = 'backward'
        elif domain == Domain.LIRPA_CROWN:
            self.method = 'CROWN'
        elif domain == Domain.LIRPA_CROWN_OPT:
            self.method = 'CROWN-Optimized'
        elif domain == Domain.LIRPA_CROWN_FORWARD:
            self.method = 'Forward+Backward'

        self.model = None
        self.ilb = None
        self.iub = None
        self.input = None
        self.out_spec = None
        self.batch_size = None
        self.prop = None
        self.args = None
        self.final_coef_mat = None
        self.experiment_topk_success_var = None
        self.baseline_topk_success_var = None
        self.problem_lb = None
        self.experiment_topk_error_var = None
        self.baseline_topk_error_var = None
        self.sparsification_result = None

    def get_sparsification_result(self):
        return self.sparsification_result

    def build(self, net, prop, relu_mask=None):
        self.ilb = util.reshape_input(prop.input_props[-1].input_lb, self.dataset)
        self.iub = util.reshape_input(prop.input_props[-1].input_ub, self.dataset)
        self.input = (self.ilb + self.iub) / 2
        self.batch_size = self.input.shape[0]
        self.model = BoundedModule(net, torch.empty_like(self.input), device=prop.input_props[-1].input_lb.device)
        self.out_spec = prop.out_constr.constr_mat[0].T.unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.prop = prop

    def is_linear(self, net_name):
        if 'mnist-net_256x2.onnx' in net_name:
            return True
        else:
            return False

    def get_modified_constraint_and_bias(self, pruned_linear_layer, constraint_mat):
        weight = pruned_linear_layer.weight.T
        bias = pruned_linear_layer.bias
        new_constraint = weight @ constraint_mat
        new_bias = bias @ constraint_mat
        return new_constraint, new_bias

    def get_experiment_topk_error(self):
        return self.experiment_topk_error_var

    def get_baseline_topk_error(self):
        return self.baseline_topk_error_var

    def check_experiment_topk_success(self):
        return self.experiment_topk_success_var

    def check_baseline_topk_success(self):
        return self.baseline_topk_success_var


    def experimental_indices_to_prune(self, topk, priority, zero_feature_indices, 
                                      nonzero_feature_indices, final_layer):
        if priority is FeaturePriority.Random:
            initial_sparsity = nonzero_feature_indices.size()[0]
            inidices_to_prune_count = max(0, initial_sparsity - topk)
            if inidices_to_prune_count == 0:
                return []
            else:
                indices_to_prune = random.sample(sorted(nonzero_feature_indices), k=inidices_to_prune_count)
            return indices_to_prune
        elif priority is FeaturePriority.Gradient:
            sorted_gradient_norm_indices = get_gradient_based_sparsification_index(self.final_coef_mat, 
                                                                                   zero_feature_indices,
                                                                                   self.args.topk_priority_norm)
            indices_to_prune = sorted_gradient_norm_indices[:-topk]
            return indices_to_prune
        elif priority in [FeaturePriority.Weight_ABS, FeaturePriority.Weight_SIGN]:
            weight_sorted_indices = get_sparsification_indices_weight(final_layer_wt=final_layer.weight,
                                                                      const_mat=self.prop.out_constr.constr_mat[0],
                                                                      zero_feature_indices=zero_feature_indices,
                                                                      priority=priority)
            indices_to_prune = weight_sorted_indices[:-topk]
            return indices_to_prune
        else:
            raise ValueError(f'Unrecognised feature priority {priority}')


    def compute_experiment_topk_success(self, net, zero_feature_indices, nonzero_feature_indices, 
                                    final_layer):
        if self.args.topk is None:
            self.experiment_topk_success_var = None
            return
        priority = self.args.topk_priority
        indices_to_prune = self.experimental_indices_to_prune(topk=self.args.topk, priority=priority,
                                                              zero_feature_indices=zero_feature_indices,
                                                              nonzero_feature_indices=nonzero_feature_indices,
                                                              final_layer=final_layer)
        # Pop the last layer from the nets
        _ = net.pop()  
        final_layer_copy = deepcopy(final_layer)
        prune_last_layer(final_layer_copy.weight, zero_feature_indices)
        prune_last_layer(final_layer_copy.weight, indices_to_prune)
        net.append(final_layer_copy)
        pytorch_model = get_pytorch_net(model=net, remove_last_layer=False,
                                                    all_linear=self.is_linear(self.args.net))
        
        self.build(net=pytorch_model, prop=self.prop)
        lb = self.compute_lb(C=self.out_spec)
        if self.problem_lb is not None:
            err = abs(self.problem_lb - lb) / (abs(self.problem_lb) + 1e-9)
            self.experiment_topk_error_var = err
        if lb >= 0:
            self.experiment_topk_success_var = True
        else:
            self.experiment_topk_success_var = False


    def compute_baseline_topk_success(self, net, zero_feature_indices, nonzero_feature_indices, 
                                    final_layer):
        if self.args.topk is None:
            self.baseline_topk_success_var = None
            return
        # Pop the last layer from the net
        _ = net.pop()        
        initial_sparsity = nonzero_feature_indices.size()[0]
        inidices_to_prune_count = initial_sparsity - self.args.topk
        final_layer_copy = deepcopy(final_layer)
        prune_last_layer(final_layer_copy.weight, zero_feature_indices)
        prune_last_layer(final_layer_copy.weight, nonzero_feature_indices[:inidices_to_prune_count])
        net.append(final_layer_copy)
        pytorch_model = get_pytorch_net(model=net, remove_last_layer=False,
                                                    all_linear=self.is_linear(self.args.net))
        
        self.build(net=pytorch_model, prop=self.prop)
        lb = self.compute_lb(C=self.out_spec)
        if self.problem_lb is not None:
            err = abs(self.problem_lb - lb) / (abs(self.problem_lb) + 1e-9)
            self.baseline_topk_error_var = err
        if lb >= 0:
            self.baseline_topk_success_var = True
        else:
            self.baseline_topk_success_var = False

    #  Not updated update before use
    def extract_abstract_features(self, net, zero_feature_indices, nonzero_feture_indices, 
                                    final_layer):
        print("Final layer shape", final_layer.weight.shape)        
        initial_sparsity = nonzero_feture_indices.size()[0]
        pruned_feture_count = 0
        l = 0
        r = initial_sparsity - 1
        # Compute the model without the last layer.
        while l <= r:
            mid = (l + r) // 2
            if mid <= 0:
                break
            # Pop the last layer from the net
            _ = net.pop()
            final_layer_copy = deepcopy(final_layer)
            indices_to_prune = nonzero_feture_indices[:mid]
            prune_last_layer(final_layer_copy.weight, zero_feature_indices)            
            prune_last_layer(final_layer_copy.weight, indices_to_prune)
            net.append(final_layer_copy)
            pytorch_model = get_pytorch_net(model=net, remove_last_layer=False,
                                                        all_linear=self.is_linear(self.args.net))
            self.build(net=pytorch_model, prop=self.prop)
            lb = self.compute_lb(C=self.out_spec)
            if lb >= 0:
                pruned_feture_count = max(pruned_feture_count, mid)
                l = mid + 1
            else:
                r = mid - 1
        optimal_sparsity = initial_sparsity - pruned_feture_count
        return optimal_sparsity

    def restore_net(self, net, final_layer):
        _ = net.pop()
        net.append(final_layer)
        return net

    def compute_sparse_features(self, net, f_lb, f_ub):
        linear_layers = get_linear_layers(net)
        final_layer = linear_layers[-1]
        nozero_count, zero_feature_indices, nonzero_feature_indices = get_sparsification_indices(f_lb, 
                                        f_ub, final_layer.weight, self.prop.out_constr.constr_mat[0],
                                        priority_heuristic=self.args.priority_heuristic)
        optimal_sparsity = self.extract_abstract_features(net=net, zero_feature_indices=zero_feature_indices,
                                                           nonzero_feture_indices=nonzero_feature_indices,
                                                           final_layer=final_layer)
        # restore the original network given it can be modified already.
        net = self.restore_net(net=net, final_layer=final_layer)
        self.compute_experiment_topk_success(net=net, zero_feature_indices=zero_feature_indices, 
                                             nonzero_feature_indices=nonzero_feature_indices, final_layer=final_layer)
        # restore the original network given it can be modified already.        
        net = self.restore_net(net=net, final_layer=final_layer)
        self.compute_baseline_topk_success(net=net, zero_feature_indices=zero_feature_indices, 
                                             nonzero_feature_indices=nonzero_feature_indices, final_layer=final_layer)
        self.sparsification_result = {}
        self.sparsification_result["Initial sparsity"] = nozero_count
        self.sparsification_result["Optimal Sparsity"] = optimal_sparsity
        self.sparsification_result["zero indices"] = zero_feature_indices
        self.sparsification_result["Indices prune"] = nonzero_feature_indices[:(nozero_count - optimal_sparsity)]
        self.sparsification_result["Remaining indices"] = nonzero_feature_indices[(nozero_count - optimal_sparsity):]
        return



    def verify_property(self, prop, args):
        with torch.no_grad():
            net = util.get_net(args.net, args.dataset)
            self.args = args
            self.sparsification_result = None
            # Temporary fix for avoiding sparsification.
            pytorch_model = get_pytorch_net(model=net, remove_last_layer=False, all_linear=self.is_linear(args.net))
            if args.enable_sparsification:
                pytorch_model_wt_last_layer = get_pytorch_net(model=net, remove_last_layer=True, all_linear=self.is_linear(args.net))
            else:
                pytorch_model_wt_last_layer = None
            f_lb = None
            f_ub = None      
            if args.enable_sparsification:
                self.build(net=pytorch_model_wt_last_layer, prop=prop)
                f_lb, f_ub = self.compute_lb_ub(C=None)
                f_lb = torch.squeeze(f_lb)
                f_ub = torch.squeeze(f_ub)
            
            self.build(net=pytorch_model, prop=prop)
            lb = self.compute_lb(C=self.out_spec)
            self.problem_lb = lb
            verification_result = Status.UNKNOWN
            print("Method", self.method)
            print("Lower bound", lb)
            if lb >= 0:
                verification_result = Status.VERIFIED
            else:
                return verification_result
            if args.enable_sparsification and verification_result == Status.VERIFIED:
                self.compute_sparse_features(net=net, f_lb=f_lb, f_ub=f_ub)
            return verification_result

    
    def compute_lb_ub(self, C=None):
        ptb = PerturbationLpNorm(x_L=self.ilb, x_U=self.iub)
        lirpa_input_spec = BoundedTensor(self.input, ptb)
        final_name = self.model.final_name
        # Take the first of the input names.
        print("Model input name", self.model.input_name[0])
        input_name = self.model.input_name[0]        
        olb, oub, A_dict = self.model.compute_bounds(x=(lirpa_input_spec,), method=self.method, 
                                             C=C, return_A=True,  needed_A_dict={ final_name: [input_name] })

        lA = A_dict[final_name][input_name]['lA']
        uA = A_dict[final_name][input_name]['uA']
        lA = torch.squeeze(lA)
        uA = torch.squeeze(uA)
        lA = lA.view((lA.shape[0], -1))
        uA = uA.view((lA.shape[0], -1))
        self.final_coef_mat = lA + uA     
        return olb, oub

    def compute_lb(self, C=None, complete=False):
        start_time = time.time()
        ptb = PerturbationLpNorm(x_L=self.ilb, x_U=self.iub)
        lirpa_input_spec = BoundedTensor(self.input, ptb)
        olb, _ = self.model.compute_bounds(x=(lirpa_input_spec,), method=self.method, C=C)
        olb = olb + self.prop.out_constr.constr_mat[1]

        if self.prop.input_props[-1].is_conjunctive():
            lb = torch.min(olb, dim=1).values
        else:
            lb = torch.max(olb, dim=1).values

        if complete:
            return lb, True, None
        else:
            return lb

    def get_all_bounds(self):
        lbs, ubs = [], []
        lbs.append(self.ilb)
        ubs.append(self.iub)
        for node_name, node in self.model._modules.items():
            if type(node) in [BoundLinear, BoundConv] and node_name != 'last_final_node':
                lbs.append(node.lower)
                lbs.append(torch.clamp(node.lower, min=0))
                ubs.append(node.upper)
                ubs.append(torch.clamp(node.upper, min=0))
        return lbs, ubs