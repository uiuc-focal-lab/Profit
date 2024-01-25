import torch
from src.common import FeaturePriority, PriorityHeuristic
import random


def prune_last_layer(weight, indices):
    sz = weight.size()
    for ind in indices:
        if ind < sz[1]:
            with torch.no_grad():
                weight[:, ind] = 0
        else:
            raise ValueError("Inidices out of range")


def get_sparsification_indices(f_lb, f_ub, final_layer_wt,
                            const_mat, priority_heuristic=None):
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
    # print("default indices", nonzero_fetures_indices)
    if priority_heuristic is PriorityHeuristic.Default:
        return nonzero_count, zero_fetures_indices, nonzero_fetures_indices
    elif priority_heuristic is PriorityHeuristic.Gradient:
        nonzero_indices_ordering = get_sparsification_indices_weight(final_layer_wt=final_layer_wt,
                                                                     const_mat=const_mat, zero_feature_indices=zero_fetures_indices,
                                                                     priority=FeaturePriority.Weight_ABS) 
        # print("gradient indices", nonzero_indices_ordering[-nonzero_count:])
        return nonzero_count, zero_fetures_indices, nonzero_indices_ordering[-nonzero_count:]
    elif priority_heuristic is PriorityHeuristic.Random:
        randomized_idx = torch.randperm(nonzero_count)
        # print("random indices", nonzero_fetures_indices[randomized_idx])
        return nonzero_count, zero_fetures_indices, nonzero_fetures_indices[randomized_idx]
    else:
        raise ValueError(f"Unrecognized {priority_heuristic} Heuristic")

def get_sparsification_indices_weight(final_layer_wt, const_mat, zero_feature_indices, priority):
    out_constraint_mat = const_mat.T
    final_wt = out_constraint_mat @ final_layer_wt
    if priority is FeaturePriority.Weight_ABS:
        final_wt = torch.abs(final_wt)
    wt_bounds = torch.max(final_wt, dim=0)
    wt_bounds = wt_bounds[0]
    wt_bounds[zero_feature_indices] = -100.0
    sorted_wt_bounds = torch.sort(wt_bounds)
    return sorted_wt_bounds[1]