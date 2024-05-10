import os
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import pickle

def get_stats(initial_sparsity, final_sparsity, filename, dir_name):
    fn = filename.replace(".dat", "")
    assert len(initial_sparsity) == len(final_sparsity)
    if len(initial_sparsity) == 0 or len(final_sparsity) == 0:
        return

    sparsity_ratio = []
    
    for i, x in enumerate(initial_sparsity):
        if final_sparsity[i] != 0:
            sparsity_ratio.append(x / final_sparsity[i])
    final_sparsity = np.array(final_sparsity)
    sparsity_ratio = np.array(sparsity_ratio)
    mean_sparsity_ration = np.mean(sparsity_ratio)
    median_sparsity_ration = np.median(sparsity_ratio)
    max_sparsity_ration = np.max(sparsity_ratio)
    min_sparsity_ration = np.min(sparsity_ratio)
    initial_sparsity_mean = np.mean(initial_sparsity)
    initial_sparsity_median = np.median(initial_sparsity)
    final_sparsity_mean = np.mean(final_sparsity)
    final_sparsity_median = np.median(final_sparsity)
    length = len(final_sparsity)
    length = math.ceil(length / 200 * 500)
    fn = dir_name + fn + "_stat.dat"
    file = open(fn, "w+")
    file.write("Proved properties {}\n".format(len(initial_sparsity)))
    file.write("Initial sparsity Mean {}\n".format(initial_sparsity_mean))
    file.write("Initial sparsity Median {}\n".format(initial_sparsity_median))
    file.write("Final sparsity Mean {}\n".format(final_sparsity_mean))
    file.write("Final sparsity Median {}\n".format(final_sparsity_median))        
    file.write("Mean {}\n".format(mean_sparsity_ration))
    file.write("Median {}\n".format(median_sparsity_ration))
    file.write("Max {}\n".format(max_sparsity_ration))     
    file.write("Min {}\n".format(min_sparsity_ration))
    less_than_5 = (final_sparsity <= 5).sum()
    less_than_10 = (final_sparsity <= 10).sum()
    file.write("less than 5 num {}\n".format(less_than_5))
    file.write("less than 10 num {}\n".format(less_than_10))
    file.write("less than 5 {}\n".format(less_than_5 / final_sparsity.shape[0] * 100))
    file.write("less than 10 {}\n".format(less_than_10 / final_sparsity.shape[0] * 100))
    file.close()

def get_name(dict, name):
    for k in dict.keys():
        if k in name:
            # print(name, dict[k])
            return dict[k]

def get_historgrams(initial_sparsity, final_sparsity, filename, dir_name, dict):
    fig, axs = plt.subplots(1, 1,
                        figsize =(3, 3),
                        tight_layout = True)
    if len(initial_sparsity) == 0 or len(final_sparsity) == 0:
        return
    max_initial_sparsity = max(initial_sparsity)
    min_initial_sparsity = min(initial_sparsity)
    xticks_initial = []
    m = (min_initial_sparsity // 10) *10
    while m <= max_initial_sparsity:
        xticks_initial.append(m)
        m += 10
    max_final_sparsity = max(final_sparsity)
    xticks_final = []
    m = 0
    increase = max_final_sparsity // 5
    while m <= max_final_sparsity:
        xticks_final.append(m)
        m += increase    
    # axs[0].set_xticks(xticks_initial)
    axs.set_xticks(xticks_final)
    # axs[0].set(xlabel='Proof Feature Count', ylabel='Number of Proofs')
    # axs[0].set_title('Baseline')
    axs.hist(np.array(final_sparsity), bins=30)
    axs.set(xlabel='Proof Feature Count', ylabel='Number of Proofs')
    axs.set_title('Random')
    # fig.suptitle(get_name(dict, filename) , y=0.92)

    plt.savefig(dir_name + 'plots/' + filename+'.png')
    get_stats(initial_sparsity, final_sparsity, filename, dir_name)

def process_data(file, filename, dir_name, dict):
    line = file.readline()
    index = None
    initial_sparsity = {}
    initial_sparsity_list = []
    final_sparsity_list = []
    final_sparsity = {}
    non_zero_indices = {}
    final_indices = {}
    starting_tokens = ['property', 'Initial', 'Optimal', 'Indices', 'Remaining']
    while line:
        tokens = line.split(' ', 2)
        line = file.readline()
        if tokens[0] == 'property':
            index= int(tokens[2])
            continue
        if tokens[0] == 'Initial':
            if index is not None:
                if index not in initial_sparsity.keys():
                    initial_sparsity[index] = int(tokens[2])
                    initial_sparsity_list.append(int(tokens[2]))
            continue
        if tokens[0] == 'Optimal':
            if index is not None:
                if index not in final_sparsity.keys():
                    final_sparsity[index] = int(tokens[2])
                    final_sparsity_list.append(int(tokens[2]))
            continue            
        if tokens[0] == 'Indices':
            if index is not None:
                indices = tokens[2]
                while line:
                    tokens = line.split(' ', 2)
                    if tokens[0] in starting_tokens:
                        break
                    indices = indices + line
                    line = file.readline()
                    # print(indices)
                if index not in non_zero_indices.keys():
                    non_zero_indices[index] = indices
            continue
        if tokens[0] == 'Remaining':
            if index is not None:
                indices = tokens[2]
                while line:
                    tokens = line.split(' ', 2)
                    if tokens[0] in starting_tokens:
                        break
                    indices = indices + line
                    line = file.readline()
                if index not in final_indices.keys():
                   final_indices[index] = indices
            continue
    # filename = filename.replace("./mnist-deepz/", "")
    filename = filename.replace(".dat", "")
    pruned_indices_fn = dir_name + filename + "_pruned_indices.dat"
    kept_indices_fn = dir_name + 'final-indices/' + filename + "_kept_indices.dat"
    with open(pruned_indices_fn, 'wb') as f:
        pickle.dump(non_zero_indices, f)
    with open(kept_indices_fn, 'wb') as f:
        pickle.dump(final_indices, f)
        # print("key length ", len(final_sparsity.keys()))
        # print("Key length", len(initial_sparsity_list))
    get_historgrams(initial_sparsity_list, final_sparsity_list, filename, dir_name, dict)




if __name__ == "__main__":
    cifar_net_names = [
     'convSmallRELU__Point.onnx',
     'cifar10_8_255_colt.onnx', 
     'cifar_cnn_2layer_width_2_best.pth', 
     'convSmall_pgd_cifar.onnx'
     ]
    domains = ['Domain.LIRPA_CROWN']
    cifar_eps = [0.2/255, 2/255]

    cifar_dict = {}
    cifar_dict['convSmallRELU__Point.onnx'] = "Standard Training"
    cifar_dict['cifar10_8_255_colt.onnx'] = "COLT Training"
    cifar_dict['cifar10convSmallRELUDiffAI.onnx'] = "Provably Robust Training (DIFFAI)"
    cifar_dict['cifar_cnn_2layer_width_2_best.pth'] = "Provably Robust Training (CROWN-IBP)"
    cifar_dict['convSmall_pgd_cifar.onnx'] = "Empirically Robust Training (PGD)"    

    data_dir_name = './cifar-alpha-crown/'
    cifar_res_dir_name = 'cifar-alpha-crown-res/'
    for net_name in cifar_net_names:
        for domain in domains:
            for ep in cifar_eps:
                if net_name in ['convSmallRELU__Point.onnx']:
                    if ep > (0.2 / 255):
                        continue
                filename ='{}_{}_{}.dat'.format(net_name, domain, ep)
                # print(filename)
                file = open(data_dir_name + filename, 'r+')
                process_data(file, filename, dir_name=cifar_res_dir_name, dict=cifar_dict)
                file.close()