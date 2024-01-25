import os
from copy import deepcopy

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
    initial_sparsity = np.array(initial_sparsity)
    final_sparsity = np.array(final_sparsity)
    initial_sparsity_mean = np.mean(initial_sparsity)
    initial_sparsity_median = np.median(initial_sparsity)
    final_sparsity_mean = np.mean(final_sparsity)
    final_sparsity_median = np.median(final_sparsity)
    fn = dir_name + fn + "_stat.dat"
    total_proved_prop = len(sparsity_ratio)
    file = open(fn, "w+")
    file.write("Total proved property {}\n".format(total_proved_prop))
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
    file.write("less than 5 {}\n".format(less_than_5))
    file.write("less than 10 {}\n".format(less_than_10))
    file.write("less than 5 percent {}\n".format(less_than_5 / final_sparsity.shape[0] * 100))
    file.write("less than 10 percent {}\n".format(less_than_10 / final_sparsity.shape[0] * 100))
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
    # axs[0].hist(np.array(initial_sparsity), bins=35)
    if len(initial_sparsity) == 0 or len(final_sparsity) == 0:
        return
    max_initial_sparsity = max(initial_sparsity)
    min_initial_sparsity = min(initial_sparsity)
    xticks_initial = []
    m = (min_initial_sparsity // 5) *5
    increase = (max_initial_sparsity - min_initial_sparsity) // 5
    # print("increase", increase)
    increase = (increase // 5) *5
    if increase == 0:
        increase = 5
    while m <= max_initial_sparsity:
        xticks_initial.append(m)
        m += increase
    min_final_sparsity = min(final_sparsity)
    max_final_sparsity = max(final_sparsity)
    xticks_final = []
    increase = (max_final_sparsity - min_final_sparsity) // 5
    increase = (increase // 5) *5
    m = 0
    if increase == 0:
        increase = 3
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
    # print("initial sparsity dict length", len(initial_sparsity.keys()))
    # print("initial sparsity dict length", len(initial_sparsity_list))    
    get_historgrams(initial_sparsity_list, final_sparsity_list, filename, dir_name, dict)




if __name__ == "__main__":
    mnist_net_names = [
    'mnist_standard_modified.pt',
    'convMedGRELU__PGDK_w_0.3.onnx',
    'mnist_0.1.onnx', 
    'mnist_cnn_2layer_width_1_best_modified.pth', 
    ]
    domains = [
            'Domain.LIRPA_CROWN', 
             ]
    mnist_eps = [0.02, 0.1]

    mnist_dict = {}
    mnist_dict['mnist_standard_modified.pt'] = "Standard Training"
    mnist_dict['mnist_0.1.onnx'] = "COLT Training"
    mnist_dict['mnistconvSmallRELUDiffAI.onnx'] = "Provably Robust Training (DIFFAI)"
    mnist_dict['mnist_cnn_2layer_width_1_best_modified.pth'] = "Provably Robust Training (CROWN-IBP)"
    mnist_dict['convMedGRELU__PGDK_w_0.3.onnx'] = "Empirically Robust Training (PGD)"    

    data_dir_name = './mnist-alpha-crown/'
    mnist_res_dir_name = 'mnist-alpha-crown-res/'
    for net_name in mnist_net_names:
        for domain in domains:
            for ep in mnist_eps:
                if net_name in ['mnist_standard_modified.pt'] and ep > 0.02:
                    continue
                filename ='{}_{}_{}.dat'.format(net_name, domain, ep)
                file = open(data_dir_name + filename, 'r+')
                process_data(file, filename, dir_name=mnist_res_dir_name, dict=mnist_dict)
                file.close()

    # data_dir_name = './mnist-box/'
    # mnist_res_dir_name = 'mnist-box-res/'
    # domains = ['Domain.BOX']
    # for net_name in mnist_net_names:
    #     for domain in domains:
    #         for ep in mnist_eps:
    #             if net_name in ['mnist-net_256x2.onnx', 'mnistconvSmallRELU__PGDK.onnx']:
    #                 if ep > 0.02:
    #                     continue
    #             filename ='{}_{}_{}.dat'.format(net_name, domain, ep)
    #             file = open(data_dir_name + filename, 'r+')
    #             process_data(file, filename, dir_name=mnist_res_dir_name, dict=mnist_dict)
    #             file.close()