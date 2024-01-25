import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import pickle
import seaborn as sns

def process_data(network_results, file, baseline_name="baseline"):
    line = file.readline()
    while line:
        tokens = line.split(' ')
        net_name_temp = tokens[0]
        eps = tokens[1]
        net_name = net_name_temp +'_'+ eps
        line = file.readline()
        if net_name not in network_results.keys():
            network_results[net_name] = {}
        topk = int(tokens[2])
        priority = tokens[3]
        success_percentage = float(tokens[5])
        success_percentage_baseline = float(tokens[4])
        if priority not in network_results[net_name].keys():
            network_results[net_name][priority] = {}
        if baseline_name not in network_results[net_name].keys():
            network_results[net_name][baseline_name] = {}
        if topk not in network_results[net_name][priority].keys():
            network_results[net_name][priority][topk] = success_percentage
        if topk not in network_results[net_name][baseline_name].keys():
            network_results[net_name][baseline_name][topk] = success_percentage_baseline   

def generate_plots(x_axis_nums, priority_list, priority_res, prioriy_name_map, full_net_name, plots_dir):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    colors = ['blue', 'black', 'red']
    for i, priority in enumerate(priority_list):
          ax.plot(x_axis_nums, priority_res[priority], color=colors[i], label=prioriy_name_map[priority])
    yticks = [0.2*(i+1) for i in range(10)]
    xticks = x_axis_nums
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel('No. of Proof features', fontsize=17)    
    ax.set_ylabel('Relative change in output', fontsize=16)
    plt.legend(loc=1, fontsize="10")
    plt.savefig(f'{plots_dir}{full_net_name}.png', dpi=300)



def get_plots_wrpper(network_results, network_names, eps_list, priority_list, priority_names_map,
              x_axis_nums, plots_dir=''):
    for net_name in network_names:
        for eps in eps_list:
            full_net_name = net_name+'_'+str(eps)
            if full_net_name not in network_results.keys():
                continue
            priority_res = {}
            for priority in priority_list:
                percentage_dict = network_results[full_net_name][priority]
                percentage_list = []
                for x_num in x_axis_nums:
                    percentage_list.append(percentage_dict[x_num])
                priority_res[priority] = percentage_list
            generate_plots(x_axis_nums=x_axis_nums, priority_list=priority_list, priority_res=priority_res,
                           prioriy_name_map=priority_names_map, full_net_name=full_net_name, plots_dir=plots_dir)



if __name__ == "__main__":
    network_names = [
        'mnist_0.1.onnx',
        'mnist_cnn_2layer_width_1_best_modified.pth',
        'mnistconvSmallRELU__PGDK.onnx',
    ]
    eps_list = [0.02]
    priority_names_map = {}
    algo_name = "ProFIt"
    baseline_name = "baseline"
    priority_list = [baseline_name, "FeaturePriority.Random", "FeaturePriority.Weight_ABS"]
    priority_names_map[baseline_name] = algo_name
    priority_names_map["FeaturePriority.Random"] = "Random"
    priority_names_map["FeaturePriority.Weight_ABS"] = "Gradient"
    network_results = {}
    filename = './neurips_error.dat'
    plots_dir = './plot_err/'
    file = open(filename, 'r+')
    x_axis_nums = [i for i in range(2, 21, 2)]
    process_data(network_results=network_results, file=file, baseline_name=baseline_name)
    get_plots_wrpper(network_results=network_results, network_names=network_names, eps_list=eps_list, 
                     priority_list=priority_list, priority_names_map=priority_names_map, x_axis_nums=x_axis_nums,
                     plots_dir=plots_dir)
    file.close()