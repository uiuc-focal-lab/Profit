#!/bin/bash

# total number of cpu cores that will be used
cpus_total=1
# absolute cpu number (cpu index on the machine, based on htop command)
cpu_abs=0
# the relative task number, ranging from 0 to $cpus_total-1
# Basically, we split the input pandas table into $cpus_total-1 parts and task_no is the index of a chosen part, 
# the properties of which will be verified. 
# This is mainly used for running several python scripts concurrently.
task_no=0

# base model
timeout=3600
pdprops="base_easy.pkl"
#pdprops="base_med.pkl"
#pdprops="base_hard.pkl"
nn_name="cifar_base_kw"

## wide model
#timeout=7200
#pdprops="wide.pkl"
#nn_name="cifar_wide_kw"

## deep model
#timeout=7200
#pdprops="deep.pkl"
#nn_name="cifar_deep_kw"

# method
# para="--bab_kw"
para="--bab_gnn"
#para="--bab_online"
#para="--gurobi"

py="/home/sugare2/Python-3.7.2/python"

echo "taskset --cpu-list $cpu_abs $py experiments/bab_mip.py --cpu_id $task_no --timeout $timeout --cpus_total $cpus_total --pdprops $pdprops $para --nn_name $nn_name --record"
taskset --cpu-list $cpu_abs $py experiments/bab_mip.py --cpu_id $task_no --timeout $timeout --cpus_total $cpus_total --pdprops $pdprops $para --nn_name $nn_name --record 



