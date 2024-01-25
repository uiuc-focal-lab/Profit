# To run ProFIt on MNIST networks use the following scripts.
./scripts/run_sparsification_mnist_alpha_crown.sh

# To run ProFIt on MNIST networks use the following scripts.
./scripts/run_sparsification_cifar_alpha_crown.sh

# For proof feature size distribution plots for each priority heuristic run the following.
- Random
python3 neurips_results_random/data_processing_mnist.py
python3 neurips_results_random/data_processing_cifar.py

- Gradient
python3 neurips_results_gradient/data_processing_mnist.py
python3 neurips_results_gradient/data_processing_cifar.py

- ProFIt
python3 neurips_results/data_processing_mnist.py
python3 neurips_results/data_processing_cifar.py


# For running experiments for evaluating each priority heuristic run the following.
python3 quantitive_eval/priority_plot_generator.py
python3 quantitive_eval/priority_err_plot_genarator.py