

<p align="center">
  <img width="400" alt="Profit" src="./iclr_code/assets/Profit_logo.png" />
</p>

# ProFIt: ProoF Interpretation Technique 

<p align="left">
    ℹ️&nbsp;<a href="#-about">About</a>
    | 📖&nbsp;<a href="#-more-about-syncode">More About ProFIt</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 👀&nbsp;<a href="#-example-usage">Observations</a>
</p>


## ℹ️ About
<img width="1000" alt="proof_interpret" src="./iclr_code/assets/proof_interpret.png">

**ProFIt** is a novel framework for interpreting verifier-generated proofs for Deep Neural Networks (DNNs). Unlike existing works in DNN interpretability that work for DNN prediction on individual inputs, ProFIt can interpret DNNs for local input regions potentially containing infinitely many images. Using ProFIt, we make several interesting observations regarding the impact of robust training methods on the input features the trained DNN focuses on.

## 📖 More About **ProFIt**

### How **ProFIt** works?
<img width="1000" alt="profit_overview" src="./iclr_code/assets/profit_overview.png">

ProFIt involves two key steps - 1) Proof feature extraction and 2) Proof feature visualization. 
1. **Proof feature extraction** In this step, we extract neuron-level information for the proof at each layer. Given proofs are high-dimensional convex shapes (e.g. Zonotopes, convex polytopes) we project into each dimension to obtain bounding intervals or proof features. However, the number of proof features can be large making them hard to interpret individually. We propose a novel proof feature pruning algorithm that extracts - 1) small (easy to interpret), 2) sufficient (proof preserving) proof features while retaining important (higher priority) proof features.   

2. **Proof feature visualization** For each bounding interval $[l_i , u_i]$, the bounds $l_i = N_l^i(I)$ and $u_i = N_u^i(I)$ can be expressed as a differentiable function where $I$ is the input region. Then we use gradient maps corresponding to the midpoint $\frac{l_i + u_i}{2}$ for visualization.




## 🚀 Quick Start
### Python Installation and Usage Instructions
Clone this repository:
```
git clone https://github.com/uiuc-focal-lab/Profit.git
```
Install dependencies:
Conda Environment at profit.yml and can be installed with the following command.

```
conda env create -f environment.yml
```
Run proof feature extraction:
Move to the ``iclr_code`` folder and run one of the tests from ``src/tests`` folder.

```
python3 -m unittest -v src.tests.testFileName.testClassName.testName

```

Example MNIST network
```
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_02.test_deepz_standard
```

Example CIFAR10 network
```
python3 -m unittest -v src.tests.test_sparse_proofs_cifar10_deepz.TestSparsificationCIFAR_02.test_deepz_standard
```

### Parameters

### Run on custom networks



## 👀 Observations
## 📜 Citation
<p>
    <a href="https://openreview.net/forum?id=Ev10F9TWML"><img src="https://img.shields.io/badge/Paper-arXiv-blue"></a>
</p>

```
@inproceedings{
banerjee2024interpreting,
title={Interpreting Robustness Proofs of Deep Neural Networks},
author={Debangshu Banerjee and Avaljot Singh and Gagandeep Singh},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=Ev10F9TWML}
}
```

## Contact
For questions, please contact [Debangshu Banerjee](mailto:db21@illinois.edu).