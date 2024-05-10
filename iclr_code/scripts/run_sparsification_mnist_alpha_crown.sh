#!/bin/bash

# standard 
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_02.test_deepz_standard > results/output.txt
# pgd trained
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_02.test_deepz_pgd_trained > results/output.txt
# diff ai trained
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_02.test_deepz_diffai_trained > results/output.txt
# colt trained
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_02.test_deepz_colt_trained > results/output.txt
# crown-ibp trained
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_02.test_deepz_crown_trained > results/output.txt
# pgd trained
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_1.test_deepz_pgd_trained > results/output.txt
# diff ai trained
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_1.test_deepz_diffai_trained > results/output.txt
# colt trained
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_1.test_deepz_colt_trained > results/output.txt
# crown-ibp trained
python3 -m unittest -v src.tests.test_sparse_proofs_mnist_deepz.TestMnistSparsification_1.test_deepz_crown_trained > results/output.txt