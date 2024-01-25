import src.proof_transfer.proof_transfer as pt
import src.proof_transfer.approximate as ap

from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.bnb import Split
from src.common import Domain
from src.proof_transfer.pt_types import ProofTransferMethod, PRUNE, REORDERING
from src.common.dataset import Dataset
from src.common import PriorityHeuristic

class TestSparsificatioCIFAR_01(TestCase):
    def test_deepz_standard(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='cifar-alpha-crown'  
        args = pt.TransferArgs(net=config.CIFAR_STANDARD_CONV, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_pgd_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/cifar10-deepz/'  
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_diffai_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/cifar10-deepz/'  
        args = pt.TransferArgs(net=config.CIFAR_CONV_DIFFAI, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_colt_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/cifar10-deepz/'  
        args = pt.TransferArgs(net=config.CIFAR_CONV_COLT, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_crown_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/cifar10-deepz/'  
        args = pt.TransferArgs(net=config.CIFAR_CROWN_IBP, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


class TestSparsificatioCIFAR_02(TestCase):
    def test_deepz_standard(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient
        args = pt.TransferArgs(net=config.CIFAR_STANDARD_CONV, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification,
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)

    def test_deepz_pgd_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient  
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)

    def test_deepz_diffai_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient  
        args = pt.TransferArgs(net=config.CIFAR_CONV_DIFFAI, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)

    def test_deepz_colt_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient  
        args = pt.TransferArgs(net=config.CIFAR_CONV_COLT, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)

    def test_deepz_crown_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient  
        args = pt.TransferArgs(net=config.CIFAR_CROWN_IBP, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=0.2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)


class TestSparsificatioCIFAR_1(TestCase):
    def test_deepz_pgd_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/cifar10-deepz/'  
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_diffai_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/cifar10-deepz/'  
        args = pt.TransferArgs(net=config.CIFAR_CONV_DIFFAI, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_colt_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/cifar10-deepz/'  
        args = pt.TransferArgs(net=config.CIFAR_CONV_COLT, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_crown_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/cifar10-deepz/'  
        args = pt.TransferArgs(net=config.CIFAR_CROWN_IBP, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=1/255,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


class TestSparsificatioCIFAR_2(TestCase):
    def test_deepz_pgd_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient  
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)

    def test_deepz_diffai_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient  
        args = pt.TransferArgs(net=config.CIFAR_CONV_DIFFAI, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)

    def test_deepz_colt_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient  
        args = pt.TransferArgs(net=config.CIFAR_CONV_COLT, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)

    def test_deepz_crown_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True
        output_dir='neurips_results_gradient/cifar-alpha-crown/'
        priority_heuristic = PriorityHeuristic.Gradient  
        args = pt.TransferArgs(net=config.CIFAR_CROWN_IBP, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                               store_in_file=store_in_file, priority_heuristic=priority_heuristic)
        pt.proof_transfer(args)
