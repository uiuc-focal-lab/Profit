import src.proof_transfer.proof_transfer as pt
import src.proof_transfer.approximate as ap

from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.bnb import Split
from src.common import Domain, FeaturePriority, FeaturePriorityNorm
from src.proof_transfer.pt_types import ProofTransferMethod, PRUNE, REORDERING
from src.common.dataset import Dataset


class TestPriorityComaparison(TestCase):
    def test_PGD_02_random(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Random        
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_PGD, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_PGD_02_weight_abs(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Weight_ABS     
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_PGD, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_PGD_02_weight_signed(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Weight_SIGN
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_PGD, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)


    def test_COLT_02_random(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Random        
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_COLT_02_weight_abs(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Weight_ABS     
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_COLT_02_weight_signed(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Weight_SIGN
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_CROWN_02_random(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Random        
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_CROWN_IBP_MODIFIED, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_CROWN_02_weight_abs(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Weight_ABS     
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_CROWN_IBP_MODIFIED, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_CROWN_02_weight_signed(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Weight_SIGN
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_CROWN_IBP_MODIFIED, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_standard_02_random(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Random        
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_standard_02_weight_abs(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Weight_ABS     
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=500, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)

    def test_standard_02_weight_signed(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        for topk in range(2, 21, 2):
            topk_priority=FeaturePriority.Weight_SIGN
            topk_priority_norm = FeaturePriorityNorm.Linf
            args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                                split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                                timeout=30, output_dir='linear-search-results/', prop_index=None, ignore_properties=[], 
                                pruning_args=pruning_args, enable_sparsification=enable_sparsification, 
                                store_in_file=store_in_file, do_linear_search=False, topk=topk, topk_priority=topk_priority, 
                                priority_norm=topk_priority_norm)
            pt.proof_transfer(args)