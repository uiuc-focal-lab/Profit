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


class TestCustom(TestCase):
    def test_custom(self):
        # Property, verifier and Network location.
        net_location=config.CIFAR_CONV_SMALL # network location (onnx format)
        domain=Domain.DEEPZ # See src/common/__init__.py for details.
        eps = 1.0/255 # perturbation bound that defines the input property.
        dataset = Dataset.CIFAR10
        output_dir='final_results/'

        # Proof feature pruning specific arguments.
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = True  
        args = pt.TransferArgs(net=net_location, domain=domain, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=dataset, eps=eps,
                               split=Split.RELU_ESIP_SCORE, count=200, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)