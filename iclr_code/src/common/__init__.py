import re
from enum import Enum

RESULT_DIR = 'results/'


def strip_name(obj, pos=-1):
    return re.split('\.|/', str(obj))[pos]


# Domains used for verification
class Domain(Enum):
    DEEPZ = 1
    DEEPPOLY = 2
    BOX = 3
    LP = 4
    LIRPA_IBP = 5
    LIRPA_CROWN = 6
    LIRPA_CROWN_IBP = 7
    LIRPA_CROWN_OPT = 8
    LIRPA_CROWN_FORWARD = 9


# Feature priority definition.
class FeaturePriority(Enum):
    Random = 1
    Gradient = 2
    Weight_ABS = 3
    Weight_SIGN = 4

class PriorityHeuristic(Enum):
    Default = 1    
    Random = 2
    Gradient = 3



# Feature priority norm definition.
class FeaturePriorityNorm(Enum):
    L2 = 1
    L1 = 2
    Linf = 3

# Used for status of the complete verifier
class Status(Enum):
    VERIFIED = 1
    ADV_EXAMPLE = 2
    UNKNOWN = 3
    MISS_CLASSIFIED = 4
