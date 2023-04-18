from enum import Enum


class CutoffMethod(Enum):
    AbsoluteDistance = 1  # L1 norm
    DistanceSquared = 2  # L2 norm
