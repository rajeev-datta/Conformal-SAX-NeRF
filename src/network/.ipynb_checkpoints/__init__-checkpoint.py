from .network import DensityNetwork
from .Lineformer import Lineformer
from .LineformerUQ import UQLineformer

def get_network(type, unc):
    if type == "mlp":
        return DensityNetwork
    elif type == "Lineformer" and not unc:
        return Lineformer
    elif type == "Lineformer" and unc:
        return UQLineformer
    else:
        raise NotImplementedError("Unknown network type!")

