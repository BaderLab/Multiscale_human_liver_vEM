import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def freeze_weight(network):
    """
    Freeze the weight of the encoder. 

    """
    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    for name, param in mod.named_parameters():
        if 'encoder' in name:
            print("name with encoder", name)
            param.requires_grad = False
    
    # Check if the encoder is frozen
    for name, param in mod.named_parameters():
        if 'encoder' in name:
            if not param.requires_grad:
                print(f"Parameter '{name}' is frozen.")
            else:
                print(f"Parameter '{name}' is NOT frozen.")

    return mod

