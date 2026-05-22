"""
This file contains utility functions for training.

NOTE: This file is not used inside the HNet package, but contains useful utilities for training the model itself.
"""

import torch

from ..models.mixer_seq import HNetForCausalLM
from ..modules.utils import apply_optimization_params

def group_params(
    model: HNetForCausalLM,
) -> list[dict[str, list[torch.Tensor] | float]]:
    """
    Creates parameter groups for the optimizer, based on the learning rate multiplier and weight decay.

    Each parameter group has the following form: 
    {
        "params": [list of parameters],
        "lr": learning rate
        "weight_decay": weight decay,
    }

    Inputs:
        model: The model to group parameters for.
        lr_multiplier: A list of learning rate multipliers, one for each stage of the hierarchy, with the outer stages first (e.g. [3.0, 1.7, 0.9]).
        weight_decay: The weight decay to apply to all parameters (except bias + norms)

    Returns:
        A list of parameter groups, each with the above form.
    """
    param_groups = []
    all_keys = set()

    for name, param in model.named_parameters():
        if name.endswith(".bias") or ".norm." in name:
            apply_optimization_params(param, weight_decay=0.0)
        
        all_keys.update(param._optim.keys())
    
    all_keys = list(all_keys)
    all_tuples = []
    param_groups = []

    for name, param in model.named_parameters():
        current_tuple = tuple(param._optim.get(key, None) for key in all_keys)
        if current_tuple not in all_tuples:
            all_tuples.append(current_tuple)
            param_groups.append({
                "params": [param],
                **param._optim,
            })
        else:
            idx = all_tuples.index(current_tuple)
            param_groups[idx]["params"].append(param)
    
    return param_groups
