import torch

from typing import Iterable, List
from collections import OrderedDict


def splite_module(module: torch.nn.Sequential,
                  balance: Iterable[int],
                  devices: List[torch.device]) -> List[torch.nn.Sequential]:
    if len(devices) != len(balance):
        raise Exception(
            f"the balance size {len(balance)} and devices size {len(devices)} not same.")

    partition = OrderedDict()
    partitions: List = []
    balance_idx = 0

    # TODO: load in different gpu device
    for name, layer in module.named_children():
        partition[name] = layer

        if len(partition) == balance[balance_idx]:

            p = torch.nn.Sequential(partition)
            p.to(devices[balance_idx])

            partitions.append(p)
            partition.clear()

            balance_idx += 1

    return partitions
