import torch

from typing import List


class Pipeline:
    def __init__(self,
                 partitions: List[torch.nn.Sequential],
                 devices: List[torch.device],
                 copy_stream: List[List[torch.cuda.stream]]) -> None:
        self.partitions_ = partitions
        self.devices_ = devices
        self.copy_stream_ = copy_stream
