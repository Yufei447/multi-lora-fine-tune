from mlora.pipeline.pipe import splite_module
from mlora.pipeline.worker import Worker, create_workers

import torch

from typing import List


class Pipeline:
    # module : the troch.nn.Sequential module
    # balance: list of number of layers in each partition (gpu)
    # chunks : number of batch (parallelized batch)
    def __init__(self,
                 module: torch.nn.Sequential,
                 balance: List[int],
                 chunks: int) -> None:
        devices = range(torch.cuda.device_count())

        self.devices_: List[torch.device] = [torch.device(d) for d in devices]
        self.partitions_: List[torch.nn.Sequential] = splite_module(
            module, balance, self.devices_)
        # copy stream: device_id * chunk + chunk_id
        # each device has chunk copy stream
        self.copy_stream_: List[torch.cuda.Stream] = []

        for device in self.devices_:
            self.copy_stream_.append(
                [torch.cuda.Stream(device) for _ in range(chunks)])

        self.workers_: List[Worker] = create_workers(self.devices_)

    def run(self) -> None:
        partitions = self.partitions_

        def clock_cycles():
            pass
