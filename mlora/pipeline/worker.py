import torch

from queue import Queue
from typing import List
from threading import Thread


class Worker:
    def __init__(self,
                 device: torch.device) -> None:
        self.device_ = device
        self.inqueue_ = Queue()
        self.outqueue_ = Queue()


def worker_run(worker: Worker) -> None:
    pass


def create_workers(devices: List[torch.device]) -> List[Worker]:
    workers: List[Worker] = []

    for idx, device in enumerate(devices):
        workers.append(Worker(device))

        worker_thread = Thread(
            target=worker_run, args=(workers[idx], ), daemon=True)
        worker_thread.start()

    return workers
