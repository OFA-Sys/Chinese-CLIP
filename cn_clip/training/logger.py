import logging
from logging import Filter
from logging.handlers import QueueHandler, QueueListener

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

logging.raiseExceptions = False

def setup_primary_logging(log_file, level, rank):
    log_queue = Queue(-1)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%Y-%m-%d,%H:%M:%S')

    if rank == 0:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    if rank == 0:
        listener = QueueListener(log_queue, file_handler, stream_handler)
    else:
        listener = QueueListener(log_queue, stream_handler)

    listener.start()

    return log_queue


class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True


def setup_worker_logging(rank, log_queue, level):
    queue_handler = QueueHandler(log_queue)

    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)

    queue_handler.setLevel(level)

    root_logger = logging.getLogger()
    if len(root_logger.handlers) > 0:
        root_logger.removeHandler(root_logger.handlers[0])
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(level)
