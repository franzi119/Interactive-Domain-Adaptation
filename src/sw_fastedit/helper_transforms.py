from __future__ import annotations

import logging
import time
from typing import Hashable, Iterable, Mapping
import gc

import torch
from monai.config import KeysCollection
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    MapTransform,
    Transform,
)

from sw_fastedit.utils.helper import (
    describe_batch_data,
)
from sw_fastedit.utils.logger import get_logger, setup_loggers

logger = None

cast_labels_to_zero_and_one = lambda x: torch.where(x > 0, 1, 0)

def threshold_foreground(x):
    return (x > 0.005) & (x < 0.995)


class TrackTimed(Transform):
    """
    A transform that tracks and logs the execution time of another transform.

    Args:
        transform (Transform): The transform to be timed.
    """
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        global logger
        start_time = time.perf_counter()
        data = self.transform(data)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"-------- {self.transform.__class__.__qualname__:<20.20}() took {total_time:.3f} seconds")

        return data



class InitLoggerd(MapTransform):
    """
    Initializes the logger inside the dataloader thread.

    Args:
        loglevel (int): Logging level. Default is logging.INFO.
        no_log (bool): Whether to disable logging. Default is True.
        log_dir (str): Directory to save log files. Default is None.

    Note:
        If `no_log` is True, logging will be disabled, and `log_dir` will be ignored.
    """
    def __init__(self, loglevel=logging.INFO, no_log=True, log_dir=None):

        global logger
        super().__init__(None)

        self.loglevel = loglevel
        self.log_dir = log_dir
        self.no_log = no_log

        if self.no_log:
            self.log_dir = None

        setup_loggers(self.loglevel, self.log_dir)
        logger = get_logger()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        global logger
        if logger is None:
            setup_loggers(self.loglevel, self.log_dir)
        logger = get_logger()
        return data
