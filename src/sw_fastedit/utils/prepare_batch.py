# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from monai.config import IgniteInfo
from monai.transforms import apply_transform
from monai.utils import ensure_tuple, min_version, optional_import
from sw_fastedit.utils.enums import CommonKeys, GanKeys

if TYPE_CHECKING:
    from ignite.engine import EventEnum
else:
    EventEnum, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum", as_type="base"
    )

__all__ = [
    "IterationEvents",
    "get_devices_spec",
    "PrepareBatch",
    "PrepareBatchDefault",
    "PrepareBatchExtraInput",
    "default_make_latent",
    "engine_apply_transform",
    "default_metric_cmp_fn",
]

def default_prepare_batch(
    batchdata: dict[str, torch.Tensor] | torch.Tensor | Sequence[torch.Tensor],
    device: str | torch.device | None = None,
    non_blocking: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
    """
    Default function to prepare the data for current iteration.

    The input `batchdata` is either a single tensor, a pair of tensors, or a dictionary of data. In the first case the
    return value is the tensor and None, in the second case the return value is the two tensors, and in the dictionary
    case the return value depends on what keys are present. if `CommonKeys.IMAGE` and `CommonKeys.LABEL` are present
    then the tensors they key to are returned, if only `CommonKeys.IMAGE` is present that tensor and None is returned.
    If `CommonKeys.REALS` is present this is returned with None. All returned tensors are moved to the given device
    using the given non-blocking argument before being returned.

    This function implements the expected API for a `prepare_batch` callable in Ignite:
    https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html

    Args:
        batchdata: input batch data which is either a single tensor, a pair, or a dictionary
        device: device to move every returned tensor to
        non_blocking: equivalent argument for `Tensor.to`
        kwargs: further arguments for `Tensor.to`

    Returns:
        image, label(optional).
    """
    if not isinstance(batchdata, dict):
        if isinstance(batchdata, torch.Tensor):
            return batchdata.to(device=device, non_blocking=non_blocking, **kwargs), None
        elif len(batchdata) == 2:
            image, label = batchdata
            return (
                image.to(device=device, non_blocking=non_blocking, **kwargs),
                label.to(device=device, non_blocking=non_blocking, **kwargs),
            )

        raise AssertionError("Default prepare_batch expects a single tensor, a tensor pair, or dictionary input data.")

    if isinstance(batchdata.get(CommonKeys.IMAGE_SOURCE), torch.Tensor):
        return (
            batchdata[CommonKeys.IMAGE_SOURCE].to(device=device, non_blocking=non_blocking, **kwargs),
            batchdata[CommonKeys.LABEL_SEG].to(device=device, non_blocking=non_blocking, **kwargs),
            batchdata[CommonKeys.LABEL_EP].to(device=device, non_blocking=non_blocking, **kwargs),


        )
    if isinstance(batchdata.get(CommonKeys.IMAGE_TARGET), torch.Tensor):
        return (
            batchdata[CommonKeys.IMAGE_TARGET].to(device=device, non_blocking=non_blocking, **kwargs),
            batchdata[CommonKeys.LABEL_SEG].to(device=device, non_blocking=non_blocking, **kwargs),
            batchdata[CommonKeys.LABEL_EP].to(device=device, non_blocking=non_blocking, **kwargs),

        )

    if GanKeys.REALS in batchdata:
        return batchdata[GanKeys.REALS].to(device=device, non_blocking=non_blocking, **kwargs)

    return (
            batchdata[CommonKeys.IMAGE_SOURCE].to(device=device, non_blocking=non_blocking, **kwargs),
            batchdata[CommonKeys.IMAGE_TARGET].to(device=device, non_blocking=non_blocking, **kwargs),
        )