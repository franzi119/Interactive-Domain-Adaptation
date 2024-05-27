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

import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from monai.config import IgniteInfo, KeysCollection, PathLike
from monai.utils import ensure_tuple, look_up_option, min_version, optional_import

idist, _ = optional_import("ignite", IgniteInfo.OPT_IMPORT_VERSION, min_version, "distributed")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")

__all__ = ["stopping_fn_from_metric", "stopping_fn_from_loss", "write_metrics_reports", "from_engine"]



def own_from_engine(keys: KeysCollection, first: bool = False) -> Callable:
    """
    Utility function to simplify the `batch_transform` or `output_transform` args of ignite components
    when handling dictionary or list of dictionaries(for example: `engine.state.batch` or `engine.state.output`).
    Users only need to set the expected keys, then it will return a callable function to extract data from
    dictionary and construct a tuple respectively.

    If data is a list of dictionaries after decollating, extract expected keys and construct lists respectively,
    for example, if data is `[{"A": 1, "B": 2}, {"A": 3, "B": 4}]`, from_engine(["A", "B"]): `([1, 3], [2, 4])`.

    It can help avoid a complicated `lambda` function and make the arg of metrics more straight-forward.
    For example, set the first key as the prediction and the second key as label to get the expected data
    from `engine.state.output` for a metric::

        from monai.handlers import MeanDice, from_engine

        metric = MeanDice(
            include_background=False,
            output_transform=from_engine(["pred", "label"])
        )

    Args:
        keys: specified keys to extract data from dictionary or decollated list of dictionaries.
        first: whether only extract specified keys from the first item if input data is a list of dictionaries,
            it's used to extract the scalar data which doesn't have batch dim and was replicated into every
            dictionary when decollating, like `loss`, etc.


    """
    _keys = ensure_tuple(keys)
    print(_keys)
    print('type keys', type(keys))
    def _wrapper(data):
        nonlocal _keys
        if(_keys == ('',)):
            _keys = [list(data[0].keys())[0], list(data[0].keys())[1]]
            print("handler",keys)
        if isinstance(data, dict):
            return tuple(data[k] for k in _keys)
        if isinstance(data, list) and isinstance(data[0], dict):
            print("handlers utils", list(data[0].keys()))
            print("handlers utils", list(data[0].keys())[1])
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [data[0][k] if first else [i[k] for i in data] for k in _keys]
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper