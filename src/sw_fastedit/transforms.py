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


from pathlib import Path
from typing import Callable
import traceback
import warnings

import numpy as np
import cv2
import torch

import monai
from monai.data.meta_tensor import MetaTensor
from monai.config import DtypeLike, KeysCollection
from monai.data import image_writer
from monai.data.image_reader import ImageReader
from monai.transforms.io.array import LoadImage, SaveImage
from monai.transforms.transform import MapTransform, Transform
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix
from monai.data.folder_layout import default_name_formatter, FolderLayout, FolderLayoutBase


from monai.config import DtypeLike, NdarrayOrTensor, PathLike
from monai.data import image_writer
from monai.data.meta_obj import get_track_meta
from monai.data.folder_layout import FolderLayout, FolderLayoutBase, default_name_formatter
from monai.data.image_reader import (
    ImageReader,
    ITKReader,
    NibabelReader,
    NrrdReader,
    NumpyReader,
    PILReader,
    PydicomReader,
)
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import is_no_channel
from monai.transforms.transform import Transform
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils import GridSamplePadMode
from monai.utils import ImageMetaKey as Key
from monai.utils import OptionalImportError, convert_to_dst_type, ensure_tuple, look_up_option, optional_import

from monai.utils import TransformBackends, convert_data_type, convert_to_tensor, ensure_tuple, look_up_option



__all__ = ["LoadImaged", "LoadImageD", "LoadImageDict", "SaveImaged", "SaveImageD", "SaveImageDict"]

DEFAULT_POST_FIX = PostFix.meta()


import gc
import os
import json
import logging
import re
from typing import Dict, Hashable, List, Mapping, Tuple
from copy import deepcopy
from pydoc import locate

import torch
from scipy.ndimage import gaussian_filter
from monai.config import KeysCollection
from monai.data import MetaTensor, PatchIterd
from monai.losses import DiceLoss
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    MapTransform,
    Randomizable,
    LazyTransform,
    InvertibleTransform,
    Flip,
)
from monai.transforms.post.array import (
    AsDiscrete,
)

from collections.abc import Hashable, Mapping, Sequence

from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utility.array import AddExtremePointsChannel
from monai.transforms.utils import extreme_points_to_image, get_extreme_points
from monai.transforms.utils_pytorch_numpy_unification import concatenate
from monai.utils.type_conversion import convert_to_dst_type
from monai.transforms.utils import check_non_lazy_pending_ops
from monai.transforms.utils_pytorch_numpy_unification import where
from monai.transforms.utility.array import SplitDim
from monai.transforms.traits import MultiSampleTrait
import numpy as np



from sw_fastedit.utils.enums import CommonKeys
from sw_fastedit.utils.enums import GanKeys

from sw_fastedit.click_definitions import LABELS_KEY, ClickGenerationStrategy
from sw_fastedit.utils.distance_transform import get_random_choice_from_tensor, get_border_points_from_mask
from monai.transforms.utils import distance_transform_edt
from sw_fastedit.utils.helper import get_global_coordinates_from_patch_coordinates, get_tensor_at_coordinates, timeit
#from FastGeodis import generalised_geodesic3d


logger = logging.getLogger("sw_fastedit")


def get_guidance_tensor_for_key_label(data, key_label, device) -> torch.Tensor:
    """Makes sure the guidance is in a tensor format."""
    tmp_gui = data.get(key_label, torch.tensor([], dtype=torch.int32, device=device))
    if isinstance(tmp_gui, list):
        tmp_gui = torch.tensor(tmp_gui, dtype=torch.int32, device=device)
    assert type(tmp_gui) is torch.Tensor or type(tmp_gui) is MetaTensor
    return tmp_gui


# TODO Franzi - one transform class - AddExtremePoints - already included in MONAI


class AddExtremePointsChanneld(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddExtremePointsChannel`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: key to label source to get the extreme points.
        background: Class index of background label, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.
        sigma: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        rescale_min: minimum value of output data.
        rescale_max: maximum value of output data.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = AddExtremePointsChannel.backend
    

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        background: int = 0,
        pert: float = 0.0,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor = 5.0,
        rescale_min: float = -1.0,
        rescale_max: float = 1.0,
        allow_missing_keys: bool = False,
        label_names = {},
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_names = label_names 
        self.background = background
        self.pert = pert
        self.points: list[tuple[int, ...]] = []
        self.label_key = label_key
        self.sigma = sigma
        self.rescale_min = rescale_min
        self.rescale_max = rescale_max
        

    def randomize(self, label: NdarrayOrTensor) -> None:
        self.points = get_extreme_points(label, rand_state=self.R, background=self.background, pert=self.pert)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        label = d[self.label_key]
        self.randomize(label[0, :])

        d['guidance'] = self.points
        return d



def get_extreme_points(
    img: NdarrayOrTensor, rand_state: np.random.RandomState | None = None, background: int = 0, pert: float = 0.0
) -> list[tuple[int, ...]]:
    """
    Generate extreme points from an image. These are used to generate initial segmentation
    for annotation models. An optional perturbation can be passed to simulate user clicks.

    Args:
        img:
            Image to generate extreme points from. Expected Shape is ``(spatial_dim1, [, spatial_dim2, ...])``.
        rand_state: `np.random.RandomState` object used to select random indices.
        background: Value to be consider as background, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.

    Returns:
        A list of extreme points, its length is equal to 2 * spatial dimension of input image.
        The output format of the coordinates is:

        indices of [1st_spatial_dim_min, 1st_spatial_dim_max, 2nd_spatial_dim_min, ..., Nth_spatial_dim_max]

    Raises:
        ValueError: When the input image does not have any foreground pixel.
    """
    check_non_lazy_pending_ops(img, name="get_extreme_points")
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    #image is label (0,1), get indices where label is not background
    #print('img', img.shape)
    #print('unique img', torch.unique(img))

    indices = where(img != background)

    #npindices = np.array(indices)
    #print('indices', npindices.shape)
    #print('unique', np.unique(npindices))

    #indices[0] is x-axis, indices[1] is y-axis, indices[2] is z-axis
    if np.size(indices[0]) == 0:
        raise ValueError("get_extreme_points: no foreground object in mask!")

    def _get_point(val, dim):
        """
        Select one of the indices within slice containing val.

        Args:
            val : value for comparison
            dim : dimension in which to look for value
        """

        idx = where(indices[dim] == val)[0]

        idx = idx.cpu() if isinstance(idx, torch.Tensor) else idx
        idx = rand_state.choice(idx) if rand_state is not None else idx
        pt = []
        for j in range(img.ndim):
            # add +- pert to each dimension
            val = int(indices[j][idx] + 2.0 * pert * (rand_state.rand() if rand_state is not None else 0.5 - 0.5))
            val = max(val, 0)
            val = min(val, img.shape[j] - 1)
            pt.append(val)
        return pt

    points = []
    for i in range(img.ndim):
        points.append(tuple(_get_point(indices[i].min(), i)))
        points.append(tuple(_get_point(indices[i].max(), i)))

    return points


class AddEmptySignalChannels(MapTransform):
    """
        Adds empty channels to the signal which will be filled with the guidance signal later.
        E.g. for two labels: 1x192x192x256 -> 3x192x192x256
    """
    def __init__(self, device, keys: KeysCollection = None):
        super().__init__(keys)
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set up the initial batch data

        in_channels = len(data[LABELS_KEY]) 
        tmp_image = data[CommonKeys.IMAGE][0:0+1, ...] #load data into temp
        assert len(tmp_image.shape) == 4
        new_shape = list(tmp_image.shape)
        new_shape[0] = in_channels
        # Set the signal to 0 for all input images
        # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
        # e.g. (3,128,128,128) for two labels
        inputs = torch.zeros(new_shape) #, device=self.device)
        inputs[0] = data[CommonKeys.IMAGE][0] #load image into inputs
        if isinstance(data[CommonKeys.IMAGE], MetaTensor):
            data[CommonKeys.IMAGE].array = inputs
        else:
            data[CommonKeys.IMAGE] = inputs

        return data
    
class AddEmptySignalChannelsLabel(MapTransform):
    """
        Adds empty channels to the signal which will be filled with the guidance signal later.
        E.g. for two labels: 1x192x192x256 -> 3x192x192x256
    """
    def __init__(self, device, keys: KeysCollection = None):
        super().__init__(keys)
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set up the initial batch data

        in_channels = len(data[LABELS_KEY]) 
        tmp_image = data[CommonKeys.LABEL][0 : 0 + 1, ...]
        assert len(tmp_image.shape) == 4
        new_shape = list(tmp_image.shape)
        new_shape[0] = in_channels
        # Set the signal to 0 for all input images
        # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
        # e.g. (3,128,128,128) for two labels
        inputs = torch.zeros(new_shape) #, device=self.device)
        inputs[0] = data[CommonKeys.LABEL][0]
        if isinstance(data[CommonKeys.LABEL], MetaTensor):
            data[CommonKeys.LABEL].array = inputs
        else:
            data[CommonKeys.LABEL] = inputs
        return data



class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    backend = AsDiscrete.backend

    def __init__(
        self,
        keys: KeysCollection,
        argmax: Sequence[bool] | bool = False,
        to_onehot: Sequence[int | None] | int | None = None,
        threshold: Sequence[float | None] | float | None = None,
        rounding: Sequence[str | None] | str | None = None,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold value.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"]. it also can be a sequence of str or None,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            kwargs: additional parameters to ``AsDiscrete``.
                ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
                These default to ``0``, ``True``, ``torch.float`` respectively.

        """
        super().__init__(keys, allow_missing_keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        self.to_onehot = []
        for flag in ensure_tuple_rep(to_onehot, len(self.keys)):
            if isinstance(flag, bool):
                raise ValueError("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
            self.to_onehot.append(flag)

        self.threshold = []
        for flag in ensure_tuple_rep(threshold, len(self.keys)):
            if isinstance(flag, bool):
                raise ValueError("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
            self.threshold.append(flag)

        self.rounding = ensure_tuple_rep(rounding, len(self.keys))
        self.converter = AsDiscrete()
        self.converter.kwargs = kwargs

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, argmax, to_onehot, threshold, rounding in self.key_iterator(
            d, self.argmax, self.to_onehot, self.threshold, self.rounding
        ):
            d[key] = self.converter(d[key], argmax, to_onehot, threshold, rounding)
        return d
    

class AsDiscrete(Transform):
    """
    Convert the input tensor/array into discrete values, possible operations are:

        -  `argmax`.
        -  threshold input value to binary values.
        -  convert input value to One-Hot format (set ``to_one_hot=N``, `N` is the number of classes).
        -  round the value to the closest integer.

    Args:
        argmax: whether to execute argmax function on input data before transform.
            Defaults to ``False``.
        to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
            Defaults to ``None``.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold.
            Defaults to ``None``.
        rounding: if not None, round the data according to the specified option,
            available options: ["torchrounding"].
        kwargs: additional parameters to `torch.argmax`, `monai.networks.one_hot`.
            currently ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
            These default to ``0``, ``True``, ``torch.float`` respectively.

    Example:

        >>> transform = AsDiscrete(argmax=True)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[1.0, 1.0]]]

        >>> transform = AsDiscrete(threshold=0.6)
        >>> print(transform(np.array([[[0.0, 0.5], [0.8, 3.0]]])))
        # [[[0.0, 0.0], [1.0, 1.0]]]

        >>> transform = AsDiscrete(argmax=True, to_onehot=2, threshold=0.5)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[0.0, 0.0]], [[1.0, 1.0]]]

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        argmax: bool = False,
        to_onehot: int | None = None,
        threshold: float | None = None,
        rounding: str | None = None,
        **kwargs,
    ) -> None:
        self.argmax = argmax
        if isinstance(to_onehot, bool):  # for backward compatibility
            raise ValueError("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
        self.to_onehot = to_onehot
        self.threshold = threshold
        self.rounding = rounding
        self.kwargs = kwargs

    def __call__(
        self,
        img: NdarrayOrTensor,
        argmax: bool | None = None,
        to_onehot: int | None = None,
        threshold: float | None = None,
        rounding: str | None = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the input tensor data to convert, if no channel dimension when converting to `One-Hot`,
                will automatically add it.
            argmax: whether to execute argmax function on input data before transform.
                Defaults to ``self.argmax``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                Defaults to ``self.to_onehot``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold value.
                Defaults to ``self.threshold``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"].

        """
        if isinstance(to_onehot, bool):
            raise ValueError("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t, *_ = convert_data_type(img, torch.Tensor)
        if argmax or self.argmax:
            img_t = torch.argmax(img_t, dim=self.kwargs.get("dim", 0), keepdim=self.kwargs.get("keepdim", True))

        to_onehot = self.to_onehot if to_onehot is None else to_onehot
        if to_onehot is not None:
            if not isinstance(to_onehot, int):
                raise ValueError(f"the number of classes for One-Hot must be an integer, got {type(to_onehot)}.")
            img_t = one_hot(
                img_t, num_classes=to_onehot, dim=self.kwargs.get("dim", 0), dtype=self.kwargs.get("dtype", torch.float)
            )

        threshold = self.threshold if threshold is None else threshold
        if threshold is not None:
            img_t = img_t >= threshold

        rounding = self.rounding if rounding is None else rounding
        if rounding is not None:
            look_up_option(rounding, ["torchrounding"])
            img_t = torch.round(img_t)

        img, *_ = convert_to_dst_type(img_t, img, dtype=self.kwargs.get("dtype", torch.float))
        return img
    


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])

    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    # if sh[dim] != 1:
    #     raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

class NormalizeLabelsInDatasetd(MapTransform):
    """
    Normalize label values according to the label names dictionary.

    Args:
        keys: the ``keys`` parameter will be used to get and set the actual data item to transform.
        labels: dictionary mapping label names to label values.
        allow_missing_keys: whether to ignore it if keys are missing.
        device: device this transform shall run on.

    Returns:
        The transformed data with the new label mapping stored under the key LABELS_KEY.
    """
    def __init__(
        self,
        keys: KeysCollection,
        labels=None,
        allow_missing_keys: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.labels = labels
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set the labels dict if no labels were provided
        data[LABELS_KEY] = self.labels
        #print('label unique_0', torch.unique(data['label']))
        for key in self.key_iterator(data):
            if "label" in key:
                label = data[key]
                if isinstance(label, str):
                    raise AttributeError("Label is expected to be a tensor, but is a string.")

                # Initialize a dictionary to store new label numbers
                new_labels = {"background": 0}
                
                # Create a tensor to store normalized labels
                normalized_label = torch.zeros_like(label, device=self.device)

                # Assign new labels based on the provided dictionary
                #print('label unique_1', torch.unique(label))
                for idx, (key_label, val_label) in enumerate(self.labels.items(), start=1):
                    #print("key label", key_label, idx)
                    if key_label != "background":
                        new_labels[key_label] = idx
                        normalized_label[label == val_label] = idx
                #print('label unique_2', torch.unique(normalized_label))

                # Store the new labels dictionary
                data[LABELS_KEY] = new_labels

                # Update the label tensor
                if isinstance(data[key], MetaTensor):
                    data[key].array = normalized_label
                else:
                    data[key] = normalized_label
            else:
                raise UserWarning("Only keys containing 'label' are allowed here!")

        return data




# class NormalizeLabelsInDatasetd(MapTransform):
#     """
#     Normalize label values according to label names dictionary

#     Args:
#         keys: the ``keys`` parameter will be used to get and set the actual data item to transform
#         labels: all label names
#         allow_missing_keys: whether to ignore it if keys are missing.
#         device: device this transform shall run on

#     Returns: data and also the new labels will be stored in data with key LABELS_KEY
#     """
#     def __init__(
#         self,
#         keys: KeysCollection,
#         labels=None,
#         allow_missing_keys: bool = False,
#         device=None,
#     ):
#         super().__init__(keys, allow_missing_keys)
#         self.labels = labels
#         self.device = device

#     def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
#         # Set the labels dict in case no labels were provided
#         data[LABELS_KEY] = self.labels

#         for key in self.key_iterator(data):
#             #print(data['label_names'].keys())
#             if "label" in key:
#                 label = data[key]
#                 #print('label unique_1', torch.unique(label))
#                 if isinstance(label, str):
#                     # Special case since label has been defined to be a string in MONAILabel
#                     raise AttributeError("Label is expected to be a tensor, but is a string.")


#                 # Dictionary containing new label numbers
#                 print('label unique_4', torch.unique(label))
#                 new_labels = {}
#                 label = torch.zeros(data[key].shape, device=self.device)
#                 # Making sure the range values and number of labels are the same
#                 for idx, (key_label, val_label) in enumerate(self.labels.items(), start=1):
#                     print("key label", key_label)
#                     print("val label", val_label)
#                     print()
#                     if key_label != "background":
#                         print("not background", idx)
#                         new_labels[key_label] = idx
#                         label[data[key] == val_label] = idx

#                     if key_label == "background":
#                         print("background", idx)
#                         new_labels["background"] = 0
#                     else:
#                         print("else")
#                         new_labels[key_label] = idx
#                         label[data[key] == val_label] = idx
#                 print('label unique_5', torch.unique(label))
#                 data[LABELS_KEY] = new_labels
#                 if isinstance(data[key], MetaTensor):
#                     data[key].array = label
#                 else:
#                     data[key] = label
#             else:
#                 raise UserWarning("Only the key label is allowed here!")
#             #print('label unique_6', torch.unique(label))
#         return data

class AddMRIorCT():
    def __init__(
        self,
        keys: KeysCollection
    ):
        self.keys = keys
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        d = dict(data)
        #logger.info(f"image file name: {d['image_meta_dict']['filename_or_obj']}")
        #logger.info(f"label file name: {d['label_meta_dict']['filename_or_obj']}")
        numbers = re.findall(r'\d+', d['image_meta_dict']['filename_or_obj'])

        # Get the last number
        last_number = numbers[-1] if numbers else None

        print(last_number)
        if (int(last_number)>= 500):
            d[GanKeys.GLABEL] = torch.zeros((1,1))
        else:
            d[GanKeys.GLABEL] = torch.ones((1,1))
        return d


class AddGuidanceSignald(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        guidance: str = "guidance",
        sigma: int = 2,
        number_intensity_ch=3,
    ):
        super().__init__(keys, allow_missing_keys)

        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch

    def signal(self, shape, points):
        signal = np.zeros(shape, dtype=np.float32)

        flag = False
        for p in points:
            if np.any(np.asarray(p) < 0):
                continue
            if len(shape) == 3:
                signal[p[-3], p[-2], p[-1]] = 1.0
            else:
                signal[p[-2], p[-1]] = 1.0
            flag = True

        if flag:
            signal = gaussian_filter(signal, sigma=self.sigma)
            #print('min max signal',np.min(signal), np.max(signal))
            #print('signal unique', np.unique(signal))
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        return torch.Tensor(signal)[None]

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            guidance = d[self.guidance]

            guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
            if guidance and (guidance[0] or guidance[1]):
                img = img[0 : 0 + self.number_intensity_ch, ...]

                shape = img.shape[-2:] if len(img.shape) == 3 else img.shape[-3:]
                device = img.device if isinstance(img, torch.Tensor) else None
                res = self.signal(shape, guidance).to(device=device)
                #pos = self.signal(shape, guidance[0]).to(device=device)
                #neg = self.signal(shape, guidance[1]).to(device=device)
                result = torch.concat([img if isinstance(img, torch.Tensor) else torch.Tensor(img), res])
                #print('result unique', torch.unique(result))
            else:
                s = torch.zeros_like(img[0])[None]
                result = torch.concat([img, s, s])
            #result = torch.round(result)
            
            d[key] = result
        return d


class AddGuidanceSignal(MapTransform):
    """
    Add Guidance signal for input image.

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        disks: This paraemters fill spheres with a radius of sigma centered around each click.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 1,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        disks: bool = False,
        gdt: bool = False,
        spacing: Tuple = None,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.disks = disks
        self.gdt = gdt
        self.spacing = spacing
        self.device = device

    def _get_corrective_signal(self, image, guidance, key_label):
        dimensions = 3 if len(image.shape) > 3 else 2
        assert (
            type(guidance) is torch.Tensor or type(guidance) is MetaTensor
        ), f"guidance is {type(guidance)}, value {guidance}"

        if guidance.size()[0]:
            first_point_size = guidance[0].numel()
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                # Assuming the guidance has either shape (1, x, y , z) or (x, y, z)
                assert (
                    first_point_size == 4 or first_point_size == 3
                ), f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                assert first_point_size == 3, f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)

            sshape = signal.shape

            for point in guidance:
                if torch.any(point < 0):
                    continue
                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if torch.max(signal[0]) > 0:
                signal_tensor = signal[0]
                if self.sigma != 0:
                    pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                    signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                    signal_tensor = signal_tensor.squeeze(0).squeeze(0)

                signal[0] = signal_tensor
                signal[0] = (signal[0] - torch.min(signal[0])) / (torch.max(signal[0]) - torch.min(signal[0]))
                if self.disks:
                    signal[0] = (signal[0] > 0.1) * 1.0  # 0.1 with sigma=1 --> radius = 3, otherwise it is a cube

                # if self.gdt:
                #     geos = generalised_geodesic3d(image.unsqueeze(0).to(self.device),
                #                                 signal[0].unsqueeze(0).unsqueeze(0).to(self.device),
                #                                 self.spacing,
                #                                 10e10,
                #                                 1.0,
                #                                 2)




                #     signal[0] = geos[0][0]

            if not (torch.min(signal[0]).item() >= 0 and torch.max(signal[0]).item() <= 1.0):
                raise UserWarning(
                    "[WARNING] Bad signal values",
                    torch.min(signal[0]),
                    torch.max(signal[0]),
                )



            if signal is None:
                raise UserWarning("[ERROR] Signal is None")
            return signal
        else:
            if dimensions == 3:
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)
            if signal is None:
                print("[ERROR] Signal is None")
            return signal

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:

        for key in self.key_iterator(data):
            if key == "image":
                
                image = data[key]

                #image = data[key].to(torch.device("cuda:0"))
                #assert image.is_cuda 
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]

                label_key = list(data[LABELS_KEY].keys())[0]
                label_guidance = get_guidance_tensor_for_key_label(data, label_key, self.device)
                logger.debug(f"Converting guidance for label {label_key}:{label_guidance} into a guidance signal..")

                if label_guidance is not None and label_guidance.numel():
                    signal = self._get_corrective_signal(
                        image,
                        label_guidance.to(device=self.device),
                        key_label=label_key,
                    )
                    
                    assert torch.sum(signal) > 0
                else:
                    signal = self._get_corrective_signal(
                        image,
                        torch.Tensor([]).to(device=self.device),
                        key_label=label_key,
                    )

                #assert signal.is_cuda 
                #assert tmp_image.is_cuda 
                tmp_image = torch.cat([tmp_image, signal], dim=0)
                if isinstance(data[key], MetaTensor):
                    data[key].array = tmp_image
                else:
                    data[key] = tmp_image

                return data
            else:
                raise UserWarning("This transform only applies to image key")
        raise UserWarning("image key has not been been found")




class PrintShape(MapTransform):


    def __init__(
        self,
        keys: KeysCollection = None,
        allow_missing_keys: bool = False,
        prev_transform: str = None

    ):
        super().__init__(keys, allow_missing_keys)
        self.prev_transform = prev_transform

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        if self.keys != None:
            #for key in self.key_iterator(data):
            print(self.prev_transform, torch.unique(data['label']))
        #print(data.keys())
        return data

class PrintKeys(MapTransform):


    def __init__(
        self,
        keys: KeysCollection = None,
        allow_missing_keys: bool = False,
        prev_transform: str = None

    ):
        super().__init__(keys, allow_missing_keys)
        self.prev_transform = prev_transform

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        print(data.keys())
        return data
    

class SplitDimd(MapTransform, MultiSampleTrait):
    backend = SplitDim.backend

    def __init__(
        self,
        keys: KeysCollection,
        output_postfixes: Sequence[str] | None = None,
        dim: int = 0,
        keepdim: bool = True,
        update_meta: bool = True,
        list_output: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfixes: the postfixes to construct keys to store split data.
                for example: if the key of input data is `pred` and split 2 classes, the output
                data keys will be: pred_(output_postfixes[0]), pred_(output_postfixes[1])
                if None, using the index number: `pred_0`, `pred_1`, ... `pred_N`.
            dim: which dimension of input image is the channel, default to 0.
            keepdim: if `True`, output will have singleton in the split dimension. If `False`, this
                dimension will be squeezed.
            update_meta: if `True`, copy `[key]_meta_dict` for each output and update affine to
                reflect the cropped image
            list_output: it `True`, the output will be a list of dictionaries with the same keys as original.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.output_postfixes = output_postfixes
        self.splitter = SplitDim(dim, keepdim, update_meta)
        self.list_output = list_output
        if self.list_output is None and self.output_postfixes is not None:
            raise ValueError("`output_postfixes` should not be provided when `list_output` is `True`.")

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor] | list[dict[Hashable, torch.Tensor]]:
        d = dict(data)
        all_keys = list(set(self.key_iterator(d)))

        if self.list_output:
            output = []
            results = [self.splitter(d[key]) for key in all_keys]
            for row in zip(*results):
                new_dict = dict(zip(all_keys, row))
                # fill in the extra keys with unmodified data
                for k in set(d.keys()).difference(set(all_keys)):
                    new_dict[k] = deepcopy(d[k])
                output.append(new_dict)
            return output

        for key in all_keys:
            rets = self.splitter(d[key])
            postfixes: Sequence = list(range(len(rets))) if self.output_postfixes is None else self.output_postfixes
            if len(postfixes) != len(rets):
                raise ValueError(f"count of splits must match output_postfixes, {len(postfixes)} != {len(rets)}.")
            for i, r in enumerate(rets):
                split_key = f"{key}_{postfixes[i]}"
                if split_key in d:
                    raise RuntimeError(f"input data already contains key {split_key}.")
                d[split_key] = r
        return d


class SaveImagedSlices(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SaveImage`.

    Note:
        Image should be channel-first shape: [C,H,W,[D]].
        If the data is a patch of an image, the patch index will be appended to the filename.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            For example, for data with key ``image``, the metadata by default is in ``image_meta_dict``.
            The metadata is a dictionary contains values such as ``filename``, ``original_shape``.
            This argument can be a sequence of strings, mapped to the ``keys``.
            If ``None``, will try to construct ``meta_keys`` by ``key_{meta_key_postfix}``.
        meta_key_postfix: if ``meta_keys`` is ``None``, use ``key_{meta_key_postfix}`` to retrieve the metadict.
        output_dir: output image directory.
                    Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        output_postfix: a string appended to all output file names, default to ``trans``.
                        Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        output_ext: output file extension name, available extensions: ``.nii.gz``, ``.nii``, ``.png``, ``.dcm``.
                    Handled by ``folder_layout`` instead, if ``folder_layout`` not ``None``.
        resample: whether to resample image (if needed) before saving the data array,
            based on the ``spatial_shape`` (and ``original_affine``) from metadata.
        mode: This option is used when ``resample=True``. Defaults to ``"nearest"``.
            Depending on the writers, the possible options are:

            - {``"bilinear"``, ``"nearest"``, ``"bicubic"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.
            Possible options are {``"zeros"``, ``"border"``, ``"reflection"``}
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
            [0, 255] (``uint8``) or [0, 65535] (``uint16``). Default is ``None`` (no scaling).
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if None, use the data type of input data. To set the output data type, use ``output_dtype``.
        output_dtype: data type for saving data. Defaults to ``np.float32``.
        allow_missing_keys: don't raise exception if key is missing.
        squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D is also 1, it will be saved as (H,W). If `false`,
            image will always be saved as (H,W,D,C).
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. It's used to compute ``input_file_rel_path``, the relative path to the file from
            ``data_root_dir`` to preserve folder structure when saving in case there are files in different
            folders with the same file names. For example, with the following inputs:

            - input_file_name: ``/foo/bar/test1/image.nii``
            - output_postfix: ``seg``
            - output_ext: ``.nii.gz``
            - output_dir: ``/output``
            - data_root_dir: ``/foo/bar``

            The output will be: ``/output/test1/image/image_seg.nii.gz``

            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        separate_folder: whether to save every file in a separate folder. For example: for the input filename
            ``image.nii``, postfix ``seg`` and folder_path ``output``, if ``separate_folder=True``, it will be saved as:
            ``output/image/image_seg.nii``, if ``False``, saving as ``output/image_seg.nii``. Default to ``True``.
            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        print_log: whether to print logs when saving. Default to ``True``.
        output_format: an optional string to specify the output image writer.
            see also: ``monai.data.image_writer.SUPPORTED_WRITERS``.
        writer: a customised ``monai.data.ImageWriter`` subclass to save data arrays.
            if ``None``, use the default writer from ``monai.data.image_writer`` according to ``output_ext``.
            if it's a string, it's treated as a class name or dotted path;
            the supported built-in writer classes are ``"NibabelWriter"``, ``"ITKWriter"``, ``"PILWriter"``.
        output_name_formatter: a callable function (returning a kwargs dict) to format the output file name.
            see also: :py:func:`monai.data.folder_layout.default_name_formatter`.
            If using a custom ``folder_layout``, consider providing your own formatter.
        folder_layout: A customized ``monai.data.FolderLayoutBase`` subclass to define file naming schemes.
            if ``None``, uses the default ``FolderLayout``.
        savepath_in_metadict: if ``True``, adds a key ``saved_to`` to the metadata, which contains the path
            to where the input image has been saved.
    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        output_dir: Path | str = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = False,
        mode: str = "nearest",
        padding_mode: str = GridSamplePadMode.BORDER,
        scale: int | None = None,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike | None = np.float32,
        allow_missing_keys: bool = False,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
        output_format: str = "",
        writer: type[image_writer.ImageWriter] | str | None = None,
        output_name_formatter: Callable[[dict, Transform], dict] | None = None,
        folder_layout: monai.data.FolderLayoutBase | None = None,
        savepath_in_metadict: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        #self.output_postfix = output_postfix
        #self.output_dir = os.path.join(output_dir, output_postfix)
        self.separate_folder = separate_folder
        self.saver = SaveImageSlices(
            output_dir=output_dir,
            output_postfix=output_postfix,
            output_ext=output_ext,
            resample=resample,
            mode=mode,
            padding_mode=padding_mode,
            scale=scale,
            dtype=dtype,
            output_dtype=output_dtype,
            squeeze_end_dims=squeeze_end_dims,
            data_root_dir=data_root_dir,
            separate_folder=separate_folder,
            print_log=print_log,
            output_format=output_format,
            writer=writer,
            output_name_formatter=output_name_formatter,
            folder_layout=folder_layout,
            savepath_in_metadict=savepath_in_metadict,
        )

    def set_options(self, init_kwargs=None, data_kwargs=None, meta_kwargs=None, write_kwargs=None):
        self.saver.set_options(init_kwargs, data_kwargs, meta_kwargs, write_kwargs)
        return self

    def __call__(self, data):
        d = dict(data)
        #print('output_dir', self.output_dir)
        #print('output_postfix', self.output_postfix)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if meta_key is None and meta_key_postfix is not None:
                meta_key = f"{key}_{meta_key_postfix}"
            meta_data = d.get(meta_key) if meta_key is not None else None

        
            self.saver(img=d[key], meta_data=meta_data)
        return d
    


class SaveImageSlices(Transform):
    """
    Save the image (in the form of torch tensor or numpy ndarray) and metadata dictionary into files.

    The name of saved file will be `{input_image_name}_{output_postfix}{output_ext}`,
    where the `input_image_name` is extracted from the provided metadata dictionary.
    If no metadata provided, a running index starting from 0 will be used as the filename prefix.

    Args:
        output_dir: output image directory.
        Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        output_postfix: a string appended to all output file names, default to `trans`.
        Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        output_ext: output file extension name.
        Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        output_dtype: data type (if not None) for saving data. Defaults to ``np.float32``.
        resample: whether to resample image (if needed) before saving the data array,
            based on the ``"spatial_shape"`` (and ``"original_affine"``) from metadata.
        mode: This option is used when ``resample=True``. Defaults to ``"nearest"``.
            Depending on the writers, the possible options are

            - {``"bilinear"``, ``"nearest"``, ``"bicubic"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.
            Possible options are {``"zeros"``, ``"border"``, ``"reflection"``}
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
            [0, 255] (``uint8``) or [0, 65535] (``uint16``). Default is ``None`` (no scaling).
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if ``None``, use the data type of input data. To set the output data type, use ``output_dtype``.
        squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D is also 1, it will be saved as (H,W). If ``False``,
            image will always be saved as (H,W,D,C).
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. It's used to compute ``input_file_rel_path``, the relative path to the file from
            ``data_root_dir`` to preserve folder structure when saving in case there are files in different
            folders with the same file names. For example, with the following inputs:

            - input_file_name: ``/foo/bar/test1/image.nii``
            - output_postfix: ``seg``
            - output_ext: ``.nii.gz``
            - output_dir: ``/output``
            - data_root_dir: ``/foo/bar``

            The output will be: ``/output/test1/image/image_seg.nii.gz``

            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        separate_folder: whether to save every file in a separate folder. For example: for the input filename
            ``image.nii``, postfix ``seg`` and ``folder_path`` ``output``, if ``separate_folder=True``, it will be
            saved as: ``output/image/image_seg.nii``, if ``False``, saving as ``output/image_seg.nii``.
            Default to ``True``.
            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        print_log: whether to print logs when saving. Default to ``True``.
        output_format: an optional string of filename extension to specify the output image writer.
            see also: ``monai.data.image_writer.SUPPORTED_WRITERS``.
        writer: a customised ``monai.data.ImageWriter`` subclass to save data arrays.
            if ``None``, use the default writer from ``monai.data.image_writer`` according to ``output_ext``.
            if it's a string, it's treated as a class name or dotted path (such as ``"monai.data.ITKWriter"``);
            the supported built-in writer classes are ``"NibabelWriter"``, ``"ITKWriter"``, ``"PILWriter"``.
        channel_dim: the index of the channel dimension. Default to ``0``.
            ``None`` to indicate no channel dimension.
        output_name_formatter: a callable function (returning a kwargs dict) to format the output file name.
            If using a custom ``monai.data.FolderLayoutBase`` class in ``folder_layout``, consider providing
            your own formatter.
            see also: :py:func:`monai.data.folder_layout.default_name_formatter`.
        folder_layout: A customized ``monai.data.FolderLayoutBase`` subclass to define file naming schemes.
            if ``None``, uses the default ``FolderLayout``.
        savepath_in_metadict: if ``True``, adds a key ``"saved_to"`` to the metadata, which contains the path
            to where the input image has been saved.
    """

    def __init__(
        self,
        output_dir: PathLike = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        output_dtype: DtypeLike | None = np.float32,
        resample: bool = False,
        mode: str = "nearest",
        padding_mode: str = GridSamplePadMode.BORDER,
        scale: int | None = None,
        dtype: DtypeLike = np.float64,
        squeeze_end_dims: bool = True,
        data_root_dir: PathLike = "",
        separate_folder: bool = True,
        print_log: bool = True,
        output_format: str = "",
        writer: type[image_writer.ImageWriter] | str | None = None,
        channel_dim: int | None = 0,
        output_name_formatter: Callable[[dict, Transform], dict] | None = None,
        folder_layout: FolderLayoutBase | None = None,
        savepath_in_metadict: bool = False,
    ) -> None:
        self.folder_layout: FolderLayoutBase
        if folder_layout is None:
            self.folder_layout = FolderLayout(
                output_dir=output_dir,
                postfix=output_postfix,
                extension=output_ext,
                parent=separate_folder,
                makedirs=True,
                data_root_dir=data_root_dir,
            )
        else:
            self.folder_layout = folder_layout

        self.fname_formatter: Callable
        if output_name_formatter is None:
            self.fname_formatter = default_name_formatter
        else:
            self.fname_formatter = output_name_formatter

        self.output_ext = output_ext.lower() or output_format.lower()
        if isinstance(writer, str):
            writer_, has_built_in = optional_import("monai.data", name=f"{writer}")  # search built-in
            if not has_built_in:
                writer_ = locate(f"{writer}")  # search dotted path
            if writer_ is None:
                raise ValueError(f"writer {writer} not found")
            writer = writer_
        self.writers = image_writer.resolve_writer(self.output_ext) if writer is None else (writer,)
        self.writer_obj = None

        _output_dtype = output_dtype
        if self.output_ext == ".png" and _output_dtype not in (np.uint8, np.uint16, None):
            _output_dtype = np.uint8
        if self.output_ext == ".dcm" and _output_dtype not in (np.uint8, np.uint16, None):
            _output_dtype = np.uint8
        self.init_kwargs = {"output_dtype": _output_dtype, "scale": scale}
        self.data_kwargs = {"squeeze_end_dims": squeeze_end_dims, "channel_dim": channel_dim}
        self.meta_kwargs = {"resample": resample, "mode": mode, "padding_mode": padding_mode, "dtype": dtype}
        self.write_kwargs = {"verbose": print_log}
        self._data_index = 0
        self.savepath_in_metadict = savepath_in_metadict
        self.output_dir = output_dir
        self.output_postfix = output_postfix

    def set_options(self, init_kwargs=None, data_kwargs=None, meta_kwargs=None, write_kwargs=None):
        """
        Set the options for the underlying writer by updating the `self.*_kwargs` dictionaries.

        The arguments correspond to the following usage:

            - `writer = ImageWriter(**init_kwargs)`
            - `writer.set_data_array(array, **data_kwargs)`
            - `writer.set_metadata(meta_data, **meta_kwargs)`
            - `writer.write(filename, **write_kwargs)`

        """
        if init_kwargs is not None:
            self.init_kwargs.update(init_kwargs)
        if data_kwargs is not None:
            self.data_kwargs.update(data_kwargs)
        if meta_kwargs is not None:
            self.meta_kwargs.update(meta_kwargs)
        if write_kwargs is not None:
            self.write_kwargs.update(write_kwargs)
        return self

    def __call__(self, img: torch.Tensor | np.ndarray, meta_data: dict | None = None):
        """
        Args:
            img: target data content that save into file. The image should be channel-first, shape: `[C,H,W,[D]]`.
            meta_data: key-value pairs of metadata corresponding to the data.
        """
        meta_data = img.meta if isinstance(img, MetaTensor) else meta_data
        #print('img', img.shape)
        kw = self.fname_formatter(meta_data, self)
        filename = self.folder_layout.filename(**kw)
        #print('kw', kw)
        #print('filename', filename)
        #print('output_dir', self.output_dir)
        #print('output_postfix', self.output_postfix)
        modified_path = filename.rsplit('/', 1)[0]
        modified_path = os.path.join(modified_path, self.output_postfix)
        #print('modiefied path', modified_path)
        try:
            os.makedirs(modified_path, exist_ok=True)
            print(f"Directory created successfully: {modified_path}")
        except Exception as e:
            print(f"Error creating directory: {e}")

        for i in range(img.shape[1]):
            slice = img[0][i]*255
            cv2.imwrite(os.path.join(modified_path, f'{i}_ep_slice_.png'), slice.detach().cpu().numpy().astype(np.uint8))


        # if meta_data:
        #     meta_spatial_shape = ensure_tuple(meta_data.get("spatial_shape", ()))
        #     if len(meta_spatial_shape) >= len(img.shape):
        #         self.data_kwargs["channel_dim"] = None
        #     elif is_no_channel(self.data_kwargs.get("channel_dim")):
        #         warnings.warn(
        #             f"data shape {img.shape} (with spatial shape {meta_spatial_shape}) "
        #             f"but SaveImage `channel_dim` is set to {self.data_kwargs.get('channel_dim')} no channel."
        #         )


        # for i in range(img[key].shape[2]):
        #     slice = img[key][0][0][i]*255
        #     cv2.imwrite(os.path.join(self.output_dir, self.output_postfix, '{i}_ep_slice_.png'), slice.detach().cpu().numpy().astype(np.uint8))

        # err = []
        # for writer_cls in self.writers:
        #     try:
        #         writer_obj = writer_cls(**self.init_kwargs)
        #         writer_obj.set_data_array(data_array=img, **self.data_kwargs)
        #         writer_obj.set_metadata(meta_dict=meta_data, **self.meta_kwargs)
        #         writer_obj.write(filename, **self.write_kwargs)
        #         self.writer_obj = writer_obj
        #     except Exception as e:
        #         err.append(traceback.format_exc())
        #         logging.getLogger(self.__class__.__name__).debug(e, exc_info=True)
        #         logging.getLogger(self.__class__.__name__).info(
        #             f"{writer_cls.__class__.__name__}: unable to write {filename}.\n"
        #         )
        #     else:
        #         self._data_index += 1
        #         if self.savepath_in_metadict and meta_data is not None:
        #             meta_data["saved_to"] = filename
        #         return img
        # msg = "\n".join([f"{e}" for e in err])
        # raise RuntimeError(
        #     f"{self.__class__.__name__} cannot find a suitable writer for {filename}.\n"
        #     "    Please install the writer libraries, see also the installation instructions:\n"
        #     "    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies.\n"
        #     f"   The current registered writers for {self.output_ext}: {self.writers}.\n{msg}"
        # )






class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation
    """

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "pred":
                for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                    if key_label != "background":
                        data[f"pred_{key_label}"] = data[key][idx + 1, ...][None]
                        data[f"label_{key_label}"] = data["label"][idx + 1, ...][None]
            elif key != "pred":
                logger.info("This transform is only for pred key")
        return data


class FlipChanneld(MapTransform, InvertibleTransform, LazyTransform):
    """
    A transformation class to flip specified channels along the specified spatial axis in tensor-like data.
    This is an invertible transform and can be applied lazily.

    Args:
        keys: Keys specifying the data in the dictionary to be processed.
        spatial_axis: Spatial axis or axes along which flipping should occur.
        channels: Channels to be flipped.
        allow_missing_keys: If True, allows for keys in `keys` to be missing in the input dictionary.
        lazy: If True, the transform is applied lazily.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_axis: Sequence[int] | int | None = None,
        channels: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = Flip(spatial_axis=spatial_axis)
        self.channels = channels

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val



    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            for channel in self.channels:
                d[key][channel:channel+1] = self.flipper(d[key][channel:channel+1], lazy=lazy_)
        return d




    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            for channel in self.channels:
                d[key][channel:channel+1] = self.flipper.inverse(d[key][channel:channel+1])
        return d