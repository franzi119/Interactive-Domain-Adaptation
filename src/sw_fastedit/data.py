from __future__ import annotations

import glob
import logging
import os

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from monai.data import ThreadDataLoader
from monai.data.dataset import PersistentDataset
from monai.data.folder_layout import FolderLayout
from monai.transforms import (
    Activationsd,
    Compose,
    CopyItemsd,
    DivisiblePadd,
    EnsureChannelFirstd,
    Identityd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    SaveImaged,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SignalFillEmptyd,
    Spacingd,
    ToTensord,
)
from monai.utils.enums import CommonKeys

from sw_fastedit.utils.costum_sampler import AlternatingSampler, SpecificSampler

from sw_fastedit.helper_transforms import (
    InitLoggerd,
    TrackTimed,
)
from sw_fastedit.transforms import (
    AddExtremePointsChanneld,
    NormalizeLabelsInDatasetd,
    AddGuidanceSignald,
    SplitDimd,
    SaveImagedSlices,
    AsDiscreted,
    ZScoreNormalized,

)
from sw_fastedit.utils.helper import convert_mha_to_nii, convert_nii_to_mha

logger = logging.getLogger("sw_fastedit")


dataset_names = ["AMOS"]



def get_spacing(args):
    """
    Get the voxel spacing for the specified dataset.

    Args:
        args: Additional arguments containing the dataset information.

    Returns:
        Tuple: A tuple representing the voxel spacing in (x, y, z) dimensions.

    Raises:
        UserWarning: If the specified dataset is not recognized.

    Note:
        This function returns the voxel spacing based on the dataset specified in the arguments.
        Currently supported datasets are "AutoPET," "AutoPET2," "AutoPET2_Challenge," "HECKTOR," and "MSD_Spleen."
    """
    #option from the AMOS22 challenge paper Fabian Isensee et al. http://dx.doi.org/10.1007/978-3-658-41657-7_7
    AMOS_SPACING = (3*1.0, 3*1.0, 3*1.5)

    if args.dataset == "AMOS":
        return AMOS_SPACING
    else:
        raise UserWarning(f"No valid dataset found: {args.dataset}")

def get_pre_transforms_train_as_list_ct(labels: Dict, device, args, input_keys, label, image):
    """
    Get a list of pre-transforms for training data.

    Args:
        labels (Dict): Dictionary of labels.
        device: The device on which to perform the transformations.
        args: Additional arguments containing information for preprocessing.
        input_keys (tuple): Tuple of input keys, default is ("image", "label").

    Returns:
        List: A list of pre-transforms for training data.

    Note:
        This function returns a list of MONAI transforms to be applied to training data.
        It includes operations such as loading images, normalization, cropping, flipping, and more.
    """
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # data Input keys have to be ["image", "label"] for train, and at least ["image"] for val
    if args.dataset in dataset_names:
        t = [
            # Initial transforms on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(
                keys=input_keys,
                reader="ITKReader",
                image_only=False,
                simple_keys=True,
            ),
            ToTensord(keys=input_keys, device=cpu_device, track_meta=True),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys=label, labels=labels, device=cpu_device),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=image, pixdim=spacing),
            Spacingd(keys=label, pixdim=spacing, mode="nearest") if (label in input_keys) else Identityd(keys=input_keys),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically (only for MRI)
            ScaleIntensityRangePercentilesd(
                keys=image, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ) if args.same_normalization else (
                ScaleIntensityRanged(
                    keys=image, a_min=-45, a_max=105, b_min=-1.0, b_max=1.0, clip=True
                ) if args.organ == 6 else (
                    ScaleIntensityRanged(
                        keys=image, a_min=-150, a_max=250, b_min=-1.0, b_max=1.0, clip=True
                    ) if args.organ in [7, 10] else
                    ScaleIntensityRangePercentilesd(
                        keys=image, lower=0.05, upper=99.95, b_min=-1.0, b_max=1.0, clip=True, relative=False
                    )
                )
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0),   
            #Data augmentation
            RandFlipd(keys=input_keys, spatial_axis=[0], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[1], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=input_keys, prob=0.10, max_k=3),
            #For debugging, replacing nan
            SignalFillEmptyd(input_keys),
            #add ground truth extreme points to label
            AddExtremePointsChanneld(label_names = args.labels,keys = label, label_key = label,sigma = args.sigma,),
            #convert extreme points to gausian heatmap
            AddGuidanceSignald(keys=label,sigma=args.sigma),
            SplitDimd(keys=('label')),
            CopyItemsd(keys=("label_0", "label_1"), times=1,
                    names=("label_seg", "label_ep"), allow_missing_keys=True)
         

            # Move to GPU
            # WARNING: Activating the line below leads to minimal gains in performance
            # However you are buying these gains with a lot of weird errors and problems
            # So my recommendation after months of fiddling is to leave this off
            # Until MONAI has fixed the underlying issues
            # ToTensord(keys=("image", "label"), device=device, track_meta=False),
        ]

        if args.debug:
            for i in range(len(t)):
                t[i] = TrackTimed(t[i])
            print(t)

    return t


def get_pre_transforms_train_as_list_mri(labels: Dict, device, args, input_keys, label, image):
    """
    Get a list of pre-transforms for training data.

    Args:
        labels (Dict): Dictionary of labels.
        device: The device on which to perform the transformations.
        args: Additional arguments containing information for preprocessing.
        input_keys (tuple): Tuple of input keys, default is ("image", "label").

    Returns:
        List: A list of pre-transforms for training data.

    Note:
        This function returns a list of MONAI transforms to be applied to training data.
        It includes operations such as loading images, normalization, cropping, flipping, and more.
    """
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    # data Input keys have to be ["image", "label"] for train, and at least ["image"] for val
    if args.dataset in dataset_names:
        t = [
            # Initial transforms on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(
                keys=input_keys,
                reader="ITKReader",
                image_only=False,
                simple_keys=True,
            ),
            ToTensord(keys=input_keys, device=cpu_device, track_meta=True),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys=label, labels=labels, device=cpu_device),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=image, pixdim=spacing),
            Spacingd(keys=label, pixdim=spacing, mode="nearest") if (label in input_keys) else Identityd(keys=input_keys),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically (only for MRI)
            ScaleIntensityRangePercentilesd(
                keys=image, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ) if args.same_normalization else (
                ZScoreNormalized(keys=image, clip=True)
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0),   
            #Data augmentation
            RandFlipd(keys=input_keys, spatial_axis=[0], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[1], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=input_keys, prob=0.10, max_k=3),
            #For debugging, replacing nan
            SignalFillEmptyd(input_keys),
            #add ground truth extreme points to label
            AddExtremePointsChanneld(label_names = args.labels,keys = label, label_key = label,sigma = args.sigma,),
            #convert extreme points to gausian heatmap
            AddGuidanceSignald(keys=label,sigma=args.sigma),
            SplitDimd(keys=('label')),
            CopyItemsd(keys=("label_0", "label_1"), times=1,
                    names=("label_seg", "label_ep"), allow_missing_keys=True)
         

            # Move to GPU
            # WARNING: Activating the line below leads to minimal gains in performance
            # However you are buying these gains with a lot of weird errors and problems
            # So my recommendation after months of fiddling is to leave this off
            # Until MONAI has fixed the underlying issues
            # ToTensord(keys=("image", "label"), device=device, track_meta=False),
        ]

        if args.debug:
            for i in range(len(t)):
                t[i] = TrackTimed(t[i])
            print(t)

    return t


def get_pre_transforms_val_as_list_ct(labels: Dict, device, args, input_keys, label, image):
    """
    Get a list of pre-transforms for validation data.

    Args:
        labels (Dict): Dictionary of labels.
        device: The device on which to perform the transformations.
        args: Additional arguments containing information for preprocessing.
        input_keys (tuple): Tuple of input keys, default is ("image", "label").

    Returns:
        List: A list of pre-transforms for validation data.

    Note:
        This function returns a list of MONAI transforms to be applied to validation data.
        It includes operations such as loading images, normalization, cropping, and more.
    """
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # data Input keys have to be at least ["image"] for val
    if args.dataset in dataset_names:
        t = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys=label, labels=labels, device=cpu_device, allow_missing_keys=True),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=image, pixdim=spacing),
            Spacingd(keys=label, pixdim=spacing, mode="nearest") if (label in input_keys) else Identityd(keys=input_keys, allow_missing_keys=True),
            ScaleIntensityRangePercentilesd(
                keys=image, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ) if args.same_normalization else (
                ScaleIntensityRanged(
                    keys=image, a_min=-45, a_max=105, b_min=-1.0, b_max=1.0, clip=True
                ) if args.organ == 6 else (
                    ScaleIntensityRanged(
                        keys=image, a_min=-150, a_max=250, b_min=-1.0, b_max=1.0, clip=True
                    ) if args.organ in [7, 10] else
                    ScaleIntensityRangePercentilesd(
                        keys=image, lower=0.05, upper=99.95, b_min=-1.0, b_max=1.0, clip=True, relative=False
                    )
                )
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0),
            #add ground truth extreme points to label
            AddExtremePointsChanneld(label_names = args.labels,keys = label, label_key = label,sigma = args.sigma,),
            AddGuidanceSignald(keys=label,sigma=args.sigma),
            SplitDimd(keys=('label')),
            CopyItemsd(keys=("label_0", "label_1"), times=1,
                    names=("label_seg", "label_ep"), allow_missing_keys=True)
    
        ]
        
    if args.debug:
        for i in range(len(t)):
            t[i] = TrackTimed(t[i])

    return t


def get_pre_transforms_val_as_list_mri(labels: Dict, device, args, input_keys, label, image):
    """
    Get a list of pre-transforms for validation data.

    Args:
        labels (Dict): Dictionary of labels.
        device: The device on which to perform the transformations.
        args: Additional arguments containing information for preprocessing.
        input_keys (tuple): Tuple of input keys, default is ("image", "label").

    Returns:
        List: A list of pre-transforms for validation data.

    Note:
        This function returns a list of MONAI transforms to be applied to validation data.
        It includes operations such as loading images, normalization, cropping, and more.
    """
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    # data Input keys have to be at least ["image"] for val
    if args.dataset in dataset_names:
        t = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys=label, labels=labels, device=cpu_device, allow_missing_keys=True),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=image, pixdim=spacing),
            Spacingd(keys=label, pixdim=spacing, mode="nearest") if (label in input_keys) else Identityd(keys=input_keys, allow_missing_keys=True), 
            ScaleIntensityRangePercentilesd(
                keys=image, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ) if args.same_normalization else (
                ZScoreNormalized(keys=image, clip=True)
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0),
            #add ground truth extreme points to label
            AddExtremePointsChanneld(label_names = args.labels,keys = label, label_key = label,sigma = args.sigma,),
            AddGuidanceSignald(keys=label,sigma=args.sigma),
            SplitDimd(keys=('label')),
            CopyItemsd(keys=("label_0", "label_1"), times=1,
                    names=("label_seg", "label_ep"), allow_missing_keys=True)
    
        ]
        
    if args.debug:
        for i in range(len(t)):
            t[i] = TrackTimed(t[i])

    return t

def get_device(data):
    """
    Get the device information for the input data.

    Args:
        data: Input data for which the device information is retrieved.

    Returns:
        str: A string containing the device information.
    """
    return f"device - {data.device}"




def get_post_transforms_dual_dynunet(labels, *, save_pred=False, output_dir=None, pretransform=None):
    """
    Get the post transforms used for processing and saving predictions.

    Args:
        labels (list): List of label names.
        save_pred (bool): Flag to indicate whether to save predictions.
        output_dir (str): Output directory for saving predictions.
        pretransform (Compose): Pre-transform to be applied before inverting the prediction.

    Returns:
        Compose: A composition of transforms for post-processing and saving predictions.
    """
    cuda_device = torch.device("cuda:0")
    if save_pred:
        if output_dir is None:
            raise UserWarning("output_dir may not be empty when save_pred is enabled...")
        if pretransform is None:
            logger.warning("Make sure to add a pretransform here if you want the prediction to be inverted")

    input_keys = ("pred_ep",)
    t = [
        SplitDimd(keys=('image_target'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SplitDimd(keys=('image_source'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),
        CopyItemsd(keys=("pred_seg", "label_seg", "image_source_0", "image_target_0","pred_ep", "pred_ep_processed", "label_ep",), times=1,
                    names=("pred_seg_for_save", "seg_for_save", "image_for_save_source", "image_for_save_target", "pred_ep_for_save", "pred_ep_processed_for_save", "ep_for_save"), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        Activationsd(keys=("pred_seg",), softmax=True),
        AsDiscreted(
            keys="pred_seg_for_save",
            argmax=True,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        AsDiscreted(
            keys=("pred_seg", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        SaveImaged(
            keys=("pred_seg_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="pred_seg",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("seg_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="label_seg",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("image_for_save_source","image_for_save_target"),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="image",
            output_dtype=np.float32,
            separate_folder=True,
            resample=False,
            allow_missing_keys=True,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImagedSlices(
            keys=("pred_ep_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="pred_ep",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImagedSlices(
            keys=("pred_ep_processed_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="pred_ep_processed",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),


        SaveImagedSlices(
            keys=("ep_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="label_ep",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        ToTensord(keys=("label_ep", "pred_ep"), device=cuda_device, allow_missing_keys=True),

    ]
    return Compose(t)


def get_post_transforms(labels, *, save_pred=False, output_dir=None, pretransform=None):
    """
    Get the post transforms used for processing and saving predictions.

    Args:
        labels (list): List of label names.
        save_pred (bool): Flag to indicate whether to save predictions.
        output_dir (str): Output directory for saving predictions.
        pretransform (Compose): Pre-transform to be applied before inverting the prediction.

    Returns:
        Compose: A composition of transforms for post-processing and saving predictions.
    """
    cuda_device = torch.device("cuda:0")
    if save_pred:
        if output_dir is None:
            raise UserWarning("output_dir may not be empty when save_pred is enabled...")
        if pretransform is None:
            logger.warning("Make sure to add a pretransform here if you want the prediction to be inverted")

    input_keys = ("pred_seg",)
    t = [
        SplitDimd(keys=('image_target'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SplitDimd(keys=('image_source'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),


        CopyItemsd(keys=("pred_seg", "label_seg", "image_source_0", "image_target_0"), times=1,
                    names=("pred_seg_for_save", "seg_for_save", "image_for_save_source", "image_for_save_target"), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        Activationsd(keys=("pred_seg",), softmax=True),
        AsDiscreted(
            keys="pred_seg_for_save",
            argmax=True,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        AsDiscreted(
            keys=("pred_seg", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        SaveImaged(
            keys=("pred_seg_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="pred_seg",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("seg_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="label_seg",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("image_for_save_source","image_for_save_target"),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="image",
            output_dtype=np.float32,
            separate_folder=True,
            resample=False,
            allow_missing_keys=True,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        ToTensord(keys=("image_source", "image_target", "label_seg", "pred_seg"), device=cuda_device, allow_missing_keys=True),
    ]
    return Compose(t)


def get_post_transforms_ep(labels, *, save_pred=False, output_dir=None, pretransform=None):
    """
    Get the post transforms used for processing and saving predictions.

    Args:
        labels (list): List of label names.
        save_pred (bool): Flag to indicate whether to save predictions.
        output_dir (str): Output directory for saving predictions.
        pretransform (Compose): Pre-transform to be applied before inverting the prediction.

    Returns:
        Compose: A composition of transforms for post-processing and saving predictions.
    """
    cuda_device = torch.device("cuda:0")
    if save_pred:
        if output_dir is None:
            raise UserWarning("output_dir may not be empty when save_pred is enabled...")
        if pretransform is None:
            logger.warning("Make sure to add a pretransform here if you want the prediction to be inverted")

    input_keys = ("pred_ep",)
    t = [
        SplitDimd(keys=('image_target'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SplitDimd(keys=('image_source'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),
        CopyItemsd(keys=("image_source_0", "image_target_0","pred_ep", "pred_ep_processed", "label_ep",), times=1,
                    names=("image_for_save_source", "image_for_save_target", "pred_ep_for_save", "pred_ep_processed_for_save", "ep_for_save"), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImaged(
            keys=("image_for_save_source","image_for_save_target"),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="image",
            output_dtype=np.float32,
            separate_folder=True,
            resample=False,
            allow_missing_keys=True,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImagedSlices(
            keys=("pred_ep_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="pred_ep",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SaveImagedSlices(
            keys=("pred_ep_processed_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="pred_ep_processed",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),


        SaveImagedSlices(
            keys=("ep_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="label_ep",
            output_dtype=np.uint8,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        ToTensord(keys=("label_ep", "pred_ep"), device=cuda_device, allow_missing_keys=True),

    ]
    return Compose(t)



def get_AMOS_file_list(args, dataset) -> List[List, List, List]:
    """
    Get file lists for AutoPET dataset.

    Args:
        args: Command line arguments.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        A tuple containing lists of training, validation, and test data dictionaries.
        Each dictionary contains the paths to the image and label files.
    """
    if dataset == 'source':
        if args.source_dataset == "image_ct":
            image_type = "CT"
        elif args.source_dataset == "image_mri":
            image_type = "MRI"
    elif dataset == 'target':
        if args.target_dataset == "image_ct":
            image_type = "CT"
        elif args.target_dataset == "image_mri":
            image_type = "MRI"

    train_images = sorted(glob.glob(os.path.join(args.input_dir, image_type, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.input_dir, image_type,"labelsTr", "*.nii.gz")))

    test_images = sorted(glob.glob(os.path.join(args.input_dir, image_type,"imagesTs", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(args.input_dir, image_type,"labelsTs", "*.nii.gz")))

    if(dataset=='source'):
        train_data = [
            {"image_source": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
        ]
        val_data = [{"image_source": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
        test_data = [{"image_source": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
    else:
        train_data = [
            {"image_target": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
        ]
        val_data = [{"image_target": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
        test_data = [{"image_target": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]

    return train_data, val_data, test_data


def get_filename_without_extensions(nifti_path):
    """
    Extracts the filename without extensions from a given NIfTI file path.

    Args:
        nifti_path (str): The path to the NIfTI file.

    Returns:
        str: The filename without extensions.

    Example:
        "SUV.nii.gz" to "SUV". The resulting string represents the filename without extensions.
    """
    return Path(os.path.basename(nifti_path)).with_suffix("").with_suffix("").name





def get_data(args, dataset):
    """
    Retrieves data for training, validation, and testing based on the specified dataset in the command-line arguments.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.

    Returns:
        Tuple containing lists of training, validation, and test data dictionaries, each with "image" and "label" keys.
    """
    logger.info(f"{args.dataset=}")

    test_data = []

    if args.dataset == "AMOS":
        train_data, val_data, test_data = get_AMOS_file_list(args, dataset)

    if args.train_on_all_samples:
        train_data += val_data
        val_data = train_data
        test_data = train_data
        logger.warning("All validation data has been added to the training. Validation and Testing on them no longer makes sense.")

    logger.info(f"len(train_data): {len(train_data)}, len(val_data): {len(val_data)}, len(test_data): {len(test_data)}")

    # For debugging with small dataset size
    # train_data = train_data[0 : args.limit] if args.limit else train_data
    # val_data = val_data[0 : args.limit] if args.limit else val_data
    # test_data = test_data[0 : args.limit] if args.limit else test_data

    return train_data, val_data, test_data



def get_train_loader(args, pre_transforms_train_source, pre_transforms_train_target):
    """
    Retrieves a DataLoader for training based on the specified command-line arguments and pre-transforms.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.
        pre_transforms_train: Pre-transforms to be applied to the training data.

    Returns:
        DataLoader for training with asynchronous data loading using PersistentDataset and ThreadDataLoader.
    """
    train_data_source, val_data_source, test_data = get_data(args, 'source')
    train_data_target, val_data_target, test_data = get_data(args, 'target')


    total_l_source = len(train_data_source) + len(val_data_source)
    total_l_target = len(train_data_target) + len(val_data_target)
    total_l = total_l_source + total_l_target

    train_ds_source = PersistentDataset(train_data_source, pre_transforms_train_source, cache_dir=args.cache_dir)
    train_ds_target = PersistentDataset(train_data_target, pre_transforms_train_target, cache_dir=args.cache_dir)

    alternatingSampler = AlternatingSampler(train_ds_source, train_ds_target)
    train_ds = ConcatDataset([train_ds_source, train_ds_target])

    train_loader = ThreadDataLoader(
        train_ds,
        buffer_timeout=0.02,
        shuffle=False,
        sampler = alternatingSampler,
        num_workers=args.num_workers,
        batch_size=1,

    )
    logger.info("{} :: Total Records used for Training is: {}/{}".format(args.gpu, len(train_ds), total_l))
    return train_loader


def get_train_loader_separate(args, pre_transforms_train_source, pre_transforms_train_target):
    """
    Retrieves a DataLoader for training based on the specified command-line arguments and pre-transforms.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.
        pre_transforms_train: Pre-transforms to be applied to the training data.

    Returns:
        DataLoader for training with asynchronous data loading using PersistentDataset and ThreadDataLoader.
    """
    train_data_source, val_data_source, test_data = get_data(args, 'source')
    train_data_target, val_data_target, test_data = get_data(args, 'target')


    total_l_source = len(train_data_source) + len(val_data_source)
    total_l_target = len(train_data_target) + len(val_data_target)
    total_l = total_l_source + total_l_target

    train_ds_source = PersistentDataset(train_data_source, pre_transforms_train_source, cache_dir=args.cache_dir)
    train_ds_target = PersistentDataset(train_data_target, pre_transforms_train_target, cache_dir=args.cache_dir)

    alternatingSampler = SpecificSampler(train_ds_source, train_ds_target)

    train_loader = ThreadDataLoader(
        train_ds_source,
        shuffle=False,
        sampler=alternatingSampler,
        num_workers=args.num_workers,
        batch_size=1,

    )
    logger.info("{} :: Total Records used for Training is: {}/{}".format(args.gpu, len(train_ds_source), total_l))
    return train_loader


def get_val_loader_separate(args, pre_transforms_val, dataset:str):
    """
    Retrieves a DataLoader for validation based on the specified command-line arguments and pre-transforms.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.
        pre_transforms_val: Pre-transforms to be applied to the validation data.

    Returns:
        DataLoader for validation with asynchronous data loading using PersistentDataset and ThreadDataLoader.
    """
    train_data, val_data, test_data = get_data(args, dataset)


    total_l = len(train_data + val_data)

    val_ds = PersistentDataset(val_data, pre_transforms_val, cache_dir=args.cache_dir)

    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=args.num_workers,
        batch_size=1,
    )
    logger.info("{} :: Total Records used for Validation is: {}/{}".format(args.gpu, len(val_ds), total_l))

    return val_loader



def get_val_loader(args, pre_transforms_val_source, pre_transforms_val_target):
    """
    Retrieves a DataLoader for validation based on the specified command-line arguments and pre-transforms.

    Args:
        args: Command-line arguments specifying the dataset, input directory, and other options.
        pre_transforms_val: Pre-transforms to be applied to the validation data.

    Returns:
        DataLoader for validation with asynchronous data loading using PersistentDataset and ThreadDataLoader.
    """
    train_data_source, val_data_source, test_data = get_data(args, 'source')
    target, val_data_target, test_data = get_data(args, 'target')


    total_l_source = len(train_data_source) + len(val_data_source)
    total_l_target = len(target) + len(val_data_target)
    total_l = total_l_source + total_l_target


    val_ds_source = PersistentDataset(val_data_source, pre_transforms_val_source, cache_dir=args.cache_dir)
    val_ds_target = PersistentDataset(val_data_target, pre_transforms_val_target, cache_dir=args.cache_dir)
    val_ds = ConcatDataset([val_ds_source, val_ds_target])

    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=args.num_workers,
        batch_size=1,
    )
    logger.info("{} :: Total Records used for Validation is: {}/{}".format(args.gpu, len(val_ds), total_l))

    return val_loader


