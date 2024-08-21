from __future__ import annotations

import glob
import logging
import os
import shutil

# from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from monai.apps import CrossValidation
from monai.data import DataLoader, Dataset, ThreadDataLoader, partition_dataset
from monai.data.dataset import PersistentDataset
from monai.data.folder_layout import FolderLayout
from monai.transforms import (
    Activationsd,
    CenterSpatialCropd,
    Compose,
    CopyItemsd,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    Identityd,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    SaveImaged,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SignalFillEmptyd,
    Spacingd,
    ToDeviced,
    ToTensord,
)
from monai.utils.enums import CommonKeys

from sw_fastedit.utils.alternatingSampler import AlternatingSampler, SpecificSampler

from sw_fastedit.helper_transforms import (
    CheckTheAmountOfInformationLossByCropd,
    InitLoggerd,
    TrackTimed,
    threshold_foreground,
    cast_labels_to_zero_and_one,
)
from sw_fastedit.transforms import (
    AddExtremePointsChanneld,
    NormalizeLabelsInDatasetd,
    AddGuidanceSignald,
    SplitDimd,
    SaveImagedSlices,
    AsDiscreted,
    PrintShape,
)
from sw_fastedit.utils.helper import convert_mha_to_nii, convert_nii_to_mha

logger = logging.getLogger("sw_fastedit")


PET_dataset_names = ["AutoPET", "AutoPET2", "AutoPET_merged", "HECKTOR", "AutoPET2_Challenge", "AMOS"]



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
    AUTOPET_SPACING = (2.03642011, 2.03642011, 3.0)
    MSD_SPLEEN_SPACING = (2 * 0.79296899, 2 * 0.79296899, 5.0)
    HECKTOR_SPACING = (2.03642011, 2.03642011, 3.0)
    #2 options from the AMOS22 challenge paper Fabian Isensee
    #AMOS_SPACING = (1.03642011, 2.03642011, 1.0)
    #AMOS_SPACING = (3*0.69, 3*0.69, 3*2.0)
    AMOS_SPACING = (3*1.5, 3*1.0, 3*1.0) 
    # TODO Franzi Define AMOS Spacings

    if args.dataset == "AutoPET" or args.dataset == "AutoPET2" or args.dataset == "AutoPET2_Challenge":
        return AUTOPET_SPACING
    elif args.dataset == "HECKTOR":
        return HECKTOR_SPACING
    elif args.dataset == "MSD_Spleen":
        return MSD_SPLEEN_SPACING
    elif args.dataset == "AMOS":
        return AMOS_SPACING
    else:
        raise UserWarning(f"No valid dataset found: {args.dataset}")

#TODO: Franzi change transforms for AMOS dataset here
def get_pre_transforms_train_as_list(labels: Dict, device, args, input_keys, label, image):
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
    if args.dataset in PET_dataset_names:
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
            #PrintShape(keys=input_keys, prev_transform='load image'),
            ToTensord(keys=input_keys, device=cpu_device, track_meta=True),
            #PrintShape(keys=input_keys, prev_transform='to tensor'),
            EnsureChannelFirstd(keys=input_keys),
            #PrintShape(keys=input_keys, prev_transform='ensure channel first'),
            NormalizeLabelsInDatasetd(keys=label, labels=labels, device=cpu_device),
            #PrintShape(keys=input_keys, prev_transform='normalize labels'),
            Orientationd(keys=input_keys, axcodes="RAS"),
            #PrintShape(keys=input_keys, prev_transform='orientation'),
            Spacingd(keys=image, pixdim=spacing),
            #PrintShape(keys=input_keys, prev_transform='spacing image'),
            Spacingd(keys=label, pixdim=spacing, mode="nearest") if (label in input_keys) else Identityd(keys=input_keys),
            #PrintShape(keys=input_keys, prev_transform='spacing label'),
            CropForegroundd(
                keys=input_keys,
                source_key=image,
                select_fn=threshold_foreground,
            )
            if args.crop_foreground
            else Identityd(keys=input_keys, allow_missing_keys=True),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically
            ScaleIntensityRanged(keys=image, a_min=-45, a_max=105, b_min=0.0, b_max=1.0, clip=True)
            if args.scale_intensity_ranged
            else ScaleIntensityRangePercentilesd(
                keys=image, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0)
            if args.inferer == "SimpleInferer"
            else Identityd(keys=input_keys, allow_missing_keys=True),  # UNet needs this, 32 for 6 layers, for 7 at least 64        
            #PrintShape(keys=input_keys, prev_transform='divisible padd'),    
            #Data augmentation
            RandFlipd(keys=input_keys, spatial_axis=[0], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[1], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=input_keys, prob=0.10, max_k=3),
            #PrintShape(keys=input_keys, prev_transform='augmentation'),
            #For debugging, replacing nan
            SignalFillEmptyd(input_keys),
            #PrintShape(keys=input_keys, prev_transform='signal fill empty'),
            #add ground truth extreme points to label
            AddExtremePointsChanneld(label_names = args.labels,keys = label, label_key = label,sigma = args.sigma,),
            #PrintShape(keys=input_keys, prev_transform='add extreme points'),
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

#TODO: Franzi for AMOS dataset, SAME AS TRAIN BUT WITHOUT AUGMENTATION
def get_pre_transforms_val_as_list(labels: Dict, device, args, input_keys, label, image):
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
    if args.dataset in PET_dataset_names:
        t = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            #PrintShape(keys=input_keys, prev_transform='load image'),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys=label, labels=labels, device=cpu_device, allow_missing_keys=True),
            # Only for HECKTOR, filter out the values > 1
            #Lambdad(keys=label, func=cast_labels_to_zero_and_one) if (args.dataset == "HECKTOR") else Identityd(keys=input_keys, allow_missing_keys=True),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=image, pixdim=spacing),
            Spacingd(keys=label, pixdim=spacing, mode="nearest") if (label in input_keys) else Identityd(keys=input_keys, allow_missing_keys=True),
            # CheckTheAmountOfInformationLossByCropd(
            #     keys=label, roi_size=args.val_crop_size, crop_foreground=args.crop_foreground
            # )
            # if label in input_keys
            # else Identityd(keys=input_keys, allow_missing_keys=True),
            CropForegroundd(
                keys=input_keys,
                source_key=image,
                select_fn=threshold_foreground,
            )
            if args.crop_foreground
            else Identityd(keys=input_keys, allow_missing_keys=True),
            # CenterSpatialCropd(keys=input_keys, roi_size=args.val_crop_size)
            # if args.val_crop_size is not None
            # else Identityd(keys=input_keys, allow_missing_keys=True),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically
            ScaleIntensityRanged(keys=image, a_min=-45, a_max=105, b_min=0.0, b_max=1.0, clip=True)
            if args.scale_intensity_ranged
            else ScaleIntensityRangePercentilesd(
                keys=image, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0)
            if args.inferer == "SimpleInferer"
            else Identityd(keys=input_keys, allow_missing_keys=True),
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

    input_keys = ("pred_ep",)
    t = [
        SplitDimd(keys=('image_target'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        SplitDimd(keys=('image_source'), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),


        CopyItemsd(keys=("pred_seg", "pred_ep", "pred_ep_processed", "label_ep", "label_seg", "image_source_0", "image_target_0"), times=1,
                    names=("pred_seg_for_save", "pred_ep_for_save", "pred_ep_processed_for_save", "ep_for_save", "seg_for_save", "image_for_save", "image_for_save"), allow_missing_keys=True)
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        #PrintKeys(),

        
        # Invertd(
        #     keys=("pred_seg_for_save", "label_for_save",),
        #     orig_keys="image",
        #     nearest_interp=False,
        #     transform=pretransform,
        # )
        # if (save_pred and pretransform is not None)
        # else Identityd(keys=input_keys, allow_missing_keys=True),
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

        SaveImaged(
            keys=("image_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="image",
            output_dtype=np.float32,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        ToTensord(keys=("image_source", "image_target", "label_seg", "label_ep", "pred_seg", "pred_ep"), device=cuda_device, allow_missing_keys=True),

        
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



        CopyItemsd(keys=("pred_ep", "pred_ep_processed", "label_ep",), times=1,
                    names=( "pred_ep_for_save", "pred_ep_processed_for_save", "ep_for_save"), allow_missing_keys=True)
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


def get_post_transforms_dynunet(labels, *, save_pred=False, output_dir=None, pretransform=None):
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
                    names=("pred_seg_for_save", "seg_for_save", "image_for_save", "image_for_save"), allow_missing_keys=True)
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
            keys=("image_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix="image",
            output_dtype=np.float32,
            separate_folder=True,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),

        ToTensord(keys=("image_source", "image_target", "label_seg", "pred_seg"), device=cuda_device, allow_missing_keys=True),
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
    train_data = train_data[0 : args.limit] if args.limit else train_data
    val_data = val_data[0 : args.limit] if args.limit else val_data
    test_data = test_data[0 : args.limit] if args.limit else test_data

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
    alternatingSampler = SpecificSampler(train_ds_source)

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



def get_metrics_loader(args, file_glob="*.nii.gz"):
    """
    Generates a metrics loader for evaluating predictions based on specified command-line arguments.

    Args:
        args: Command-line arguments specifying the labels directory, predictions directory, and other options.
        file_glob (str): File pattern to match predictions.

    Returns:
        List of dictionaries containing label and prediction file paths for evaluation.
    """
    labels_dir = args.labels_dir
    predictions_dir = args.predictions_dir
    predictions_glob = os.path.join(predictions_dir, file_glob)
    test_predictions = sorted(glob.glob(predictions_glob))
    test_datalist = []

    for pred_file_name in test_predictions:
        logger.info(f"{pred_file_name=}")
        assert os.path.exists(pred_file_name)
        file_name = get_filename_without_extensions(pred_file_name)
        label_file_name = os.path.join(labels_dir, f"{file_name}{file_glob[1:]}")
        assert os.path.exists(label_file_name)
        logger.info(f"{label_file_name=}")
        test_datalist.append({CommonKeys.LABEL: label_file_name, CommonKeys.PRED: pred_file_name})

    test_datalist = test_datalist[0 : args.limit] if args.limit else test_datalist
    total_l = len(test_datalist)
    assert total_l > 0

    logger.info("{} :: Total Records used for Dataloader is: {}".format(args.gpu, total_l))

    return test_datalist


def get_metrics_transforms(device, labels, args):
    """
    Generates a set of transforms for processing predictions during metrics calculation.

    Args:
        device: Device to which tensors are moved.
        labels: Dictionary containing labels information.
        args: Command-line arguments specifying debugging options, logging, and output directory.

    Returns:
        Composed set of transforms for metrics calculation.
    """
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    t = [
        InitLoggerd(loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir),
        LoadImaged(
            keys=["pred", "label"],
            reader="ITKReader",
            image_only=False,
        ),
        ToDeviced(keys=["pred", "label"], device=device),
        EnsureChannelFirstd(keys=["pred", "label"]),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(False, False),
            to_onehot=(len(labels), len(labels)),
        ),
    ]

    return Compose(t)
