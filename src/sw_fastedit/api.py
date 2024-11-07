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

# Code extension and modification by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology #
# zdravko.marinov@kit.edu #
# Further code extension and modification by B.Sc. Matthias Hadlich, Karlsuhe Institute of Techonology #
# matthiashadlich@posteo.de #
# Further code extension and modification by B.Sc. Franziska Seiz, Karlsuhe Institute of Techonology #
# franzi.seiz96@gmail.com #

from __future__ import annotations

import logging
import os
import random
from collections import OrderedDict
from functools import reduce
from pickle import dump
from typing import Iterable, List
import sys
from itertools import chain

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from ignite.engine import Events
from ignite.handlers import TerminateOnNan
from monai.data import set_track_meta
from sw_fastedit.utils.trainer import  SupervisedTrainerEp, SupervisedTrainerDynUnet, SupervisedTrainerPada, SupervisedTrainerDextr, SupervisedTrainerDualDynUNet, SupervisedTrainerUgda
from sw_fastedit.utils.evaluator import SupervisedEvaluatorEp, SupervisedEvaluatorDynUnet, SupervisedEvaluatorPada, SupervisedEvaluatorDextr, SupervisedEvaluatorDualDynUnet, SupervisedEvaluatorUgda
from sw_fastedit.utils.validation_handler import ValidationHandler
from monai.handlers import (
    CheckpointLoader,
    CheckpointSaver,
    GarbageCollector,
    IgniteMetricHandler,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    from_engine,
)
from monai.handlers.regression_metrics import MeanSquaredError
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss, DiceLoss
from monai.networks.nets.dynunet import DynUNet
from monai.optimizers.novograd import Novograd
from monai.transforms import Compose
from monai.utils import set_determinism


from sw_fastedit.data import (
    get_post_transforms_dual_dynunet,
    get_post_transforms,
    get_post_transforms_ep,
    get_pre_transforms_train_as_list_ct,
    get_pre_transforms_train_as_list_mri,
    get_pre_transforms_val_as_list_ct,
    get_pre_transforms_val_as_list_mri,
    get_train_loader,
    get_train_loader_separate,
    get_val_loader_separate,
)

from sw_fastedit.discriminator import Discriminator
from sw_fastedit.utils.helper import count_parameters, is_docker, run_once, handle_exception



logger = logging.getLogger("sw_fastedit")
output_dir = None


def get_optimizer(optimizer: str, lr: float, networks):
    """
    Get an optimizer for the given neural network.

    Parameters:
        optimizer (str): The optimizer to use. Options: "Novograd" or "Adam" (default).
        lr (float): The learning rate for the optimizer.
        network: The neural network for which the optimizer is created.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer with the provided learning rate.

    Example:
        # Get an Adam optimizer with a learning rate of 0.001 for a neural network
        optimizer = get_optimizer("Adam", 0.001, my_neural_network)
    """

    params = chain(*[network.parameters() for network in networks])

    if optimizer == "Novograd":
        optimizer = Novograd(params, lr)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr)
    return optimizer


def get_loss_function(loss_args, loss_kwargs=None):
    """
    Get a loss function for a semantic segmentation task.

    Parameters:
        loss_args (str): The type of loss function. Options: "DiceCELoss" or "DiceLoss".
        loss_kwargs (dict, optional): Additional keyword arguments for the loss function.
        Default loss_kwargs: {squared_pred: True, include_background:True}
        -   squared_pred enables faster convergence, possibly even better results in the long run

    Returns:
        torch.nn.modules.loss: A callable instance of the specified loss function.

    Example:
        # Get a DiceCELoss with default arguments
        loss_fn = get_loss_function("DiceCELoss")

        # Get a DiceLoss with custom arguments
        custom_loss_kwargs = {"squared_pred": False, "include_background": False}
        loss_fn_custom = get_loss_function("DiceLoss", loss_kwargs=custom_loss_kwargs)

    TODO Franzi:
        # Define the Losses from the paper (L_ext, L_seg, L_d, L_adv)
    """
    if loss_kwargs is None:
        loss_kwargs = {}
    if loss_args == "DiceCELoss":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, **loss_kwargs)
    if loss_args == "BCE":
        loss_function = nn.BCEWithLogitsLoss()
    if loss_args == "MSELoss":
        loss_function = nn.MSELoss()

    return loss_function


def get_network(labels: Iterable, discriminator: bool = True, extreme_points: bool = True, segmentation: bool = True):

    """
    Get a network for semantic segmentation.

    Parameters:
        network_str (str): The type of network. Options: "dynunet", "smalldynunet", "bigdynunet", "hugedynunet".
        labels (Iterable): List of label names.
        non_interactive (bool, optional): Flag indicating whether the network is used in non-interactive mode.

    Returns:
        nn.Module: An instance of the specified U-Net-based neural network.

    Additional Information:
        in_channels (int): The input channels for the network. For interactive runs, it is 1 + len(labels),
            where each additional channel represents a guidance signal per label with the size of the image.
            For non-interactive runs, it is 1, representing the image.
        out_channels (int): The number of output channels, equal to len(labels).

    Example:
        # Get a DynUNet with default configuration
        my_labels = ["tumor", "background"]
        unet_model = get_network("dynunet", my_labels)

    TODO Franzi:
        # Define the model architecture as in the paper
    """


    in_channels = 2 #1 for organ and 1 for extreme points (generated or interactive)
    out_channels = len(labels)
    networks = []
    #all networks should always be defined to make loading and saving of networks consistent
    
    #extreme point network
    networks.append(DynUNet(
        spatial_dims=3,
        in_channels=1, #image as input
        out_channels=1,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    ))
    # segmentation network
    if (extreme_points):
        networks.append(DynUNet(
            spatial_dims=3,
            in_channels=2, #extreme point output/ground truth ep + image as input
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        ))
        #discriminator network, extreme are not passed as an extra channel (pada)
        networks.append(Discriminator(num_in_channels=2))
    else:
    # segmentation network without extreme points
        networks.append(DynUNet(
            spatial_dims=3,
            in_channels=1, #only image as input
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        ))
        #discriminator network, extreme points are not passed as an extra channel
        networks.append(Discriminator(num_in_channels=2))


    parameters = 0
    if extreme_points:
        parameters += count_parameters(networks[0])
    if segmentation:
        parameters += count_parameters(networks[1])
    if discriminator:
        parameters += count_parameters(networks[2])

    logger.info(f"Selected network {networks.__class__.__qualname__}")
    logger.info(f"Number of parameters: {parameters:,}")


    return networks

def get_network_ugda(labels: Iterable, discriminator: bool = True, extreme_points: bool = True, segmentation: bool = True):

    """
    Get a network for semantic segmentation.

    Parameters:
        network_str (str): The type of network. Options: "dynunet", "smalldynunet", "bigdynunet", "hugedynunet".
        labels (Iterable): List of label names.
        non_interactive (bool, optional): Flag indicating whether the network is used in non-interactive mode.

    Returns:
        nn.Module: An instance of the specified U-Net-based neural network.

    Additional Information:
        in_channels (int): The input channels for the network. For interactive runs, it is 1 + len(labels),
            where each additional channel represents a guidance signal per label with the size of the image.
            For non-interactive runs, it is 1, representing the image.
        out_channels (int): The number of output channels, equal to len(labels).

    Example:
        # Get a DynUNet with default configuration
        my_labels = ["tumor", "background"]
        unet_model = get_network("dynunet", my_labels)

    TODO Franzi:
        # Define the model architecture as in the paper
    """


    in_channels = 2 #1 for organ and 1 for extreme points (generated or interactive)
    out_channels = len(labels)
    networks = []
    #all networks should always be defined to make loading and saving of networks consistent
    
    #extreme point network
    networks.append(DynUNet(
        spatial_dims=3,
        in_channels=1, #image as input
        out_channels=1,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    ))
    # segmentation network
    if (extreme_points):
        networks.append(DynUNet(
            spatial_dims=3,
            in_channels=2, #extreme point output/ground truth ep + image as input
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        ))
        #discriminator network, extreme are passed as an extra channel (ugda)
        networks.append(Discriminator(num_in_channels=3))


    parameters = 0
    if extreme_points:
        parameters += count_parameters(networks[0])
    if segmentation:
        parameters += count_parameters(networks[1])
    if discriminator:
        parameters += count_parameters(networks[2])

    logger.info(f"Selected network {networks.__class__.__qualname__}")
    logger.info(f"Number of parameters: {parameters:,}")


    return networks





def get_inferers():
    """
    Retrieves training and evaluation inferers based on the specified inference strategy.

    Args:
        inferer (str): The type of inferer, either "SimpleInferer" or "SlidingWindowInferer".
        sw_roi_size: Region of interest size for the sliding window strategy.
        train_crop_size: Crop size for training data.
        val_crop_size: Crop size for validation data.
        train_sw_batch_size: Batch size for training with sliding window.
        val_sw_batch_size: Batch size for validation with sliding window.
        train_sw_overlap (float, optional): SW-overlap ratio for training with sliding window (default is 0.25).
        val_sw_overlap (float, optional): SW-overlap ratio for validation with sliding window (default is 0.25).
        cache_roi_weight_map (bool, optional): Whether to pre-compute the ROI weight map. (default is True).
        device (str, optional): Device for computation, e.g., "cpu" or "cuda" (default is "cpu").
        sw_cpu_output (bool, optional): Enable sliding window output on the CPU (default is False).

    Returns:
        Tuple[Inferer, Inferer]: Training and evaluation inferers based on the specified strategy.
    """
    
    train_inferer = SimpleInferer()
    eval_inferer = SimpleInferer()

    return train_inferer, eval_inferer


def get_scheduler(optimizer, scheduler_str: str, epochs_to_run: int, eta_min: float):
    """
    Retrieves a learning rate scheduler based on the specified scheduler strategy.

    Args:
        optimizer: The optimizer for which the scheduler will be used.
        scheduler_str (str): The type of scheduler, one of "MultiStepLR", "PolynomialLR", or "CosineAnnealingLR".
        epochs_to_run (int): The total number of epochs to run during training.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: A learning rate scheduler instance based on the chosen strategy.

    Raises:
        ValueError: If an unsupported scheduler type is provided.
    """

    if scheduler_str == "PolynomialLR":
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs_to_run, power=2)
    elif scheduler_str == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_run, eta_min=eta_min)
    return lr_scheduler


def get_val_handlers(inferer: str, gpu_size: str, *,garbage_collector=True, non_interactive=False):
    """
    Retrieves a list of event handlers for validation in a MONAI training workflow.

    Args:
        sw_roi_size (List): The region of interest size for the sliding window strategy.
        inferer (str): The type of inferer, e.g., "SimpleInferer" or "SlidingWindowInferer".
        gpu_size (str): The GPU size, one of "large" or any other value, e.g., "small".
        garbage_collector (bool, optional): Whether to include the GarbageCollector event handler (default is True).
        non_interactive (bool, optional): Whether the training loop is non-interactive, e.g., without clicks (default is False).

    Returns:
        List[Event_Handler]: A list of event handlers for validation in a MONAI training workflow.

    Notes:
        - The returned list includes a StatsHandler and an optional GarbageCollector event handler.
        - The GarbageCollector event handler is triggered based on the provided parameters.

    Raises:
        ValueError: If an unsupported inferer type is provided.

    References:
        [1] https://github.com/Project-MONAI/MONAI/issues/3423

    TODO Franzi:
        # Set the iterations = 1 and it is done
    """
    #every_x_iterations = 1 

    #val_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED

    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if garbage_collector:
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event="epoch")
        val_handlers.append(iteration_gc)

    return val_handlers


def get_train_handlers(
    lr_scheduler,
    evaluator,
    val_freq,
    eval_only: bool,
    inferer: str,
    gpu_size: str,
    garbage_collector=True,
    non_interactive=False,
):
    """
    Retrieves a list of event handlers for training in a MONAI training workflow.

    Args:
        lr_scheduler: The learning rate scheduler for the optimizer.
        evaluator: The evaluator for validation during training.
        val_freq: The frequency of validation in terms of iterations or epochs.
        eval_only (bool): Flag indicating if training is for evaluation only.
        sw_roi_size (List): The region of interest size for the sliding window strategy.
        inferer (str): The type of inferer, e.g., "SimpleInferer" or "SlidingWindowInferer".
        gpu_size (str): The GPU size, one of "large" or any other value.
        garbage_collector (bool, optional): Whether to include the GarbageCollector event handler (default is True).
        non_interactive (bool, optional): Whether the environment is non-interactive (default is False).

    Returns:
        List[Event_Handler]: A list of event handlers for training in a MONAI training workflow.

    Notes:
        - The returned list includes an LrScheduleHandler, ValidationHandler, StatsHandler, and an optional GarbageCollector event handler.
        - The GarbageCollector event handler is triggered based on the provided parameters.

    Raises:
        ValueError: If an unsupported inferer type is provided.

    References:
        [1] https://github.com/Project-MONAI/MONAI/issues/3423

    """


    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(
            validator=evaluator,
            interval=val_freq,
            epoch_level=(not eval_only),
        ),
        
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if garbage_collector:
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event='epoch')
        train_handlers.append(iteration_gc)

    return train_handlers

def get_train_handlers_separate(
    lr_scheduler,
    evaluator_source,
    evaluator_target,
    val_freq,
    eval_only: bool,
    inferer: str,
    gpu_size: str,
    garbage_collector=True,
    non_interactive=False,
):
    """
    Retrieves a list of event handlers for training in a MONAI training workflow.

    Args:
        lr_scheduler: The learning rate scheduler for the optimizer.
        evaluator: The evaluator for validation during training.
        val_freq: The frequency of validation in terms of iterations or epochs.
        eval_only (bool): Flag indicating if training is for evaluation only.
        sw_roi_size (List): The region of interest size for the sliding window strategy.
        inferer (str): The type of inferer, e.g., "SimpleInferer" or "SlidingWindowInferer".
        gpu_size (str): The GPU size, one of "large" or any other value.
        garbage_collector (bool, optional): Whether to include the GarbageCollector event handler (default is True).
        non_interactive (bool, optional): Whether the environment is non-interactive (default is False).

    Returns:
        List[Event_Handler]: A list of event handlers for training in a MONAI training workflow.

    Notes:
        - The returned list includes an LrScheduleHandler, ValidationHandler, StatsHandler, and an optional GarbageCollector event handler.
        - The GarbageCollector event handler is triggered based on the provided parameters.

    Raises:
        ValueError: If an unsupported inferer type is provided.

    References:
        [1] https://github.com/Project-MONAI/MONAI/issues/3423
    """


    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(
            validator=evaluator_source,
            interval=val_freq,
            epoch_level=(not eval_only),
        ),
        ValidationHandler(
            validator=evaluator_target,
            interval=val_freq,
            epoch_level=(not eval_only),
        ),
        
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if garbage_collector:
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event='epoch')
        train_handlers.append(iteration_gc)

    return train_handlers


def get_train_handlers_separate_adv(
    lr_scheduler,
    evaluator_source,
    evaluator_target,
    val_freq,
    eval_only: bool,
    inferer: str,
    gpu_size: str,
    garbage_collector=True,
    non_interactive=False,
):
    """
    Retrieves a list of event handlers for training in a MONAI training workflow.

    Args:
        lr_scheduler: The learning rate scheduler for the optimizer.
        evaluator: The evaluator for validation during training.
        val_freq: The frequency of validation in terms of iterations or epochs.
        eval_only (bool): Flag indicating if training is for evaluation only.
        sw_roi_size (List): The region of interest size for the sliding window strategy.
        inferer (str): The type of inferer, e.g., "SimpleInferer" or "SlidingWindowInferer".
        gpu_size (str): The GPU size, one of "large" or any other value.
        garbage_collector (bool, optional): Whether to include the GarbageCollector event handler (default is True).
        non_interactive (bool, optional): Whether the environment is non-interactive (default is False).

    Returns:
        List[Event_Handler]: A list of event handlers for training in a MONAI training workflow.

    Notes:
        - The returned list includes an LrScheduleHandler, ValidationHandler, StatsHandler, and an optional GarbageCollector event handler.
        - The GarbageCollector event handler is triggered based on the provided parameters.

    Raises:
        ValueError: If an unsupported inferer type is provided.

    References:
        [1] https://github.com/Project-MONAI/MONAI/issues/3423

    """

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler[0], print_lr=True),
        LrScheduleHandler(lr_scheduler=lr_scheduler[1], print_lr=True),
        ValidationHandler(
            validator=evaluator_source,
            interval=val_freq,
            epoch_level=(not eval_only),
        ),
        ValidationHandler(
            validator=evaluator_target,
            interval=val_freq,
            epoch_level=(not eval_only),
        ),
        
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if garbage_collector:
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event='epoch')
        train_handlers.append(iteration_gc)

    return train_handlers


def get_key_metric(metric, str_to_prepend="") -> OrderedDict:
    """
    Retrieves key metrics, particularly Mean Dice, for use in a MONAI training workflow.

    Args:
        str_to_prepend (str, optional): A string to prepend to the metric name (default is an empty string).

    Returns:
        OrderedDict: An ordered dictionary containing key metrics for training and evaluation.
    """
    key_metrics = OrderedDict()
    if (metric == 'dice'):
        key_metrics[f"{str_to_prepend}dice"] = MeanDice(output_transform=from_engine(["pred_seg", "label_seg"]), include_background=False, save_details=False)
    elif (metric == 'mse'):
        key_metrics[f"{str_to_prepend}mse"] = MeanSquaredError(output_transform=from_engine(["pred_ep", "label_ep"]))
    elif (metric == 'dice_mse'):
        key_metrics[f"{str_to_prepend}dice"] = MeanDice(output_transform=from_engine(["pred_seg", "label_seg"]), include_background=False, save_details=False)
        key_metrics[f"{str_to_prepend}mse"] = MeanSquaredError(output_transform=from_engine(["pred_ep", "label_ep"]))

    return key_metrics



def get_additional_metrics(labels, include_background=False, loss_kwargs=None, str_to_prepend=""):
    """
    Retrieves additional metrics, including DiceCELoss and SurfaceDiceMetric, for use in a MONAI training workflow.

    Args:
        labels: A list of class labels for the segmentation task.
        include_background (bool, optional): Whether to include a background class in the metric computation (default is False).
        loss_kwargs (dict, optional): Additional keyword arguments for configuring the loss function (default is None).
        str_to_prepend (str, optional): A string to prepend to the metric names (default is an empty string).

    Returns:
        OrderedDict: An ordered dictionary containing additional metrics for evaluation.

    Notes:
        - The returned dictionary includes metrics for DiceCELoss and SurfaceDiceMetric.
        - The metric names can be customized by providing a string to prepend.
        - Loss function and SurfaceDiceMetric are configured with specific parameters.

    """
    if loss_kwargs is None:
        loss_kwargs = {}
    mid = "with_bg_" if include_background else "without_bg_"


    additional_metrics = OrderedDict()

    bce_loss_dis = nn.BCEWithLogitsLoss()
    bce_loss_ignite = IgniteMetricHandler(
        loss_fn=bce_loss_dis,
        output_transform=from_engine(["gpred", "glabel"]),
        save_details=False
    )
    additional_metrics[f"{str_to_prepend}{bce_loss_dis.__class__.__name__.lower()}"] = bce_loss_ignite

    return additional_metrics



def get_trainer_ep(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerEp | None, SupervisedEvaluatorEp | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_1 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_2 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')


    post_transform = get_post_transforms_ep(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.labels, discriminator=False, extreme_points=True, segmentation=False)
    networks[0] = networks[0].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions =  get_loss_function(loss_args=args.loss_mse, loss_kwargs=loss_kwargs)
    
    optimizer = get_optimizer(args.optimizer, args.learning_rate_ep, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs, eta_min=args.eta_min_ep)


    val_key_metric = get_key_metric(metric='mse' ,str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )

    evaluator_1 = SupervisedEvaluatorEp(
        device=device,
        val_data_loader=val_loader_1,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        metric_cmp_fn=lambda current_metric, previous_best: current_metric < previous_best,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ), 
    )
    evaluator_2 = SupervisedEvaluatorEp(
        device=device,
        val_data_loader=val_loader_2,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        metric_cmp_fn=lambda current_metric, previous_best: current_metric < previous_best,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ), 
    )

    if args.source_dataset == 'image_ct':
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric(metric='mse', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate(
        lr_scheduler,
        evaluator_1,
        evaluator_2,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerEp(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        metric_cmp_fn=lambda current_metric, previous_best: current_metric < previous_best,
        train_handlers=train_handlers,

    )

    if not args.eval_only:
            save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
            }
    else:
        save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
        }


    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        key_metric_negative_sign=True,
        final_filename="pretrained_deepedit_" + args.target_dataset + "-final.pt",
        file_prefix=args.target_dataset,
    ).attach(evaluator_2)



    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            del save_dict["opt"]
            del save_dict["lr"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        print(checkpoint['opt']['param_groups'])
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_1)
            handler(evaluator_2)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr

    return trainer, [evaluator_1, evaluator_2], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics

def get_trainer_ep_source(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerEp | None, SupervisedEvaluatorEp | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    val_loader_1 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_2 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')


    post_transform = get_post_transforms_ep(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.labels, discriminator=False, extreme_points=True, segmentation=False)
    networks[0] = networks[0].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions =  get_loss_function(loss_args=args.loss_mse, loss_kwargs=loss_kwargs)
    
    optimizer = get_optimizer(args.optimizer, args.learning_rate_ep, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs, eta_min=args.eta_min_ep)


    val_key_metric = get_key_metric(metric='mse' ,str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )

    evaluator_1 = SupervisedEvaluatorEp(
        device=device,
        val_data_loader=val_loader_1,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        metric_cmp_fn=lambda current_metric, previous_best: current_metric < previous_best,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ), 
    )
    evaluator_2 = SupervisedEvaluatorEp(
        device=device,
        val_data_loader=val_loader_2,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        metric_cmp_fn=lambda current_metric, previous_best: current_metric < previous_best,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ), 
    )

    if args.source_dataset == 'image_ct':
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader_separate(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric(metric='mse', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate(
        lr_scheduler,
        evaluator_1,
        evaluator_2,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerEp(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        metric_cmp_fn=lambda current_metric, previous_best: current_metric < previous_best,
        train_handlers=train_handlers,

    )

    if not args.eval_only:
            save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
            }
    else:
        save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
        }


    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        key_metric_negative_sign=True,
        final_filename="pretrained_deepedit_" + args.target_dataset + "-final.pt",
        file_prefix=args.target_dataset,
    ).attach(evaluator_2)



    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            del save_dict["opt"]
            del save_dict["lr"]
        del save_dict["net_dis"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        print(checkpoint['opt']['param_groups'])
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_1)
            handler(evaluator_2)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr

    return trainer, [evaluator_1, evaluator_2], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics


def get_trainer_dynunet(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerDynUnet | None, SupervisedEvaluatorDynUnet | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_1 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_2 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.labels, discriminator=False, extreme_points=False, segmentation=True)
    networks[1] = networks[1].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = get_loss_function(loss_args=args.loss_dynunet, loss_kwargs=loss_kwargs)
    

    optimizer = get_optimizer(args.optimizer, args.learning_rate, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs, eta_min=args.eta_min)


    val_key_metric = get_key_metric(metric = 'dice', str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator_1 = SupervisedEvaluatorDynUnet(
        device=device,
        val_data_loader=val_loader_1,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )
    evaluator_2 = SupervisedEvaluatorDynUnet(  
        device=device,
        val_data_loader=val_loader_2,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )
    if args.source_dataset == 'image_ct':
        pre_transforms_train_1 = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_2 = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_1 = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_2 = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader(args, pre_transforms_train_source=pre_transforms_train_1, pre_transforms_train_target=pre_transforms_train_2)


    train_key_metric = get_key_metric(metric = 'dice', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate(
        lr_scheduler,
        evaluator_1,
        evaluator_2,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerDynUnet(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,

    )

    if not args.eval_only:
            save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
            }
    else:
        save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
        }



    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        final_filename=f"pretrained_deepedit_{args.target_dataset}" + args.network + "-final.pt",
    ).attach(evaluator_2)


    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            del save_dict["opt"]
            del save_dict["lr"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        print(checkpoint['opt']['param_groups'])
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_1)
            handler(evaluator_2)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr

    return trainer, [evaluator_1, evaluator_2], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics

def get_trainer_dynunet_source(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerDynUnet | None, SupervisedEvaluatorDynUnet | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_source = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_target = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.labels, discriminator=False, extreme_points=False, segmentation=True)
    networks[1] = networks[1].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = get_loss_function(loss_args=args.loss_dynunet, loss_kwargs=loss_kwargs)
    
    optimizer = get_optimizer(args.optimizer, args.learning_rate, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs, args.eta_min)

    val_key_metric = get_key_metric(metric = 'dice', str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator_source = SupervisedEvaluatorDynUnet( 
        device=device,
        val_data_loader=val_loader_source,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    evaluator_target = SupervisedEvaluatorDynUnet( 
        device=device,
        val_data_loader=val_loader_target,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    if args.source_dataset == 'image_ct':
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader_separate(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric(metric = 'dice', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate(
        lr_scheduler,
        evaluator_source,
        evaluator_target,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerDynUnet(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,

    )

    if not args.eval_only:
            save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
            }
    else:
        save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
        }




    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        final_filename="pretrained_deepedit_target" + args.network + "-final.pt",
    ).attach(evaluator_target)


    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            del save_dict["opt"]
            del save_dict["lr"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        print(checkpoint['opt']['param_groups'])
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_source)
            handler(evaluator_target)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr

        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")

    return trainer, [evaluator_source, evaluator_target], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics

def get_trainer_dualdynunet(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerDualDynUNet | None, SupervisedEvaluatorDualDynUnet | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_source = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_target = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms_dual_dynunet(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network_ugda(args.labels, discriminator=False, extreme_points=True, segmentation=True)
    networks[0] = networks[0].to(sw_device)
    networks[1] = networks[1].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = []
    loss_functions.append(get_loss_function(loss_args='MSELoss', loss_kwargs=loss_kwargs)) #segmentation loss
    loss_functions.append(get_loss_function(loss_args='DiceCELoss', loss_kwargs=loss_kwargs)) #adversarial loss and discriminator loss
   
    optimizer = (get_optimizer(args.optimizer, args.learning_rate, [networks[0], networks[1]]))
    lr_scheduler = (get_scheduler(optimizer, args.scheduler, args.epochs, eta_min=args.eta_min))

    val_key_metric = get_key_metric(metric = 'dice_mse', str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator_source = SupervisedEvaluatorDualDynUnet(
        args=args,
        device=device,
        val_data_loader=val_loader_source,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    evaluator_target = SupervisedEvaluatorDualDynUnet(
        args=args,
        device=device,
        val_data_loader=val_loader_target,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    if args.source_dataset == 'image_ct':
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader_separate(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)

    train_key_metric = get_key_metric(metric = 'dice_mse', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate(
        lr_scheduler,
        evaluator_source,
        evaluator_target,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerDualDynUNet(
        args=args,
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,
    )

    save_dict = {
        "trainer": trainer,
        "net_ep": networks[0],
        "net_seg": networks[1],
        "net_dis": networks[2],
        "opt": optimizer,
        "lr": lr_scheduler,
    }
    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        final_filename="pretrained_deepedit_target" + args.target_dataset + "-final.pt",
        file_prefix=args.target_dataset,
    ).attach(evaluator_target)


    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            print(save_dict.keys())
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            saved_trainer = save_dict["trainer"]
            del save_dict["opt"]
            del save_dict["lr_dis"]
            del save_dict["trainer"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"
        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_source)
            handler(evaluator_target)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr
            save_dict["trainer"] = saved_trainer


    return trainer, [evaluator_source, evaluator_target], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics

def get_trainer_dextr(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerDextr | None, SupervisedEvaluatorDextr | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_1 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_2 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.labels, discriminator=False, extreme_points=True, segmentation=True)
    networks[0] = networks[0].to(sw_device)
    networks[1] = networks[1].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = get_loss_function(loss_args="DiceCELoss", loss_kwargs=loss_kwargs)
    

    optimizer = get_optimizer(args.optimizer, args.learning_rate, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs, args.eta_min)


    val_key_metric = get_key_metric(metric = 'dice', str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )

    evaluator_1 = SupervisedEvaluatorDextr(
        device=device,
        val_data_loader=val_loader_1,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    evaluator_2 = SupervisedEvaluatorDextr(
        device=device,
        val_data_loader=val_loader_2,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    if args.source_dataset == 'image_ct':
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric(metric = 'dice', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate(
        lr_scheduler,
        evaluator_1,
        evaluator_2,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerDextr(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,
        args=args,
    )

    if not args.eval_only:
            save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
            }
    else:
        save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
        }

    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        final_filename=f"pretrained_deepedit_{args.target_dataset}" + args.network + "-final.pt",
        file_prefix=args.target_dataset,
    ).attach(evaluator_2)


    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            saved_trainer = save_dict["trainer"]
            del save_dict["opt"]
            del save_dict["lr"]
            del save_dict["trainer"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        print(checkpoint['opt']['param_groups'])
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_1)
            handler(evaluator_2)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr
            save_dict["trainer"] = saved_trainer

    return trainer, [evaluator_1, evaluator_2], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics

def get_trainer_dextr_source(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerDynUnet | None, SupervisedEvaluatorDynUnet | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_source = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_target = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.labels, discriminator=False, extreme_points=True, segmentation=True)
    networks[0] = networks[0].to(sw_device)
    networks[1] = networks[1].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = get_loss_function(loss_args="DiceCELoss", loss_kwargs=loss_kwargs)
    

    optimizer = get_optimizer(args.optimizer, args.learning_rate, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs, args.eta_min)


    val_key_metric = get_key_metric(metric = 'dice', str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )

    evaluator_1 = SupervisedEvaluatorDextr(
        device=device,
        val_data_loader=val_loader_source,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    evaluator_2 = SupervisedEvaluatorDextr(
        device=device,
        val_data_loader=val_loader_target,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    if args.source_dataset == 'image_ct':
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader_separate(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric(metric = 'dice', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate(
        lr_scheduler,
        evaluator_1,
        evaluator_2,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerDextr(
        args=args,
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,
    )

    if not args.eval_only:
            save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
            }
    else:
        save_dict = {
                "trainer": trainer,
                "net_ep": networks[0],
                "net_seg": networks[1],
                "net_dis": networks[2],
                "opt": optimizer,
                "lr": lr_scheduler,
        }

    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        final_filename=f"pretrained_deepedit_{args.target_dataset}" + args.network + "-final.pt",
        file_prefix=args.target_dataset,
    ).attach(evaluator_2)


    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    del save_dict["net_dis"]
    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            saved_trainer = save_dict["trainer"]
            del save_dict["opt"]
            del save_dict["lr"]
            del save_dict["trainer"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        print(checkpoint['opt']['param_groups'])
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_1)
            handler(evaluator_2)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr
            save_dict["trainer"] = saved_trainer

    return trainer, [evaluator_1, evaluator_2], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics


def get_trainer_pada(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerPada | None, SupervisedEvaluatorPada | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_source = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_target = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.labels, discriminator=True, extreme_points=args.extreme_points, segmentation=True)
    networks[1] = networks[1].to(sw_device)
    networks[2] = networks[2].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = []
    loss_functions.append(get_loss_function(loss_args='DiceCELoss', loss_kwargs=loss_kwargs)) #segmentation loss
    loss_functions.append(get_loss_function(loss_args='BCE', loss_kwargs=loss_kwargs)) #adversarial loss and discriminator loss
    optimizer = []
    lr_scheduler = []

    optimizer.append(get_optimizer(args.optimizer, args.learning_rate, [networks[1]]))
    lr_scheduler.append(get_scheduler(optimizer[0], args.scheduler, args.epochs, eta_min=args.eta_min))
    optimizer.append(get_optimizer(args.optimizer, args.learning_rate_dis, [networks[2]]))
    lr_scheduler.append(get_scheduler(optimizer[1], args.scheduler, args.epochs, eta_min=args.eta_min_dis))

    val_key_metric = get_key_metric(metric = 'dice', str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator_source = SupervisedEvaluatorPada(
        args=args,
        device=device,
        val_data_loader=val_loader_source,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    evaluator_target = SupervisedEvaluatorPada(
        args=args,
        device=device,
        val_data_loader=val_loader_target,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    if args.source_dataset == 'image_ct':
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric(metric = 'dice', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate_adv(
        lr_scheduler,
        evaluator_source,
        evaluator_target,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerPada(
        args=args,
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,
    )

    save_dict = {
        "trainer": trainer,
        "net_ep": networks[0],
        "net_seg": networks[1],
        "net_dis": networks[2],
        "opt_seg": optimizer[0],
        "opt_dis": optimizer[1],
        "lr_seg": lr_scheduler[0],
        "lr_dis": lr_scheduler[1],
    }
    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        final_filename="pretrained_deepedit_target" + args.target_dataset + "-final.pt",
        file_prefix=args.target_dataset,
    ).attach(evaluator_target)


    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            saved_opt_seg = save_dict["opt_seg"]
            saved_opt_dis = save_dict["opt_dis"]
            saved_lr_seg = save_dict["lr_seg"]
            saved_lr_dis = save_dict["lr_dis"]
            saved_net_ep = save_dict["net_ep"]
            saved_net_dis = save_dict["net_dis"]
            saved_trainers = save_dict["trainer"]
            del save_dict["opt_seg"]
            del save_dict["opt_dis"]
            del save_dict["lr_seg"]
            del save_dict["lr_dis"]
            del save_dict["net_ep"]
            del save_dict["net_dis"]
            del save_dict["trainer"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"
        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        #print(checkpoint['opt']['param_groups'])
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_source)
            handler(evaluator_target)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt_seg"] = saved_opt_seg
            save_dict["opt_dis"] = saved_opt_dis
            save_dict["lr_seg"] = saved_lr_seg
            save_dict["lr_dis"] = saved_lr_dis
            save_dict["net_ep"] = saved_net_ep
            save_dict["net_dis"] = saved_net_dis
            save_dict["trainer"] = saved_trainers


    return trainer, [evaluator_source, evaluator_target], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics




def get_trainer_ugda(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerUgda | None, SupervisedEvaluatorUgda | None, List]:
    """
    Retrieves a supervised trainer, evaluator, and related metrics for training in a MONAI deep learning workflow.

    Args:
        args: Command-line arguments and configuration settings.
        file_prefix (str, optional): Prefix to use for saving ensemble checkpoints (default is "").
        ensemble_mode (bool, optional): Flag indicating whether to run in ensemble mode (default is False).
        resume_from (str, optional): Path to a checkpoint file for resuming training (default is "None").

    Returns:
        Tuple[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
        - SupervisedTrainer: The trainer instance for training the neural network.
        - SupervisedEvaluator: The evaluator instance for validation during training.
        - List: List containing training key metric, additional metrics, validation key metric, and additional metrics.
    """
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")
    if args.source_dataset == 'image_ct':
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_val_source = Compose(get_pre_transforms_val_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_val_target = Compose(get_pre_transforms_val_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_source = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_target = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network_ugda(args.labels, discriminator=True, extreme_points=True, segmentation=True)
    networks[1] = networks[1].to(sw_device)
    networks[2] = networks[2].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = []
    loss_functions.append(get_loss_function(loss_args='DiceCELoss', loss_kwargs=loss_kwargs)) #segmentation loss
    loss_functions.append(get_loss_function(loss_args='BCE', loss_kwargs=loss_kwargs)) #adversarial loss seg + (ep) and discriminator loss
    optimizer = []
    lr_scheduler = []

    optimizer.append(get_optimizer(args.optimizer, args.learning_rate, [networks[1]]))
    lr_scheduler.append(get_scheduler(optimizer[0], args.scheduler, args.epochs, eta_min=args.eta_min))
    optimizer.append(get_optimizer(args.optimizer, args.learning_rate_dis, [networks[2]]))
    lr_scheduler.append(get_scheduler(optimizer[1], args.scheduler, args.epochs, eta_min=args.eta_min_dis))

    val_key_metric = get_key_metric(metric = 'dice', str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator_source = SupervisedEvaluatorUgda(
        args=args,
        device=device,
        val_data_loader=val_loader_source,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    evaluator_target = SupervisedEvaluatorUgda(
        args=args,
        device=device,
        val_data_loader=val_loader_target,
        networks=networks,
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )

    if args.source_dataset == 'image_ct':
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    else:
        pre_transforms_train_source = Compose(get_pre_transforms_train_as_list_mri(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
        pre_transforms_train_target = Compose(get_pre_transforms_train_as_list_ct(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric(metric = 'dice', str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers_separate_adv(
        lr_scheduler,
        evaluator_source,
        evaluator_target,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainerUgda(
        args=args,
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        networks=networks,
        optimizer=optimizer,
        loss_function=loss_functions,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,
    )

    save_dict = {
        "trainer": trainer,
        "net_ep": networks[0],
        "net_seg": networks[1],
        "net_dis": networks[2],
        "opt_seg": optimizer[0],
        "opt_dis": optimizer[1],
        "lr_seg": lr_scheduler[0],
        "lr_dis": lr_scheduler[1],
    }

    CheckpointSaver(
        save_dir=args.output_dir,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        final_filename="pretrained_deepedit_target" + args.target_dataset + "-final.pt",
        file_prefix=args.target_dataset,
    ).attach(evaluator_target)


    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            saved_opt_seg = save_dict["opt_seg"]
            saved_opt_dis = save_dict["opt_dis"]
            saved_lr_seg = save_dict["lr_seg"]
            saved_lr_dis = save_dict["lr_dis"]
            saved_net_ep = save_dict["net_ep"]
            saved_net_dis = save_dict["net_dis"]
            saved_trainers = save_dict["trainer"]
            del save_dict["opt_seg"]
            del save_dict["opt_dis"]
            del save_dict["lr_seg"]
            del save_dict["lr_dis"]
            del save_dict["net_ep"]
            del save_dict["net_dis"]
            del save_dict["trainer"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)
        #print(checkpoint.keys())
        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"
        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator_source)
            handler(evaluator_target)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt_seg"] = saved_opt_seg
            save_dict["opt_dis"] = saved_opt_dis
            save_dict["lr_seg"] = saved_lr_seg
            save_dict["lr_dis"] = saved_lr_dis
            save_dict["net_ep"] = saved_net_ep
            save_dict["net_dis"] = saved_net_dis
            save_dict["trainer"] = saved_trainers

    return trainer, [evaluator_source, evaluator_target], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics



@run_once
def init(args):
    """
    Initializes the environment with configuration settings and global variables.

    Args:
        args: Command-line arguments and configuration settings.

    Returns:
        None

    Notes:
        - This function is intended to be executed only once during the program's lifetime.

    """

    global output_dir
    # for OOM debugging
    output_dir = args.output_dir
    sys.excepthook = handle_exception

    if not is_docker():
        torch.set_num_threads(int(os.cpu_count() / 3))  # Limit number of threads to 1/3 of resources
    if args.limit_gpu_memory_to != -1:
        limit = args.limit_gpu_memory_to
        assert limit > 0 and limit < 1, f"Percentage GPU memory limit is invalid! {limit} > 0 or < 1"
        torch.cuda.set_per_process_memory_fraction(limit, args.gpu)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    # DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING..
    # I WARNED YOU..
    set_track_meta(True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    set_determinism(seed=args.seed)

    if not is_docker():
        with cp.cuda.Device(args.gpu):
            cp.random.seed(seed=args.seed)

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        np.seterr(all="raise")


def oom_observer(device, alloc, device_alloc, device_free):
    """
    Observes and handles out-of-memory (OOM) events on a specified CUDA device.

    Args:
        device (torch.device): The CUDA device where the OOM event occurred.
        alloc (int): Allocated memory in bytes.
        device_alloc (int): Device-allocated memory in bytes.
        device_free (int): Device-free memory in bytes.

    Returns:
        None

    Notes:
        - It logs a memory summary, saves an allocated state snapshot, and visualizes memory usage.
    """
    if device is not None and logger is not None:
        logger.critical(torch.cuda.memory_summary(device))
    # snapshot right after an OOM happened
    print("saving allocated state during OOM")
    print("Tips: \nReduce sw_batch_size if there is an OOM (maybe even roi_size)")
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open(f"{output_dir}/oom_snapshot.pickle", "wb"))
    torch.cuda.memory._save_memory_usage(filename=f"{output_dir}/memory.svg", snapshot=snapshot)
    torch.cuda.memory._save_segment_usage(filename=f"{output_dir}/segments.svg", snapshot=snapshot)
