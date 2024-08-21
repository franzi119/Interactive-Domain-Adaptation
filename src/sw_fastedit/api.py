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
from sw_fastedit.utils.trainer import SupervisedTrainer, SupervisedTrainerEp, SupervisedTrainerDynUnet, SupervisedTrainerDynUnetDa
from sw_fastedit.utils.evaluator import SupervisedEvaluator, SupervisedEvaluatorEp, SupervisedEvaluatorDynUnet, SupervisedEvaluatorDynUnetDa
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
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import SurfaceDiceMetric
from monai.networks.nets.dynunet import DynUNet
from monai.optimizers.novograd import Novograd
from monai.transforms import Compose
from monai.utils import set_determinism


from sw_fastedit.data import (
    get_post_transforms,
    get_post_transforms_ep,
    get_post_transforms_dynunet,
    get_pre_transforms_train_as_list,
    get_pre_transforms_val_as_list,
    get_train_loader,
    get_train_loader_separate,
    get_val_loader,
    get_val_loader_separate,
)
from sw_fastedit.dice_ce_l2 import DiceCeL2Loss
from sw_fastedit.discriminator import Discriminator
from sw_fastedit.utils.helper import count_parameters, is_docker, run_once, handle_exception


from sw_fastedit.transforms import SplitDimd

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
    if loss_args == "DiceCEL2Loss":
        loss_function = DiceCeL2Loss(to_onehot_y=True, softmax=True, **loss_kwargs)
    if loss_args == "DiceCELoss":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, **loss_kwargs)
    if loss_args == "CrossEntropy":
        loss_function = nn.BCEWithLogitsLoss()
    if loss_args == "MSELoss":
        loss_function = nn.MSELoss()

    return loss_function


def get_network(network_str: str, labels: Iterable, no_discriminator: bool = False, no_extreme_points: bool = False):

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


    in_channels = 2 #1 for liver and 1 for extreme points (generated or interactive)
    out_channels = len(labels)
    networks = []
    
    if network_str == "dynunet":
        networks.append(DynUNet(
            spatial_dims=3,
            in_channels=1, #only extreme points
            out_channels=1,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        ))

        if (not no_extreme_points):
            networks.append(DynUNet(
                spatial_dims=3,
                in_channels=2, #extreme point output + image
                out_channels=out_channels,
                kernel_size=[3, 3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
            ))
        else:
            networks.append(DynUNet(
                spatial_dims=3,
                in_channels=1, #extreme point output
                out_channels=out_channels,
                kernel_size=[3, 3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
            ))

        networks.append(Discriminator(num_in_channels=3))

    parameters = count_parameters(networks[1])
    if not no_extreme_points:
        parameters += count_parameters(networks[0])
    if(not no_discriminator):
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


def get_scheduler(optimizer, scheduler_str: str, epochs_to_run: int):
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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_run, eta_min=1e-8)
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

    TODO Franzi:
        # Set the iterations = 1 and it is done
    """

    # if sw_roi_size[0] <= 128:
    #     train_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED
    # else:
    #     train_trigger_event = (
    #         Events.ITERATION_COMPLETED(every=every_x_iterations*2) if gpu_size == "large" else Events.ITERATION_COMPLETED(every=every_x_iterations)
    #     )

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

    TODO Franzi:
        # Set the iterations = 1 and it is done
    """

    # if sw_roi_size[0] <= 128:
    #     train_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED
    # else:
    #     train_trigger_event = (
    #         Events.ITERATION_COMPLETED(every=every_x_iterations*2) if gpu_size == "large" else Events.ITERATION_COMPLETED(every=every_x_iterations)
    #     )

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


def get_key_metric(str_to_prepend="") -> OrderedDict:
    """
    Retrieves key metrics, particularly Mean Dice, for use in a MONAI training workflow.

    Args:
        str_to_prepend (str, optional): A string to prepend to the metric name (default is an empty string).

    Returns:
        OrderedDict: An ordered dictionary containing key metrics for training and evaluation.

    TODO Franzi:
        # Add MXA
    """
    key_metrics = OrderedDict()
    key_metrics[f"{str_to_prepend}dice"] = MeanDice(output_transform=from_engine(["pred_seg", "label_seg"]), include_background=False, save_details=False)
    key_metrics[f"{str_to_prepend}mse"] = MeanSquaredError(output_transform=from_engine(["pred_ep", "label_ep"]))
    return key_metrics

def get_key_metric_ep(str_to_prepend="") -> OrderedDict:
    """
    Retrieves key metrics, particularly Mean Dice, for use in a MONAI training workflow.

    Args:
        str_to_prepend (str, optional): A string to prepend to the metric name (default is an empty string).

    Returns:
        OrderedDict: An ordered dictionary containing key metrics for training and evaluation.

    """
    key_metrics = OrderedDict()
    key_metrics[f"{str_to_prepend}mse"] = MeanSquaredError(output_transform=from_engine(["pred_ep", "label_ep"]))
    return key_metrics

def get_key_metric_seg(str_to_prepend="") -> OrderedDict:
    """
    Retrieves key metrics, particularly Mean Dice, for use in a MONAI training workflow.

    Args:
        str_to_prepend (str, optional): A string to prepend to the metric name (default is an empty string).

    Returns:
        OrderedDict: An ordered dictionary containing key metrics for training and evaluation.

    TODO Franzi:
        # Add MXA
    """
    key_metrics = OrderedDict()
    key_metrics[f"{str_to_prepend}dice"] = MeanDice(output_transform=from_engine(["pred_seg", "label_seg"]), include_background=False, save_details=False)
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

    TODO Franzi:
        # Or add MXA here
    """
    if loss_kwargs is None:
        loss_kwargs = {}
    mid = "with_bg_" if include_background else "without_bg_"
    loss_function = DiceCELoss(softmax=True, **loss_kwargs)

    loss_function_metric_ignite = IgniteMetricHandler(
        loss_fn=loss_function,
        output_transform=from_engine(["pred_seg", "label_seg"]),
        save_details=False,
    )
    amount_of_classes = len(labels) if include_background else (len(labels) - 1)
    class_thresholds = (0.5,) * amount_of_classes
    surface_dice_metric = SurfaceDiceMetric(
        include_background=include_background,
        class_thresholds=class_thresholds,
        reduction="mean",
        get_not_nans=False,
        use_subvoxels=True,
    )
    surface_dice_metric_ignite = IgniteMetricHandler(
        metric_fn=surface_dice_metric,
        output_transform=from_engine(["pred_seg", "label_seg"]),
        save_details=False,
    )

    additional_metrics = OrderedDict()
    additional_metrics[f"{str_to_prepend}{loss_function.__class__.__name__.lower()}"] = loss_function_metric_ignite
    additional_metrics[f"{str_to_prepend}{mid}surface_dice"] = surface_dice_metric_ignite

    return additional_metrics




def get_supervised_evaluator(
    args,
    networks,
    inferer,
    device,
    val_loader,
    loss_function,
    post_transform,
    key_val_metric,
    additional_metrics,
) -> SupervisedEvaluator:
    """
    Retrieves a supervised evaluator for validation in a MONAI training workflow.

    Args:
        args: Command-line arguments and configuration settings.
        network: The model to be evaluated.
        inferer: The inference strategy or inferer.
        device: The computing device (e.g., "cuda" or "cpu") on which to run the evaluation.
        val_loader: The data loader for the validation dataset.
        loss_function: The loss function for evaluation.
        click_transforms: Data transforms for interactions (e.g., clicks).
        post_transform: Post-processing transformation for the output predictions.
        key_val_metric: The key validation metric.
        additional_metrics: Additional metrics for evaluation.

    Returns:
        SupervisedEvaluator: An instance of the supervised evaluator for validation.

    TODO Franzi:
        # max_interactions = 1 (all extreme points)
    """

    init(args)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        networks=networks,
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=key_val_metric,
        additional_metrics=additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )
    return evaluator


def get_supervised_evaluator_ep(
    args,
    networks,
    inferer,
    device,
    val_loader,
    loss_function,
    post_transform,
    key_val_metric,
    additional_metrics,
) -> SupervisedEvaluatorEp:
    """
    Retrieves a supervised evaluator for validation in a MONAI training workflow.

    Args:
        args: Command-line arguments and configuration settings.
        network: The model to be evaluated.
        inferer: The inference strategy or inferer.
        device: The computing device (e.g., "cuda" or "cpu") on which to run the evaluation.
        val_loader: The data loader for the validation dataset.
        loss_function: The loss function for evaluation.
        click_transforms: Data transforms for interactions (e.g., clicks).
        post_transform: Post-processing transformation for the output predictions.
        key_val_metric: The key validation metric.
        additional_metrics: Additional metrics for evaluation.

    Returns:
        SupervisedEvaluator: An instance of the supervised evaluator for validation.

    TODO Franzi:
        # max_interactions = 1 (all extreme points)
    """

    init(args)

    evaluator = SupervisedEvaluatorEp(
        device=device,
        val_data_loader=val_loader,
        networks=networks,
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=key_val_metric,
        additional_metrics=additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )
    return evaluator


def get_supervised_evaluator_dynunet(
    args,
    networks,
    inferer,
    device,
    val_loader,
    loss_function,
    post_transform,
    key_val_metric,
    additional_metrics,
) -> SupervisedEvaluatorDynUnet:
    """
    Retrieves a supervised evaluator for validation in a MONAI training workflow.

    Args:
        args: Command-line arguments and configuration settings.
        network: The model to be evaluated.
        inferer: The inference strategy or inferer.
        device: The computing device (e.g., "cuda" or "cpu") on which to run the evaluation.
        val_loader: The data loader for the validation dataset.
        loss_function: The loss function for evaluation.
        click_transforms: Data transforms for interactions (e.g., clicks).
        post_transform: Post-processing transformation for the output predictions.
        key_val_metric: The key validation metric.
        additional_metrics: Additional metrics for evaluation.

    Returns:
        SupervisedEvaluator: An instance of the supervised evaluator for validation.

    TODO Franzi:
        # max_interactions = 1 (all extreme points)
    """

    init(args)

    evaluator = SupervisedEvaluatorDynUnet(
        device=device,
        val_data_loader=val_loader,
        networks=networks,
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=key_val_metric,
        additional_metrics=additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )
    return evaluator


def get_supervised_evaluator_dynunet_da(
    args,
    networks,
    inferer,
    device,
    val_loader,
    loss_function,
    post_transform,
    key_val_metric,
    additional_metrics,
) -> SupervisedEvaluatorDynUnetDa:
    """
    Retrieves a supervised evaluator for validation in a MONAI training workflow.

    Args:
        args: Command-line arguments and configuration settings.
        network: The model to be evaluated.
        inferer: The inference strategy or inferer.
        device: The computing device (e.g., "cuda" or "cpu") on which to run the evaluation.
        val_loader: The data loader for the validation dataset.
        loss_function: The loss function for evaluation.
        click_transforms: Data transforms for interactions (e.g., clicks).
        post_transform: Post-processing transformation for the output predictions.
        key_val_metric: The key validation metric.
        additional_metrics: Additional metrics for evaluation.

    Returns:
        SupervisedEvaluator: An instance of the supervised evaluator for validation.

    TODO Franzi:
        # max_interactions = 1 (all extreme points)
    """

    init(args)

    evaluator = SupervisedEvaluatorDynUnetDa(
        device=device,
        val_data_loader=val_loader,
        networks=networks,
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=key_val_metric,
        additional_metrics=additional_metrics,
        val_handlers=get_val_handlers(
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )
    return evaluator



def get_trainer(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
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
    pre_transforms_train_source = Compose(get_pre_transforms_val_as_list(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
    pre_transforms_train_target = Compose(get_pre_transforms_val_as_list(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    val_loader = get_val_loader(args, pre_transforms_val_source=pre_transforms_train_source, pre_transforms_val_target=pre_transforms_train_target)

    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.network, args.labels, args.no_discriminator, args.no_extreme_points)
    networks[0] = networks[0].to(sw_device)
    networks[1] = networks[1].to(sw_device)
    if(not args.no_discriminator):
        networks[2].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = []
    loss_functions.append(get_loss_function(loss_args=args.loss_ugda, loss_kwargs=loss_kwargs))
    loss_functions.append(get_loss_function(loss_args=args.loss_dis, loss_kwargs=loss_kwargs))
    loss_functions.append(get_loss_function(loss_args=args.loss_dynunet, loss_kwargs=loss_kwargs))
    
    if (args.no_discriminator):
        optimizer = get_optimizer(args.optimizer, args.learning_rate, networks)
        lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)
    else:
        optimizer = get_optimizer(args.optimizer, args.learning_rate_dis, networks)
        lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)

    val_key_metric = get_key_metric(str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator = get_supervised_evaluator(
        args,
        networks=networks,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader,
        loss_function=loss_functions,
        post_transform=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
    )

    pre_transforms_train_source = Compose(get_pre_transforms_train_as_list(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
    pre_transforms_train_target = Compose(get_pre_transforms_train_as_list(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)

    train_key_metric = get_key_metric(str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers(
        lr_scheduler,
        evaluator,
        args.val_freq,
        args.eval_only,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )

    trainer = SupervisedTrainer(
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
        no_discriminator=args.no_discriminator,
        no_extreme_points=args.no_extreme_points,
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

    if ensemble_mode:
        save_dict = {
            "net_ep": networks[0],
            "net_seg": networks[1],
            "net_dis": networks[2],
        }

    if not ensemble_mode:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_interval=args.save_interval,
            save_final=True,
            final_filename="checkpoint.pt",
            save_key_metric=True,
            n_saved=2,
            file_prefix="train",
        ).attach(trainer)
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="pretrained_deepedit_" + args.network + "-final.pt",
        ).attach(evaluator)
    else:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            file_prefix=file_prefix,
        ).attach(evaluator)
    #print(checkpoint['opt']['param_groups'])


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
            handler(evaluator)

        # if args.resume_override_scheduler:
        #     # Restore params
        #     save_dict["opt"] = saved_opt
        #     save_dict["lr"] = saved_lr

        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")

    return trainer, evaluator, train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics




def get_trainer_dynunet_da(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainerDynUnetDa | None, SupervisedEvaluatorDynUnetDa | None, List]:
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
    pre_transforms_val_source = Compose(get_pre_transforms_val_as_list(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
    pre_transforms_val_target = Compose(get_pre_transforms_val_as_list(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_source = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_target = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms_dynunet(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.network, args.labels, no_discriminator=True, no_extreme_points=True)
    networks[1] = networks[1].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = get_loss_function(loss_args=args.loss_dynunet, loss_kwargs=loss_kwargs)
    
    optimizer = get_optimizer(args.optimizer, args.learning_rate, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)

    val_key_metric = get_key_metric_seg(str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator_source = get_supervised_evaluator_dynunet_da(
        args,
        networks=networks,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader_source,
        loss_function=loss_functions,
        post_transform=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
    )

    evaluator_target = get_supervised_evaluator_dynunet_da(
        args,
        networks=networks,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader_target,
        loss_function=loss_functions,
        post_transform=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
    )

    pre_transforms_train_source = Compose(get_pre_transforms_train_as_list(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
    pre_transforms_train_target = Compose(get_pre_transforms_train_as_list(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader_separate(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric_seg(str_to_prepend="train_")
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

    trainer = SupervisedTrainerDynUnetDa(
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
        no_discriminator=args.no_discriminator,
        no_extreme_points=args.no_extreme_points,
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

    if ensemble_mode:
        save_dict = {
            "net_ep": networks[0],
            "net_seg": networks[1],
            "net_dis": networks[2],
        }

    if not ensemble_mode:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_interval=args.save_interval,
            save_final=True,
            final_filename="checkpoint.pt",
            save_key_metric=True,
            n_saved=2,
            file_prefix="train",
        ).attach(trainer)
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="pretrained_deepedit_source" + args.network + "-final.pt",
        ).attach(evaluator_source)
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="pretrained_deepedit_target" + args.network + "-final.pt",
        ).attach(evaluator_target)
    else:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            file_prefix=file_prefix,
        ).attach(evaluator_source)
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            file_prefix=file_prefix,
        ).attach(evaluator_target)
    #print(checkpoint['opt']['param_groups'])


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

        # if args.resume_override_scheduler:
        #     # Restore params
        #     save_dict["opt"] = saved_opt
        #     save_dict["lr"] = saved_lr

        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")

    return trainer, [evaluator_source, evaluator_target], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics


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
    pre_transforms_train_source = Compose(get_pre_transforms_val_as_list(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
    pre_transforms_train_target = Compose(get_pre_transforms_val_as_list(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader = get_val_loader(args, pre_transforms_val_source=pre_transforms_train_source, pre_transforms_val_target=pre_transforms_train_target)

    post_transform = get_post_transforms_ep(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(network_str=args.network, labels=args.labels, no_discriminator=True, no_extreme_points=False)
    networks[0] = networks[0].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions =  get_loss_function(loss_args=args.loss_mse, loss_kwargs=loss_kwargs)
    
    optimizer = get_optimizer(args.optimizer, args.learning_rate, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)


    val_key_metric = get_key_metric_ep(str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator = get_supervised_evaluator_ep(
        args,
        networks=networks,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader,
        loss_function=loss_functions,
        post_transform=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
    )

    pre_transforms_train_source = Compose(get_pre_transforms_train_as_list(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
    pre_transforms_train_target = Compose(get_pre_transforms_train_as_list(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader(args, pre_transforms_train_source=pre_transforms_train_source, pre_transforms_train_target=pre_transforms_train_target)


    train_key_metric = get_key_metric_ep(str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )

    train_handlers = get_train_handlers(
        lr_scheduler,
        evaluator,
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
        train_handlers=train_handlers,
        no_discriminator=args.no_discriminator,
        no_extreme_points=args.no_extreme_points,
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

    if ensemble_mode:
        save_dict = {
            "net_ep": networks[0],
            "net_seg": networks[1],
            "net_dis": networks[2],
        }

    if not ensemble_mode:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_interval=args.save_interval,
            save_final=True,
            final_filename="checkpoint.pt",
            save_key_metric=True,
            n_saved=2,
            file_prefix="train",
        ).attach(trainer)
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="pretrained_deepedit_" + args.network + "-final.pt",
        ).attach(evaluator)
    else:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            file_prefix=file_prefix,
        ).attach(evaluator)
    #print(checkpoint['opt']['param_groups'])


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
            handler(evaluator)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr

    return trainer, evaluator, train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics



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
    pre_transforms_val_source = Compose(get_pre_transforms_val_as_list(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
    pre_transforms_val_target = Compose(get_pre_transforms_val_as_list(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))

    val_loader_1 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_source, dataset='source')
    val_loader_2 = get_val_loader_separate(args, pre_transforms_val=pre_transforms_val_target, dataset='target')

    post_transform = get_post_transforms_dynunet(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)

    networks = get_network(args.network, args.labels, no_discriminator=True, no_extreme_points=True)
    networks[1] = networks[1].to(sw_device)
    train_inferer, eval_inferer = get_inferers()

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_functions = get_loss_function(loss_args=args.loss_dynunet, loss_kwargs=loss_kwargs)
    

    optimizer = get_optimizer(args.optimizer, args.learning_rate, networks)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)


    val_key_metric = get_key_metric_seg(str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )
    evaluator_1 = get_supervised_evaluator_dynunet(
        args,
        networks=networks,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader_1,
        loss_function=loss_functions,
        post_transform=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
    )
    evaluator_2 = get_supervised_evaluator_dynunet(
        args,
        networks=networks,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader_2,
        loss_function=loss_functions,
        post_transform=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
    )

    pre_transforms_val_source = Compose(get_pre_transforms_train_as_list(args.labels, device, args, input_keys=('image_source', 'label'), image='image_source', label='label'))
    pre_transforms_val_target = Compose(get_pre_transforms_train_as_list(args.labels, device, args, input_keys=('image_target', 'label'), image='image_target', label='label'))
    train_loader = get_train_loader(args, pre_transforms_train_source=pre_transforms_val_source, pre_transforms_train_target=pre_transforms_val_target)


    train_key_metric = get_key_metric_seg(str_to_prepend="train_")
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
        no_discriminator=args.no_discriminator,
        no_extreme_points=args.no_extreme_points,
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

    if ensemble_mode:
        save_dict = {
            "net_ep": networks[0],
            "net_seg": networks[1],
            "net_dis": networks[2],
        }

    if not ensemble_mode:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_interval=args.save_interval,
            save_final=True,
            final_filename="checkpoint.pt",
            save_key_metric=True,
            n_saved=2,
            file_prefix="train",
        ).attach(trainer)
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename=f"pretrained_deepedit_{args.source_dataset}" + args.network + "-final.pt",
        ).attach(evaluator_1)
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename=f"pretrained_deepedit_{args.target_dataset}" + args.network + "-final.pt",
        ).attach(evaluator_2)

    #print(checkpoint['opt']['param_groups'])


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

        # if args.resume_override_scheduler:
        #     # Restore params
        #     save_dict["opt"] = saved_opt
        #     save_dict["lr"] = saved_lr

    return trainer, [evaluator_1, evaluator_2], train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics



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
