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

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import Compose, SaveImaged, SplitDimd
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.config import IgniteInfo
from monai.engines.utils import IterationEvents, default_make_latent, default_metric_cmp_fn
from monai.engines.workflow import Workflow
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import min_version, optional_import
from sw_fastedit.utils.enums import GanKeys
from sw_fastedit.utils.enums import CommonKeys as Keys
from sw_fastedit.utils.enums import EngineStatsKeys as ESKeys
from sw_fastedit.utils.prepare_batch import default_prepare_batch

logger = logging.getLogger("sw_fastedit")

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

__all__ = ["Trainer", "SupervisedTrainer", "GanTrainer"]


class Trainer(Workflow):
    """
    Base class for all kinds of trainers, inherits from Workflow.

    """

    def run(self) -> None:  # type: ignore[override]
        """
        Execute training based on Ignite Engine.
        If call this function multiple times, it will continuously run from the previous state.

        """
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        super().run()

    def get_stats(self, *vars):
        """
        Get the statistics information of the training process.
        Default to return the `rank`, `current_epoch`, `current_iteration`, `total_epochs`, `total_iterations`.

        Args:
            vars: except for the default stats, other variables name in the `self.state` to return,
                will use the variable name as the key and the state content as the value.
                if the variable doesn't exist, default value is `None`.

        """
        stats = {
            ESKeys.RANK: self.state.rank,
            ESKeys.CURRENT_EPOCH: self.state.epoch,
            ESKeys.CURRENT_ITERATION: self.state.iteration,
            ESKeys.TOTAL_EPOCHS: self.state.max_epochs,
            ESKeys.TOTAL_ITERATIONS: self.state.epoch_length,
        }
        for k in vars:
            stats[k] = getattr(self.state, k, None)
        return stats


class SupervisedTrainer(Trainer):
    """
    Standard supervised training method with image and label, inherits from ``Trainer`` and ``Workflow``.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for trainer to run.
        train_data_loader: Ignite engine use data_loader to run, must be Iterable or torch.DataLoader.
        network: network to train in the trainer, should be regular PyTorch `torch.nn.Module`.
        optimizer: the optimizer associated to the network, should be regular PyTorch optimizer from `torch.optim`
            or its subclass.
        loss_function: the loss function associated to the optimizer, should be regular PyTorch loss,
            which inherit from `torch.nn.modules.loss`.
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision training, default is False.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        optim_set_to_none: when calling `optimizer.zero_grad()`, instead of setting to zero, set the grads to None.
            more details: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: str | torch.device,
        max_epochs: int,
        train_data_loader: Iterable | DataLoader,
        networks: Sequence | torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Sequence | Callable,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_train_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Sequence | None = None,
        amp: bool = False,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
        no_discriminator: bool = False,
        no_extreme_points: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers, 
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.networks = networks
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = SimpleInferer() if inferer is None else inferer
        self.optim_set_to_none = optim_set_to_none
        self.loss_discriminator = {}
        self.no_discriminator = no_discriminator
        self.no_extreme_points = no_extreme_points




    def _iteration(self, engine: SupervisedTrainer, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: `SupervisedTrainer` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """

        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, targets_ep= batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print('Problem with batch, check prepare_batch')
        

        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("extreme points.shape is {}".format(targets_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())
        if(keys[0]== self.target_dataset):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")

        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")


        def _compute_discriminator_loss():
            engine.networks[2].train()
            engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
            #engine.networks[2].zero_grad(set_to_none=engine.optim_set_to_none)
            if(keys[0]== self.target_dataset):
                engine.state.output[GanKeys.LATENTS] = engine.inferer(engine.state.output[Keys.PRED_SEG_EP], engine.networks[2], *args, **kwargs)
                engine.fire_event(IterationEvents.FORWARD_COMPLETED)
                loss = engine.loss_function[1](engine.state.output[GanKeys.LATENTS], torch.zeros_like(engine.state.output[GanKeys.LATENTS]))
                engine.fire_event(IterationEvents.LOSS_COMPLETED)
                self.loss_discriminator['loss_target'] = loss
                print('image target loss', loss.item())
               
            else: 
                engine.state.output[GanKeys.LATENTS] = engine.inferer(engine.state.output[Keys.PRED_SEG_EP], engine.networks[2], *args, **kwargs)
                engine.fire_event(IterationEvents.FORWARD_COMPLETED)
                loss = engine.loss_function[1](engine.state.output[GanKeys.LATENTS], torch.ones_like(engine.state.output[GanKeys.LATENTS]))
                engine.fire_event(IterationEvents.LOSS_COMPLETED)
                self.loss_discriminator['loss_source'] = loss
                print('image source loss', loss.item())
        
            if 'loss_target' in self.loss_discriminator and 'loss_source' in self.loss_discriminator:


                for param in engine.networks[0].parameters():
                    param.requires_grad = False
                for param in engine.networks[1].parameters():
                    param.requires_grad = False
                for param in engine.networks[2].parameters():
                    param.requires_grad = True
                engine.state.output[GanKeys.DLOSS] = self.loss_discriminator['loss_target'] + self.loss_discriminator['loss_source']
                print('loss discriminator', engine.state.output[GanKeys.DLOSS].item())

                engine.scaler.scale(engine.state.output[GanKeys.DLOSS]).backward(retain_graph=True)
                engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
                del self.loss_discriminator['loss_target'], self.loss_discriminator['loss_source']

            return loss
        

        def _compute_adversarial_loss():

            if keys[0] == self.target_dataset:
                engine.networks[1].train()
                for param in engine.networks[0].parameters():
                    param.requires_grad = False
                for param in engine.networks[1].parameters():
                    param.requires_grad = True
                for param in engine.networks[2].parameters():
                    param.requires_grad = False
                engine.state.output[GanKeys.ADVLOSS] = engine.loss_function[1](engine.state.output[GanKeys.LATENTS], 
                                                                               target_adv = torch.ones_like(engine.state.output[GanKeys.LATENTS]))
                engine.fire_event(IterationEvents.LOSS_COMPLETED)

                engine.scaler.scale(engine.state.output[GanKeys.ADVLOSS]).backward(retain_graph=True)
                engine.fire_event(IterationEvents.BACKWARD_COMPLETED)

        def _forward_seg():
            if keys[0]== 'image_source':    
                engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg}
                engine.networks[1].train()
                engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
                engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs, engine.networks[1], *args, **kwargs)
                engine.fire_event(IterationEvents.FORWARD_COMPLETED)

                engine.state.output[Keys.LOSS] = engine.loss_function[2](input = engine.state.output[Keys.PRED_SEG], target = target_seg).mean()
                engine.fire_event(IterationEvents.LOSS_COMPLETED)

                engine.scaler.scale(engine.state.output[Keys.LOSS]).backward()
                engine.fire_event(IterationEvents.BACKWARD_COMPLETED)

            
        def _forward_seg_ep():
            engine.networks[0].train()
            engine.networks[1].train()
            
            engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

            if keys[0]== self.target_dataset:    
                engine.state.output = {Keys.IMAGE_TARGET: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: targets_ep}
            else:
                engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: targets_ep}
            
            #first network prediction ep
            engine.state.output[Keys.PRED_EP] = engine.inferer(inputs, engine.networks[0], *args, **kwargs)

            #concatenate output from first network (ep_pred) and image as input for second network
            inputs_img_ep = torch.cat((inputs, engine.state.output[Keys.PRED_EP]), dim=1) 
                   
            #second network prediction of segementation
            engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs_img_ep, engine.networks[1], *args, **kwargs)
           
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)

            #error in library if removed
            #engine.state.output[Keys.LABEL] = targets 
           
            engine.state.output[Keys.PRED_SEG_EP] = torch.cat((engine.state.output[Keys.PRED_SEG], engine.state.output[Keys.PRED_EP]), dim=1) 

            #postprocessing, try with threshold between 0.05 and 0.25 (maybe 0.3) only for discriminator
            pred_ep_processed = torch.where(engine.state.output[Keys.PRED_EP] > 0.1, engine.state.output[Keys.PRED_EP], torch.tensor(0.0, device=engine.state.device)) 
            engine.state.output['pred_ep_processed'] = pred_ep_processed

        def _compute_loss_seg_ep():
            for param in engine.networks[0].parameters():
                param.requires_grad = True

            # print("Before setting requires_grad:")
            # for i, param in enumerate(engine.networks[1].parameters()):
            #     print(f"Param {i}: requires_grad = {param.requires_grad}")

            if keys[0] == self.target_dataset:
                for param in engine.networks[1].parameters():
                    param.requires_grad = False
                engine.state.output[Keys.LOSS] = engine.loss_function[0](input = engine.state.output[Keys.PRED_SEG_EP],
                                                                         target_seg = target_seg, target_ep = targets_ep).mean()
            else:
                for param in engine.networks[1].parameters():
                    param.requires_grad = False
                engine.state.output[Keys.LOSS] = engine.loss_function[0](input = engine.state.output[Keys.PRED_SEG_EP],
                                                                         target_seg = target_seg, target_ep = targets_ep).mean()
            

         
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

            # print("After setting requires_grad:")
            # for i, param in enumerate(engine.networks[1].parameters()):
            #     print(f"Param {i}: requires_grad = {param.requires_grad}")

            engine.scaler.scale(engine.state.output[Keys.LOSS]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)

            

        if engine.amp and engine.scaler is not None:
            with torch.cuda.amp.autocast(**engine.amp_kwargs):
                if self.no_extreme_points:
                    _forward_seg()
                else:
                    _forward_seg_ep()
                    if self.no_discriminator:
                        _compute_loss_seg_ep()
                    else:
                        _compute_discriminator_loss()
                        _compute_adversarial_loss()
                    
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
           
        else:
            if self.no_extreme_points:
                _forward_seg()
            else:
                _forward_seg_ep()
                if self.no_discriminator:          
                   _compute_loss_seg_ep()
                else:
                    _compute_discriminator_loss()
                    _compute_adversarial_loss()

            engine.optimizer.step()
        
        engine.fire_event(IterationEvents.MODEL_COMPLETED)  



        return engine.state.output


class SupervisedTrainerEp(Trainer):
    """
    Standard supervised training method with image and label, inherits from ``Trainer`` and ``Workflow``.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for trainer to run.
        train_data_loader: Ignite engine use data_loader to run, must be Iterable or torch.DataLoader.
        network: network to train in the trainer, should be regular PyTorch `torch.nn.Module`.
        optimizer: the optimizer associated to the network, should be regular PyTorch optimizer from `torch.optim`
            or its subclass.
        loss_function: the loss function associated to the optimizer, should be regular PyTorch loss,
            which inherit from `torch.nn.modules.loss`.
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision training, default is False.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        optim_set_to_none: when calling `optimizer.zero_grad()`, instead of setting to zero, set the grads to None.
            more details: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: str | torch.device,
        max_epochs: int,
        train_data_loader: Iterable | DataLoader,
        networks: Sequence | torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Sequence | Callable,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_train_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Sequence | None = None,
        amp: bool = False,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
        no_discriminator: bool = False,
        no_extreme_points: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers, 
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.networks = networks
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = SimpleInferer() if inferer is None else inferer
        self.optim_set_to_none = optim_set_to_none
        self.loss_discriminator = {}
        self.no_discriminator = no_discriminator
        self.no_extreme_points = no_extreme_points
        self.source_dataset = "image_source"
        self.target_dataset = "image_target"




    def _iteration(self, engine: SupervisedTrainerEp, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: `SupervisedTrainer` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """

        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, targets_ep= batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print('Problem with batch, check prepare_batch')
        
        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("extreme points.shape is {}".format(targets_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())
        if(keys[0]== self.target_dataset):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")

        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")
            
        def _forward_ep():
            engine.networks[0].train()
            
            engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

            if keys[0]== self.target_dataset:    
                engine.state.output = {Keys.IMAGE_TARGET: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: targets_ep}
            else:
                engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: targets_ep}
            
            #first network prediction ep
            engine.state.output[Keys.PRED_EP] = engine.inferer(inputs, engine.networks[0], *args, **kwargs)
                       
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)

            #postprocessing, try with threshold between 0.05 and 0.25 (maybe 0.3) only for discriminator
            pred_ep_processed = torch.where(engine.state.output[Keys.PRED_EP] > 0.1, engine.state.output[Keys.PRED_EP], torch.tensor(0.0, device=engine.state.device)) 
            engine.state.output['pred_ep_processed'] = pred_ep_processed

        def _compute_loss_ep():
            for param in engine.networks[0].parameters():
                param.requires_grad = True
            for param in engine.networks[1].parameters():
                param.requires_grad = False

            engine.state.output[Keys.LOSS] = engine.loss_function(input=engine.state.output[Keys.PRED_EP], target=targets_ep).mean()
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

            # print("After setting requires_grad:")
            # for i, param in enumerate(engine.networks[1].parameters()):
            #     print(f"Param {i}: requires_grad = {param.requires_grad}")

            engine.scaler.scale(engine.state.output[Keys.LOSS]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)

        if engine.amp and engine.scaler is not None:
            with torch.cuda.amp.autocast(**engine.amp_kwargs):
                _forward_ep()
                _compute_loss_ep()

            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
           
        else:
            _forward_ep()
            _compute_loss_ep()
            engine.optimizer.step()
        
        engine.fire_event(IterationEvents.MODEL_COMPLETED)  

        return engine.state.output
    



class SupervisedTrainerDynUnet(Trainer):
    """
    Standard supervised training method with image and label, inherits from ``Trainer`` and ``Workflow``.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for trainer to run.
        train_data_loader: Ignite engine use data_loader to run, must be Iterable or torch.DataLoader.
        network: network to train in the trainer, should be regular PyTorch `torch.nn.Module`.
        optimizer: the optimizer associated to the network, should be regular PyTorch optimizer from `torch.optim`
            or its subclass.
        loss_function: the loss function associated to the optimizer, should be regular PyTorch loss,
            which inherit from `torch.nn.modules.loss`.
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision training, default is False.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        optim_set_to_none: when calling `optimizer.zero_grad()`, instead of setting to zero, set the grads to None.
            more details: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: str | torch.device,
        max_epochs: int,
        train_data_loader: Iterable | DataLoader,
        networks: Sequence | torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Sequence | Callable,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_train_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Sequence | None = None,
        amp: bool = False,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
        no_discriminator: bool = False,
        no_extreme_points: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers, 
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.networks = networks
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = SimpleInferer() if inferer is None else inferer
        self.optim_set_to_none = optim_set_to_none
        self.loss_discriminator = {}
        self.no_discriminator = no_discriminator
        self.no_extreme_points = no_extreme_points
        self.source_dataset = "image_source"
        self.target_dataset = "image_target"


    def _iteration(self, engine: SupervisedTrainerDynUnet, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: `SupervisedTrainer` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """

        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, targets_ep= batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print('Problem with batch, check prepare_batch')
        
        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("extreme points.shape is {}".format(targets_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())
        if(keys[0]== self.target_dataset):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")

        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")


        def _compute_seg_loss():
            if keys[0]== self.source_dataset:    
                engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg}
            else:
                engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg}

                engine.networks[1].train()
                engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

            engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs, engine.networks[1], *args, **kwargs)
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)

            engine.state.output[Keys.LOSS] = engine.loss_function(input = engine.state.output[Keys.PRED_SEG], target = target_seg).mean()
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

            engine.scaler.scale(engine.state.output[Keys.LOSS]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)


        if engine.amp and engine.scaler is not None:
            with torch.cuda.amp.autocast(**engine.amp_kwargs):
                _compute_seg_loss()
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
           
        else:
            _compute_seg_loss()
            engine.optimizer.step()
        
        engine.fire_event(IterationEvents.MODEL_COMPLETED)  
        return engine.state.output
    

class SupervisedTrainerDynUnetDa(Trainer):
    """
    Standard supervised training method with image and label, inherits from ``Trainer`` and ``Workflow``.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for trainer to run.
        train_data_loader: Ignite engine use data_loader to run, must be Iterable or torch.DataLoader.
        network: network to train in the trainer, should be regular PyTorch `torch.nn.Module`.
        optimizer: the optimizer associated to the network, should be regular PyTorch optimizer from `torch.optim`
            or its subclass.
        loss_function: the loss function associated to the optimizer, should be regular PyTorch loss,
            which inherit from `torch.nn.modules.loss`.
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision training, default is False.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        optim_set_to_none: when calling `optimizer.zero_grad()`, instead of setting to zero, set the grads to None.
            more details: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: str | torch.device,
        max_epochs: int,
        train_data_loader: Iterable | DataLoader,
        networks: Sequence | torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Sequence | Callable,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_train_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Sequence | None = None,
        amp: bool = False,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
        no_discriminator: bool = False,
        no_extreme_points: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers, 
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.networks = networks
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = SimpleInferer() if inferer is None else inferer
        self.optim_set_to_none = optim_set_to_none
        self.loss_discriminator = {}
        self.no_discriminator = no_discriminator
        self.no_extreme_points = no_extreme_points


    def _iteration(self, engine: SupervisedTrainerDynUnetDa, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: `SupervisedTrainer` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """

        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, targets_ep= batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print('Problem with batch, check prepare_batch')
        
        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("extreme points.shape is {}".format(targets_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())
        if(keys[0]== "target_dataset"):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")

        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")


        def _compute_seg_loss():
            if keys[0]== "image_source":    

                engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg}

                engine.networks[1].train()
                engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

                engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs, engine.networks[1], *args, **kwargs)
                engine.fire_event(IterationEvents.FORWARD_COMPLETED)

                engine.state.output[Keys.LOSS] = engine.loss_function(input = engine.state.output[Keys.PRED_SEG], target = target_seg).mean()
                engine.fire_event(IterationEvents.LOSS_COMPLETED)

                engine.scaler.scale(engine.state.output[Keys.LOSS]).backward()
                engine.fire_event(IterationEvents.BACKWARD_COMPLETED)


        if engine.amp and engine.scaler is not None:
            with torch.cuda.amp.autocast(**engine.amp_kwargs):
                _compute_seg_loss()
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
            
        else:
            _compute_seg_loss()
            engine.optimizer.step()
        
        engine.fire_event(IterationEvents.MODEL_COMPLETED)  
        return engine.state.output

