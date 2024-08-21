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
from torch.utils.data import DataLoader

from monai.config import IgniteInfo, KeysCollection
from monai.engines.utils import IterationEvents, default_metric_cmp_fn
from monai.engines.workflow import Workflow
from monai.inferers import Inferer, SimpleInferer
from monai.networks.utils import eval_mode, train_mode
from monai.transforms import Transform
from monai.utils import ForwardMode, ensure_tuple, min_version, optional_import
from monai.utils.enums import EngineStatsKeys as ESKeys
from monai.utils.module import look_up_option
from monai.metrics import DiceMetric, DiceHelper
from monai.metrics import MSEMetric
from monai.data import decollate_batch

from sw_fastedit.utils.prepare_batch import default_prepare_batch
from sw_fastedit.utils.enums import CommonKeys as Keys

logger = logging.getLogger("sw_fastedit")


if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

__all__ = ["Evaluator", "SupervisedEvaluator"]


class Evaluator(Workflow):
    """
    Base class for all kinds of evaluators, inherits from Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable or torch.DataLoader.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: torch.device | str,
        val_data_loader: Iterable | DataLoader,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=1,
            data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=val_handlers,
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )
        mode = look_up_option(mode, ForwardMode)
        if mode == ForwardMode.EVAL:
            self.mode = eval_mode
        elif mode == ForwardMode.TRAIN:
            self.mode = train_mode
        else:
            raise ValueError(f"unsupported mode: {mode}, should be 'eval' or 'train'.")

    def run(self, global_epoch: int = 1) -> None:  # type: ignore[override]
        """
        Execute validation/evaluation based on Ignite Engine.

        Args:
            global_epoch: the overall epoch if during a training. evaluator engine can get it from trainer.

        """
        # init env value for current validation process
        self.state.max_epochs = max(global_epoch, 1)  # at least one epoch of validation
        self.state.epoch = global_epoch - 1
        self.state.iteration = 0
        super().run()

    def get_stats(self, *vars):
        """
        Get the statistics information of the validation process.
        Default to return the `rank`, `best_validation_epoch` and `best_validation_metric`.

        Args:
            vars: except for the default stats, other variables name in the `self.state` to return,
                will use the variable name as the key and the state content as the value.
                if the variable doesn't exist, default value is `None`.

        """
        stats = {
            ESKeys.RANK: self.state.rank,
            ESKeys.BEST_VALIDATION_EPOCH: self.state.best_metric_epoch,
            ESKeys.BEST_VALIDATION_METRIC: self.state.best_metric,
        }
        for k in vars:
            stats[k] = getattr(self.state, k, None)
        return stats


class SupervisedEvaluator_backup(Evaluator):
    """
    Standard supervised evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        network: network to evaluate in the evaluator, should be regular PyTorch `torch.nn.Module`.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
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
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        networks: Sequence[torch.nn.Module],
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.network = networks
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.dice_helper = DiceHelper(include_background=False, sigmoid=False, softmax=True)
        self.mse_metric = MSEMetric()
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine: SupervisedEvaluator, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: `SupervisedEvaluator` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, target_ep = batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print("error with batch, check prepare_batch")

        print('')    
        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("labels.shape is {}".format(target_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())

        if (keys[0] == 'image_target'):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")
      
        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        #print('check labels', batchdata["label"].shape)
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")
        # put iteration outputs into engine.state
        if keys[0]== 'image_target':    
            engine.state.output = {Keys.IMAGE_TARGET: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: target_ep}
        else:
            engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: target_ep}
        inputs_seg_ep = torch.cat((inputs, target_ep), dim=1) #image and generated ep as input
        
        # execute forward computation
        with engine.mode(engine.network):
            if engine.amp:
                with torch.cuda.amp.autocast(**engine.amp_kwargs):
                    engine.state.output[Keys.PRED_EP] = engine.inferer(inputs, engine.network[0], *args, **kwargs)
                    pred_ep_processed = torch.where(engine.state.output[Keys.PRED_EP] > 0.1, engine.state.output[Keys.PRED_EP], torch.tensor(0.0, device=engine.state.device)) 
                    engine.state.output['pred_ep_processed'] = pred_ep_processed
                    inputs_seg_ep_processed = torch.cat((inputs, pred_ep_processed), dim=1) #image and generated ep as input

                    engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs_seg_ep, engine.network[1], *args, **kwargs)
            else:
                engine.state.output[Keys.PRED_EP] = engine.inferer(inputs, engine.network[0], *args, **kwargs)
                #pred_ep_processed = torch.where(engine.state.output[Keys.PRED_EP] > 0.1, engine.state.output[Keys.PRED_EP], torch.tensor(0.0, device=engine.state.device)) 
                #engine.state.output['pred_ep_processed'] = pred_ep_processed
                inputs_seg_ep_processed = torch.cat((inputs, pred_ep_processed), dim=1) #image and generated ep as input
                engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs_seg_ep, engine.network[1], *args, **kwargs)

        #evalutation extreme points
        mse_metric = self.mse_metric(y_pred=engine.state.output[Keys.PRED_EP], y=target_ep).item()

        #evaluation segmentation
        #pred = torch.argmax(engine.state.output[Keys.PRED_SEG], dim=1, keepdims=True)
        print('shape pred', engine.state.output[Keys.PRED_SEG].shape)
        print('unique pred', torch.unique(engine.state.output[Keys.PRED_SEG]))
        pred_decollate = decollate_batch(engine.state.output[Keys.PRED_SEG])
        target = decollate_batch(target_seg)
        #without argmax/softmax
        dice_metric = self.dice_metric(y_pred=pred_decollate, y=target)
        #with softmax
        dice_helper =   self.dice_helper(y_pred=engine.state.output[Keys.PRED_SEG], y=target_seg)
        logger.info(f'Dice Metric: {dice_metric} MSE Metric: {mse_metric}')
        logger.info(f'Dice Helper: {dice_helper} MSE Metric: {mse_metric}')


        #engine.state.output[Keys.LABEL] = targets
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        
        return engine.state.output


class SupervisedEvaluatorEp(Evaluator):
    """
    Standard supervised evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        network: network to evaluate in the evaluator, should be regular PyTorch `torch.nn.Module`.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
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
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        networks: Sequence[torch.nn.Module],
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.network = networks
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.dice_helper = DiceHelper(include_background=False, sigmoid=False, softmax=True)
        self.mse_metric = MSEMetric()
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine: SupervisedEvaluator, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: `SupervisedEvaluator` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, target_ep = batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print("error with batch, check prepare_batch")

        print('')    
        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("labels.shape is {}".format(target_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())

        if (keys[0] == 'image_target'):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")
      
        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        #print('check labels', batchdata["label"].shape)
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")
        # put iteration outputs into engine.state
        if keys[0]== 'image_target':    
            engine.state.output = {Keys.IMAGE_TARGET: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: target_ep}
        else:
            engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: target_ep}
        inputs_seg_ep = torch.cat((inputs, target_ep), dim=1) #image and generated ep as input
        
        # execute forward computation
        with engine.mode(engine.network):
            if engine.amp:
                with torch.cuda.amp.autocast(**engine.amp_kwargs):
                    engine.state.output[Keys.PRED_EP] = engine.inferer(inputs, engine.network[0], *args, **kwargs)
            else:
                engine.state.output[Keys.PRED_EP] = engine.inferer(inputs, engine.network[0], *args, **kwargs)
        
        pred_ep_processed = torch.where(engine.state.output[Keys.PRED_EP] > 0.1, engine.state.output[Keys.PRED_EP], torch.tensor(0.0, device=engine.state.device)) 
        engine.state.output['pred_ep_processed'] = pred_ep_processed

        #evalutation extreme points
        mse_metric = self.mse_metric(y_pred=engine.state.output[Keys.PRED_EP], y=target_ep).item()

        logger.info(f'MSE Metric: {mse_metric}')

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        
        return engine.state.output
    

class SupervisedEvaluatorDynUnet(Evaluator):
    """
    Standard supervised evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        network: network to evaluate in the evaluator, should be regular PyTorch `torch.nn.Module`.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
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
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        networks: Sequence[torch.nn.Module],
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.network = networks
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.dice_helper = DiceHelper(include_background=False, sigmoid=False, softmax=True)
        self.mse_metric = MSEMetric()
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine: SupervisedEvaluator, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: `SupervisedEvaluator` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, target_ep = batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print("error with batch, check prepare_batch")

        print('')    
        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("labels.shape is {}".format(target_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())

        if (keys[0] == 'image_target'):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")
      
        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        #print('check labels', batchdata["label"].shape)
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")
        # put iteration outputs into engine.state
        if keys[0]== 'image_target':    
            engine.state.output = {Keys.IMAGE_TARGET: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: target_ep}
        else:
            engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: target_ep}
        
        # execute forward computation
        with engine.mode(engine.network):
            if engine.amp:
                with torch.cuda.amp.autocast(**engine.amp_kwargs):
                    engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs, engine.network[1], *args, **kwargs)
            else:
                engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs, engine.network[1], *args, **kwargs)

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)

        dice_helper = self.dice_helper(y_pred=engine.state.output[Keys.PRED_SEG], y=target_seg)
        logger.info(f'Dice Helper: {dice_helper} with softmax')
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        
        return engine.state.output


class SupervisedEvaluatorDynUnetDa(Evaluator):
    """
    Standard supervised evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        network: network to evaluate in the evaluator, should be regular PyTorch `torch.nn.Module`.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
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
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        networks: Sequence[torch.nn.Module],
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.network = networks
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.dice_helper = DiceHelper(include_background=False, sigmoid=False, softmax=True)
        self.mse_metric = MSEMetric()
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine: SupervisedEvaluatorDynUnetDa, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: `SupervisedEvaluator` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, target_ep = batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print("error with batch, check prepare_batch")

        print('')    
        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("labels.shape is {}".format(target_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())

        if (keys[0] == 'image_target'):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")
      
        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        #print('check labels', batchdata["label"].shape)
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")
        # put iteration outputs into engine.state
        if keys[0]== 'image_target':    
            engine.state.output = {Keys.IMAGE_TARGET: inputs, Keys.LABEL_SEG: target_seg}
        else:
            engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg}
        
        # execute forward computation
        with engine.mode(engine.network):
            if engine.amp:
                with torch.cuda.amp.autocast(**engine.amp_kwargs):
                    engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs, engine.network[1], *args, **kwargs)
            else:
                engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs, engine.network[1], *args, **kwargs)

        #evaluation segmentation
        dice_helper = self.dice_helper(y_pred=engine.state.output[Keys.PRED_SEG], y=target_seg)
        logger.info(f'Dice Helper: {dice_helper} with softmax')

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        
        return engine.state.output


class SupervisedEvaluator(Evaluator):
    """
    Standard supervised evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        network: network to evaluate in the evaluator, should be regular PyTorch `torch.nn.Module`.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
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
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        networks: Sequence[torch.nn.Module],
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.network = networks
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.dice_helper = DiceHelper(include_background=False, sigmoid=False, softmax=True)
        self.mse_metric = MSEMetric()
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine: SupervisedEvaluator, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: `SupervisedEvaluator` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 3:
            inputs, target_seg, target_ep = batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            print("error with batch, check prepare_batch")

        print('')    
        logger.info("inputs.shape is {}".format(inputs.shape))
        logger.info("labels.shape is {}".format(target_seg.shape))
        logger.info("labels.shape is {}".format(target_ep.shape))
        # Make sure the signal is empty in the first iteration assertion holds
        assert torch.sum(inputs[:, 1:, ...]) == 0
        keys = list(batchdata.keys())

        if (keys[0] == 'image_target'):
            logger.info(f"image file name: {batchdata['image_target_meta_dict']['filename_or_obj']}")
        else:
            logger.info(f"image file name: {batchdata['image_source_meta_dict']['filename_or_obj']}")
      
        logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
        #print('check labels', batchdata["label"].shape)
        for i in range(len(batchdata["label"])):
            if torch.sum(batchdata["label"][i, 0]) < 0.1:
                logger.warning("No valid labels for this sample (probably due to crop)")
        # put iteration outputs into engine.state
        if keys[0]== 'image_target':    
            engine.state.output = {Keys.IMAGE_TARGET: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: target_ep}
        else:
            engine.state.output = {Keys.IMAGE_SOURCE: inputs, Keys.LABEL_SEG: target_seg, Keys.LABEL_EP: target_ep}
        inputs_seg_ep = torch.cat((inputs, target_ep), dim=1) #image and generated ep as input
        
        # execute forward computation
        with engine.mode(engine.network):
            if engine.amp:
                with torch.cuda.amp.autocast(**engine.amp_kwargs):
                    engine.state.output[Keys.PRED_EP] = engine.inferer(inputs, engine.network[0], *args, **kwargs)
                    pred_ep_processed = torch.where(engine.state.output[Keys.PRED_EP] > 0.1, engine.state.output[Keys.PRED_EP], torch.tensor(0.0, device=engine.state.device)) 
                    engine.state.output['pred_ep_processed'] = pred_ep_processed
                    inputs_seg_ep_processed = torch.cat((inputs, pred_ep_processed), dim=1) #image and generated ep as input

                    engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs_seg_ep, engine.network[1], *args, **kwargs)
            else:
                engine.state.output[Keys.PRED_EP] = engine.inferer(inputs, engine.network[0], *args, **kwargs)
                #pred_ep_processed = torch.where(engine.state.output[Keys.PRED_EP] > 0.1, engine.state.output[Keys.PRED_EP], torch.tensor(0.0, device=engine.state.device)) 
                #engine.state.output['pred_ep_processed'] = pred_ep_processed
                inputs_seg_ep_processed = torch.cat((inputs, pred_ep_processed), dim=1) #image and generated ep as input
                engine.state.output[Keys.PRED_SEG] = engine.inferer(inputs_seg_ep, engine.network[1], *args, **kwargs)

        #evalutation extreme points
        mse_metric = self.mse_metric(y_pred=engine.state.output[Keys.PRED_EP], y=target_ep).item()

        #evaluation segmentation
        #pred = torch.argmax(engine.state.output[Keys.PRED_SEG], dim=1, keepdims=True)
        print('shape pred', engine.state.output[Keys.PRED_SEG].shape)
        print('unique pred', torch.unique(engine.state.output[Keys.PRED_SEG]))
        pred_decollate = decollate_batch(engine.state.output[Keys.PRED_SEG])
        target = decollate_batch(target_seg)
        #without argmax/softmax
        dice_metric = self.dice_metric(y_pred=pred_decollate, y=target)
        #with softmax
        dice_helper =   self.dice_helper(y_pred=engine.state.output[Keys.PRED_SEG], y=target_seg)
        logger.info(f'Dice Metric: {dice_metric} MSE Metric: {mse_metric}')
        logger.info(f'Dice Helper: {dice_helper} MSE Metric: {mse_metric}')


        #engine.state.output[Keys.LABEL] = targets
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        
        return engine.state.output



