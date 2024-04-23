import warnings
from typing import Callable, List, Optional, Sequence, Union

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot

from monai.losses import DiceLoss, DiceCELoss
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option
from monai.data import list_data_collate, decollate_batch
from monai.transforms import AsDiscrete

logger = logging.getLogger("sw_fastedit")



class DiceCeL2Loss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1,
        lambda_ce: float = 1,
        lambda_rec: float = 1,
        args=None
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value


        # include_background should be set to True if the dice loss stagnates because the error signal is too weak from the small tumor lesions
        self.dice = DiceLoss(to_onehot_y=to_onehot_y, softmax=softmax, include_background=include_background, batch=batch)

        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.rec_loss = nn.MSELoss()
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.lambda_rec = lambda_rec


        self.post_label = AsDiscrete(to_onehot=2)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)

    
    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """


        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        input_seg =  input[:,0:2] #get segmentation predicition
        input_ep = input[:,2:3] #get extrme point prediction

        target_seg = target[:,0:1]
        target_ep = target[:,1:2]

        if len(input_seg.shape) != len(target_seg.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")
        dice_loss = self.dice(input_seg, target_seg)
        ce_loss = self.ce(input_seg, target_seg)
       

        rec_loss = self.rec_loss(input_ep, target_ep)
    
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_dice * ce_loss + self.lambda_rec * rec_loss
        logger.info(f"Total loss: {total_loss.item()}, DiceCE: {dice_loss.item()+ce_loss.item()}, Mean squared error: {rec_loss} ")

        return total_loss    