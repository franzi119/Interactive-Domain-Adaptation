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

import argparse
import logging
import os
import pathlib
import sys
import tempfile
import time
import uuid

import torch

from sw_fastedit.utils.helper import get_actual_cuda_index_of_device, get_git_information, gpu_usage
from sw_fastedit.utils.logger import get_logger, setup_loggers


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input_dir", required=True, help="Base folder for input images and labels")
    parser.add_argument("-o", "--output_dir", required=True, help="All the logs and weights will be stored here")
    parser.add_argument(
        "-d", "--data_dir", default="None", help="Only used for debugging Niftii files, so usually not required"
    )

    parser.add_argument("--organ", type=int, default=10, choices=range(1,15), help="Organ to segment")
    # a subdirectory is created below cache_dir for every run
    parser.add_argument(
        "-c",
        "--cache_dir",
        type=str,
        default="None",
        help="Code uses a CacheDataset, so stores the transforms on the disk. This parameter is where the data gets stored.",
    )
    parser.add_argument(
        "-ta",
        "--throw_away_cache",
        default=False,
        action="store_true",
        help="Use a temporary folder which will be cleaned up after the program run.",
    )
    parser.add_argument(
        "--save_pred",
        default=False,
        action="store_true",
        help="To save the prediction in the output_dir/prediction if that is desired",
    )
    parser.add_argument(
        "--gpu_size",
        default="None",
        choices=["None", "small", "medium", "large"],
        help="Influcences some performance options of the code",
    )
    parser.add_argument(
        "--limit_gpu_memory_to",
        type=float,
        default=-1,
        help="Set it to the fraction of the GPU memory that shall be used, e.g. 0.5",
    )
    parser.add_argument(
        "-t",
        "--limit",
        type=int,
        default=0,
        help="Limit the amount of training/validation samples to a fixed number",
    )
    parser.add_argument(
        "--dataset", default="AMOS", choices=["AMOS"]
    )
    parser.add_argument(
        "--use_test_data_for_validation", default=False, action="store_true", help="Use the test data instead of the split of the training data for validation. May not work for all models but is tested for AutoPET"
    )
    parser.add_argument("--train_on_all_samples", action="store_true")
    parser.add_argument(
        "--positive_crop_rate", type=float, default=0.6, help="The rate of positive samples for RandCropByPosNegLabeld"
    )

    # Configuration
    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use.")
    parser.add_argument("--no_log", default=False, action="store_true")
    parser.add_argument("--no_data", default=False, action="store_true")
    parser.add_argument("--dont_check_output_dir", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debugpy", default=False, action="store_true")

    # Model
    parser.add_argument(
        "-n",
        "--network",
        default="dynunet",
        choices=["dynunet"],
    )
    parser.add_argument(
        "-in",
        "--inferer",
        default="SimpleInferer",
        choices=["SimpleInferer"],
    )

    # source and target dataset
    parser.add_argument("--source_dataset", default="image_ct")
    parser.add_argument("--target_dataset", default="image_mri")

    parser.add_argument("--same_normalization", default=False, action="store_true")

    parser.add_argument("--sw_cpu_output", default=False, action="store_true")

    # Training
    parser.add_argument("-a", "--amp", default=True, action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    # LOSS
    # If learning rate is set to 0.001, the DiceCELoss will produce Nans very quickly
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-7)
    parser.add_argument("-lr_dis", "--learning_rate_dis", type=float, default=1e-5)
    parser.add_argument("--eta_min_dis", type=float, default=1e-5)
    parser.add_argument("-lr_ep", "--learning_rate_ep", type=float, default=1e-4)
    parser.add_argument("--eta_min_ep", type=float, default=1e-7)
    parser.add_argument("-lr_adv", "--learning_rate_adv", type=float, default=1e-5)
    parser.add_argument("--eta_min_adv", type=float, default=1e-8)
    parser.add_argument("--lambda_adv", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "Novograd"])
    parser.add_argument("--loss_ugda", default="DiceCEL2Loss", choices=["DiceCEL2Loss"])
    parser.add_argument("--loss_pada", default="DiceCeAdvLoss", choices=["DiceCeAdvLoss"])
    parser.add_argument("--loss_dis", default="BCE", choices=["BCE"])
    parser.add_argument("--loss_dynunet", default="DiceCELoss", choices=["DiceCELoss"])
    parser.add_argument("--loss_mse", default="MSELoss", choices=["MSELoss"])
    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        choices=["MultiStepLR", "PolynomialLR", "CosineAnnealingLR"],
    )
    parser.add_argument("--loss_dont_include_background", default=True, action="store_false")
    parser.add_argument("--loss_no_squared_pred", default=False, action="store_true")

    #extreme points
    parser.add_argument("--loss_ep", default="mean_squared_error")
    parser.add_argument("--optimizer_ep", default="Adam")
    parser.add_argument("--backprop_ep_separate", default="False", action="store_true") #for dualDynunet
    parser.add_argument("-ep", "--extreme_points", default=False, action="store_true") # for pada
    parser.add_argument("-pred_ep", "--pred_ep", default=False, action="store_true") # for dextr

    parser.add_argument("--resume_from", type=str, default="None")
    # Use this parameter to change the scheduler.. (when using a pre-trained network for pada or ugda)
    parser.add_argument("--resume_override_scheduler", default=False, action="store_true")
    parser.add_argument("--scale_intensity_ranged", default=False, action="store_true")
    parser.add_argument("--additional_metrics", default=False, action="store_true")
    # Can speed up the training by cropping away some percentiles of the data
    parser.add_argument("--crop_foreground", default=False, action="store_true")

    # Logging
    parser.add_argument("-f", "--val_freq", type=int, default=1)  # Epoch Level
    parser.add_argument("--save_interval", type=int, default=10)  # Save checkpoints every x epochs

    parser.add_argument("--eval_only", default=False, action="store_true")
    #include optimizer and scheduler in saving and loading, only works when using the same train script
    parser.add_argument("--save_load_optimizer_scheduler", default=False, action="store_true")
    parser.add_argument("--save_nifti", default=False, action="store_true")

    # Guidance Signal Hyperparameters
    parser.add_argument("--sigma", type=int, default=7)
    parser.add_argument("--no_disks", default=False, action="store_true")
    parser.add_argument("--gdt", default=False, action="store_true")


    # Set up additional information concerning the environment and the way the script was called
    args = parser.parse_args()
    return args


def setup_environment_and_adapt_args(args):
    args.caller_args = sys.argv
    args.env = os.environ
    args.git = get_git_information()

    device = torch.device(f"cuda:{args.gpu}")
    
    args.labels = {"organ":args.organ, "background": 0}
    #AMOS dataset
    #1 spleen
    #2 right kidney
    #3 left kidney
    #4 gall bladder
    #5 esophagus
    #6 liver
    #7 stomach
    #8 aorta
    #9 postcava
    #10 pancreas
    #11 right adrenal gland
    #12 left adrenal gland
    #13 duodenum
    #14 bladder
    #15 prostate/ uterus

    if not args.dont_check_output_dir and os.path.isdir(args.output_dir):
        raise UserWarning(
            f"output path {args.output_dir} already exists. Please choose another path or set --dont_check_output_dir"
        )
    if not os.path.exists(args.output_dir):
        pathlib.Path(args.output_dir).mkdir(parents=True)

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    if args.no_log:
        log_folder_path = None
    else:
        log_folder_path = args.output_dir
    setup_loggers(loglevel, log_folder_path)
    logger = get_logger()

    if args.eval_only:
        # Avoid a loading error from the training where it complains the number of epochs is too low
        args.epochs = 100000

    if args.cache_dir == "None":
        if not args.throw_away_cache:
            raise UserWarning("Cache directory (-c) has to be set if args.throw_away_cache is not True")
        else:
            args.cache_dir = tempfile.TemporaryDirectory().name
    else:
        if args.throw_away_cache:
            args.cache_dir = f"{args.cache_dir}/{uuid.uuid4()}"
        else:
            logger.warning("Reusing the cache_dir between different network runs may lead to cache inconsistencies.")
            logger.warning("Most importantly the crops may not be updated if you set them differently")
            logger.warning("PersistentDataset does not detect this automatically but only checks if the hash matches")
            logger.warning("Waiting shortly...")
            time.sleep(10)
            args.cache_dir = f"{args.cache_dir}"

    if not os.path.exists(args.cache_dir):
        pathlib.Path(args.cache_dir).mkdir(parents=True)

    if args.data_dir == "None":
        args.data_dir = f"{args.output_dir}/data"
        logger.info(f"--data was None, so that {args.data_dir}/data was selected instead")

    if not args.no_data:
        if not os.path.exists(args.data_dir):
            pathlib.Path(args.data_dir).mkdir(parents=True)


    args.real_cuda_device = get_actual_cuda_index_of_device(torch.device(f"cuda:{args.gpu}"))

    logger.info(f"CPU Count: {os.cpu_count()}")
    logger.info(f"Num threads: {torch.get_num_threads()}")

    args.cwd = os.getcwd()

    if args.gpu_size == "None":
        nv_total = gpu_usage(device, used_memory_only=False)[3]
        if nv_total < 25:
            args.gpu_size = "small"
        elif nv_total < 55:
            args.gpu_size = "medium"
        else:
            args.gpu_size = "large"
        logger.info(f"Selected GPU size: {args.gpu_size}, since GPU Memory: {nv_total} GB")


    return args, logger
