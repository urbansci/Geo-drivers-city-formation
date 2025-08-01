"""
# -*- coding: utf-8 -*-
Author: Weiyu Zhang
Date: 2024
Description: Main script for pre-training a Masked Autoencoder (MAE) model using distributed data parallel.
             Select model type (dem-water or clim) via the --model_type flag.
"""

import builtins
import math
import os
import random
import shutil
import warnings
import argparse
import datetime
import json
import numpy as np
import time
from pathlib import Path

import neptune
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

import models.models_mae as models_mae
import util.loader
import util.misc as misc
from engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def get_args_parser():
    """
    Creates and configures the argument parser for command-line options.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Configuration for model and data type
    parser.add_argument('--model_type', type=str, default='dem-water', choices=['dem-water', 'clim'],
                        help='Type of model and data to use: "dem-water" or "clim"')

    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model architecture to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size (overridden by model_type settings)')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='Dataset path')

    # Directory and device parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='Path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--num_workers', default=98, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--neptune', action='store_true', help='Enable Neptune logging.')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='Rank of this process')
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training')
                        
    return parser


def main():
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting from checkpoints.')
    else:
        cudnn.benchmark = True
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Adjust world size based on the number of GPUs per node
        args.world_size = ngpus_per_node * args.world_size
        # Launch distributed processes using torch.multiprocessing.spawn
        print(f"Spawning {ngpus_per_node} processes per node, total world size: {args.world_size}")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Directly call the main worker function for single-process training
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    """
    The core training worker function for each process/GPU.

    Args:
        gpu (int): The GPU ID for the current process.
        ngpus_per_node (int): The total number of GPUs on the node.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    args.gpu = gpu
    
    # Initialize Neptune logger for the master process
    run = None
    if args.gpu == 0 and args.neptune:
        run = neptune.init_run(
            project="S3-Lab/Geo-SSL",
            api_token="YOUR_API_TOKEN_HERE", # Replace with your actual token
        )
        run["parameters"] = vars(args)
        run_id = run['sys/id'].fetch()
        # Create a directory for saved models based on the Neptune run ID
        os.makedirs(f'/data0/zwy/saved_models/{run_id}', exist_ok=True)
    
    # Suppress printing for non-master processes
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
        
    # Initialize distributed training
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend="nccl", init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # --- Model, Data, and Augmentation Configuration based on model_type ---
    print(f"Configuring for model_type: {args.model_type}")
    if args.model_type == 'dem-water':
        # DEM model configuration
        model = models_mae.__dict__[args.model](
            img_size=160, 
            in_chans=2, 
            embed_dim=256, 
            decoder_embed_dim=256,
            patch_size=16,
            norm_pix_loss=args.norm_pix_loss
        )
        augmentation = []
        dataset_train = util.loader.ImageDatasetWithBinaryWater(
            args.data_path, 
            transform=transforms.Compose(augmentation)
        )

    elif args.model_type == 'clim':
        # Climate model configuration
        model = models_mae.__dict__[args.model](
            img_size=24, 
            in_chans=9, 
            embed_dim=128, 
            decoder_embed_dim=128,
            patch_size=4,
            norm_pix_loss=args.norm_pix_loss
        )
        # Climate data and transforms
        augmentation_clim = [
            transforms.Normalize(
                mean=[8.757, 806.940, 31.380, 16.095, 3.403, 717.817, 63.785, 201.198, 142.541], 
                std=[14.718, 500.542, 13.907, 9.944, 20.500, 726.910, 37.967, 199.588, 237.565]
            )
        ]
        # NOTE: Select the appropriate climate dataset loader. Using ImageDatasetClimAll as an example.
        dataset_train = util.loader.ImageDatasetClimAll(
            folder=args.data_path, 
            transform=transforms.Compose(augmentation_clim)
        )
    
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    # --- End of Configuration ---

    model.to(args.gpu)
    model_without_ddp = model
    print(f"Model: {str(model_without_ddp)}")
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    print(f"Dataset length: {len(dataset_train)}")
    
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        sampler_train = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=(sampler_train is None),
        num_workers=args.num_workers, pin_memory=True, sampler=sampler_train, drop_last=True)

    # Calculate effective batch size and learning rate
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual LR: {args.lr:.2e}")
    print(f"Accumulate grad iterations: {args.accum_iter}")
    print(f"Effective batch size: {eff_batch_size}")

    # Create optimizer and loss scaler
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    
    # Load model from checkpoint if resume path is provided
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    # Main training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(
            model, data_loader_train,
            optimizer, epoch, loss_scaler, run,
            args=args
        )
        # Save checkpoint periodically and at the end of training
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs) and args.gpu == 0:
            neptune_run_id = run['sys/id'].fetch() if run else None
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, run_id=neptune_run_id)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    main()