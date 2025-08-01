"""
-*- coding: utf-8 -*-
Author: Weiyu Zhang
Date: 2025-02-17
Description:
This script is used to extract embeddings from pretrained MAE models for downstream urban potential prediction tasks. We strongly recommend (and this script only implement) to use a file list parquet file
for data loading, which is much more efficient for large-scale data processing.
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import timm
from tqdm import tqdm
import models.models_vit as models_vit
import util.loader
from util.pos_embed import interpolate_pos_embed
import torch.nn as nn
import pandas as pd

def save_and_cleanup(data, batch_count, save_path, name):
    df = pd.DataFrame(data)
    embeddings = np.stack(df['embedding'].values)
    np.save(os.path.join(save_path, f'embeddings_{name}_{batch_count}.npy'), embeddings)
    df.drop(columns=['embedding'], inplace=True)
    csv_file = os.path.join(save_path, f'embeddings_{name}_{batch_count}.csv')
    df.to_csv(csv_file, index=False)
    print(f'Saved batch {batch_count} results to {csv_file}')

    # clean up memory
    del df, data
    torch.cuda.empty_cache()
    print('Memory cleaned up.')

def main():
    parser = argparse.ArgumentParser(description='Embeddings Extraction Script')
    parser.add_argument('--year', type=int, default=None, help='Year for climate data (9-20)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--batch_size', type=int, default=1200, help='Batch size for DataLoader')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker threads for DataLoader')
    parser.add_argument('--save_root', type=str, required=True, help='Root directory to save embeddings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained model checkpoint')
    parser.add_argument('--model', type=str, default='vit_large_patch16', help='Model architecture')
    parser.add_argument('--clim', action='store_true', help='Use climate data')
    parser.add_argument('--clim_multiband', action='store_true', help='use climate dataset(multi-band tiff file)')
    parser.add_argument('--dem', action='store_true', help='Use DEM data')
    parser.add_argument('--file_list', type=str, default=None, help='Path to file list parquet file for data')
    parser.add_argument('--base_name', type=str, default=None, help='Final embedding dimension')

    args = parser.parse_args()

    print('args.clim: ', args.clim)
    print('args.clim_multiband: ', args.clim_multiband)

    # set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for inference")

    # create model
    print(f"=> creating model '{args.model}'")
    model_instance = models_vit.__dict__[args.model](
        in_chans=9 if args.clim else 2,  # set number of in-channels according to data type
        img_size=24 if args.clim else 160,
        global_pool=False,
        embed_dim=128 if args.clim else 256,
        final_emb_dim=None,
        patch_size=4 if args.clim else 16,
    )

    # load pretrained model
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    print(f"Load pre-trained checkpoint from: {args.pretrained}")
    checkpoint_model = checkpoint['model']
    state_dict = model_instance.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Interpolate positional embeddings to match the current model's input size
    interpolate_pos_embed(model_instance, checkpoint_model)

    # load model weights
    msg = model_instance.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # Modify the model's head
    model_instance.head = nn.Sequential(nn.Identity())

    torch.cuda.set_device(args.gpu)
    model_instance = model_instance.to(device)
    model_instance.eval()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)

    # set dataset and name
    if args.clim:
        if args.year is None:
            raise ValueError("Year must be specified for climate data")
        print(f"\nProcessing climate data, year: {args.year}")
        name = f"{args.base_name}-Y{args.year}00"
        clim_file_list = args.file_list
    elif args.dem:
        print("Processing DEM data")
        dem_water_file_list = args.file_list
        name = f"{args.base_name}"
        print('name (args.base_name): ', name)
    else:
        raise ValueError("Either --clim or --dem must be specified")

    # set save path
    save_path = os.path.join(args.save_root, name)
    if os.path.isdir(save_path):
        if os.listdir(save_path):
            raise AssertionError(f'Folder already exists and is not empty: {save_path}')
    else:
        os.makedirs(save_path, exist_ok=True)

    # load data
    if args.clim:
        if not args.clim_multiband:
            raise NotImplementedError
        else:
            # multi-band climate files
            print("=> loading climate dataset (multi-band files)")
            val_dataset = util.loader.ImageValDatasetClimAllMultiBand(
                filelist=clim_file_list,
                year=args.year
            )
    elif args.dem:
        print("=> loading DEM dataset")
        val_dataset = util.loader.ImageValDatasetWithBinaryWaterParquetFilelist(
            filelist=dem_water_file_list
        )
    else:
        # other datasets
        pass

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    print("=> dataset loaded")

    data = []
    batch_count = 0
    total_batches = len(val_loader)
    save_interval = 6000  # Lower save_interval if memory is not enough

    with torch.no_grad():
        for index, (images, mean, std, ids) in tqdm(enumerate(val_loader), total=total_batches):
            # move the images to GPU
            images = images.to(device, non_blocking=True)

            # compute model output
            output = model_instance.forward_features(images)

            # move output and other data to CPU in one go
            output_cpu = output.cpu().numpy()
            mean_cpu = mean.numpy()
            std_cpu = std.numpy()

            # add data to the list in batch
            data.extend([
                {'id': id_, 'embedding': emb, 'mean': mean_, 'std': std_}
                for id_, emb, mean_, std_ in zip(ids, output_cpu, mean_cpu, std_cpu)
            ])

            # save and clean up
            if (index + 1) % save_interval == 0 or (index + 1) == total_batches:
                save_and_cleanup(data, batch_count, save_path, name)
                data = []  # reset the result list to free memory
                batch_count += 1

        # handle any remaining data
        if len(data) > 0:
            save_and_cleanup(data, batch_count, save_path, name)

if __name__ == '__main__':
    main()
