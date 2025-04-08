"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import re
import datetime

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util_cond import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from PIL import Image
import torchvision.transforms as transforms
from cm.radarloader_coloradar_benchmark import *
import random

seed = 42
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
import yaml
from easydict import EasyDict as edict

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import open3d as o3d
import math

MAX_RANGE = 16.0
IMAGE_SHAPE = 160
RANGE_RESOLUTION = 0.125

def filename_to_path(filename,base_dir):
    match = re.match(r"(\w+)_(\d+)_(\d+_r?|\d+)_(fine(?:_reconstructed)?|real|synth)(?:\.npy)", filename)
    if not match:
        raise ValueError(f"Filename format not recognized: {filename}")
    
    person, azimuth, elevation, category = match.groups()
    dir_path = os.path.join(base_dir, person, azimuth, elevation)

    # Ensure directory exists
    os.makedirs(dir_path, exist_ok=True)
    
    return os.path.join(dir_path, f"{category}.npy")

def reconstruct_from_rolling_windows(rolling_windows, overlap):
    window_size = 32
    step_size = window_size - overlap
    num_windows = rolling_windows.shape[0]  # Should be 14
    original_length = step_size * (num_windows - 1) + window_size  # Should reconstruct to 250

    reconstructed = np.zeros((32, original_length))  # Shape: (32, 250)
    count = np.zeros((32, original_length))  # To keep track of contributions

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        reconstructed[:, start_idx:end_idx] += rolling_windows[i]  # Sum overlapping regions
        count[:, start_idx:end_idx] += 1  # Count contributions

    # Normalize overlapping regions
    reconstructed /= np.where(count == 0, 1, count)  # Avoid division by zero

    return reconstructed

def compute_metrics(real, recon):
    """Compute Mean Absolute Error (MAE) and Standard Deviation (STD) between two arrays."""
    mae = np.mean(np.abs(real - recon))
    std = np.std(real - recon)
    return mae, std

def log_model_performance(model_name, mae, sd, log_file="model_performance.log"):
    # Get current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if log file exists
    file_exists = os.path.isfile(log_file)

    # Open file in append mode
    with open(log_file, "a") as f:
        # Write header if file is newly created
        if not file_exists:
            f.write("Timestamp,Model Name,MAE,Standard Deviation\n")

        # Append new log entry
        f.write(f"{timestamp},{model_name},{mae},{sd}\n")

    print(f"Logged: {timestamp} | Model: {model_name} | MAE: {mae} | SD: {sd}")

def main():
    args = create_argparser().parse_args()

    # print("args.in_ch", args.in_ch)
    # print("args.out_ch", args.out_ch)
    args.in_ch = 2
    args.out_ch = 1
    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False


    logger.log("creating data loader...")
    ### Dataloader ###
    data_dir = args.dataset_dir

    person = ['zzh']  # modify if leaving one out
    loaded_data = []
    real_data_array = []
    synth_filenames = []

    for p in person:
        real_dir = os.path.join(data_dir, p, 'real')
        synth_dir = os.path.join(data_dir, p, 'synth')
        print(synth_dir)
        synth_files = sorted([f for f in os.listdir(synth_dir) if f.endswith('_synth.npy')])
        synth_filenames.append(synth_files)

        for synth_file in synth_files:
            # Derive real filename
            synth_path = os.path.join(synth_dir, synth_file)
            real_file = synth_file.replace('_synth.npy','_real.npy')
            real_path = os.path.join(real_dir, real_file)

            if not os.path.exists(synth_path):
                print(f"Warning: Synth file {synth_path} not found")
                logger.log(f"Warning: Synth file {synth_path} not found")
                continue

            # Load file
            synth_data = np.load(synth_path)  # shape (m,32,32)
            real_data = np.load(real_path)  # shape (m,32,32)
            # Add to list
            loaded_data.append(synth_data)
            real_data_array.append(real_data)

    # Limit number of samples during testing
    #loaded_data = loaded_data[:2]
    if not args.batch_size % loaded_data[0].shape[0] == 0:
        logger.log(f"Warning: Batch_size not compatible with doppler size")

    # Concatenate along the first axis
    data = np.concatenate(loaded_data, axis=0)    # shape (n,32,32)
    logger.log(f"Number of samples loaded: {data.shape[0]}")
    synth_filenames = np.concatenate(synth_filenames, axis=0)



    # dataset_config_path = args.dataloading_config
    # with open(dataset_config_path, 'r') as fid:
    #     coloradar_config = edict(yaml.load(fid, Loader=yaml.FullLoader))

    # tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    # transform_train = transforms.Compose(tran_list)
    # test_data = init_dataset(coloradar_config, args.dataset_dir, transform_train, "test")

    test_data = SynthOnlyDataset(data)

    datal= th.utils.data.DataLoader(
        test_data,
        num_workers=16,
        batch_size=args.batch_size,
        shuffle=True)

    data = iter(datal)
    logger.log(args.model_path)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    checkpoint = th.load(args.model_path, map_location="cpu")
    new_checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_checkpoint)
    #model.load_state_dict(
    #    dist_util.load_state_dict(args.model_path, map_location="cpu")
    #)
    # model.to("cpu")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None


    generator = get_generator(args.generator, args.num_samples, args.seed)

    # i = 0

    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    cnt = 0
    doppler_num = 0
    mae_arr = []
    sd_arr = []

    for test_i, b in enumerate(datal):
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        b = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b)
        img = th.cat((b, c), dim=1)     #add a noise channel$

        b = b.to(dist_util.dev())
        radar_condition_dict = {"y": b}

        start.record()
        sample = karras_sample(
            diffusion = diffusion,
            model = model,
            shape = (args.batch_size, 1, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=radar_condition_dict,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
        )

        end.record()
        th.cuda.synchronize()
        print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

        cnt = cnt + args.batch_size
        print("cnt", cnt)

        # Convert to NumPy and remove the singleton dimension
        sample_np = sample.squeeze(1).cpu().numpy()  # Shape: (14, 32, 32)
        filename = synth_filenames[doppler_num]
        filename_fine = filename.replace("_synth.npy", "_fine.npy")  # Replace suffix
        filename_recon = filename.replace("_synth.npy", "_fine_reconstructed.npy")  # Replace suffix

        output_path_fine = filename_to_path(filename_fine,args.output_dir)
        output_path_recon = filename_to_path(filename_recon,args.output_dir)
        print(output_path_fine)
        # reconstruct data and save
        reconstructed_np = reconstruct_from_rolling_windows(sample_np,16)
        np.save(output_path_recon, reconstructed_np)
        np.save(output_path_fine, sample_np)
        # compute MAE and SD to help with fine tuning
        mae, sd = compute_metrics(real_data_array[doppler_num],sample_np)
        print(f"MAE: {mae}, SD: {sd}")
        mae_arr.append(mae)
        sd_arr.append(sd)
        doppler_num += 1
    log_model_performance(args.model_path, np.mean(mae_arr),np.mean(sd_arr))
    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        # data_dir="/home/ubuntu/coloradar_ws/consistency_model_coloradar/cmu_dataset/",
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=100,
        batch_size=14,
        sampler="euler",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1,
        steps=100,
        model_path="/home/max/mastersProject/Radar-Diffusion-main/diffusion_consistency_radar/train_results/radar_edm_custom_ddk/ema_0.999_003000.pt",
        #model_path="/home/max/mastersProject/Radar-Diffusion-main/diffusion_consistency_radar/train_results/radar_edm_custom_zzh/ema_0.999_003000.pt", 
        seed=42,
        ts="",
        in_ch = 2,
        out_ch = 1,
        dataset_dir = '/home/max/Results/data32x32',
        dataloading_config = "./config/data_loading_config_example.yml",
        # dataset_dir = '/home/zrb/Mmwave_Dataset/Coloradar/coloradar_after_preprocessing',
        # dataloading_config = "./config/data_loading_config_example.yml",
        output_dir = "/home/max/Results/generatedData_Diffusion",
        #output_dir = "/home/max/Results/data32x32/dkk/fine", ##edgar, outdoors, arpg_lab, ec_hallways, aspen, longboard
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

class SynthOnlyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        return x

if __name__ == "__main__":
    main()
