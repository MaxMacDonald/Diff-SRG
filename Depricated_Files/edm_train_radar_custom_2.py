"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger
# from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util_cond import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.train_util_cond import TrainLoop
import torch.distributed as dist
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from cm.radarloader_coloradar_benchmark import *
import torch as th
import os 

from fvcore.nn import parameter_count_table, FlopCountAnalysis
import yaml
from easydict import EasyDict as edict

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()



    logger.configure(dir = args.out_dir)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    print("dist_util.dev()", dist_util.dev())
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)



    print(parameter_count_table(model))

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size


    logger.log("creating data loader...")
    

    ### Dataloader ###
    data_dir = args.dataset_dir    

    person = ['dkk']  # modify if leaving one out
    data = []
    labels = []

    for p in person:
        real_dir = os.path.join(data_dir, p, 'real')
        synth_dir = os.path.join(data_dir, p, 'synth')

        real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('_real.npy')])

        for real_file in real_files:
            # Derive synth filename
            synth_file = real_file.replace('_real.npy', '_synth.npy')
            synth_path = os.path.join(synth_dir, synth_file)
            real_path = os.path.join(real_dir, real_file)

            if not os.path.exists(synth_path):
                print(f"Warning: Synth file {synth_path} not found for {real_file}")
                logger.log(f"Warning: Synth file {synth_path} not found for {real_file}")
                continue

            # Load files
            real_data = np.load(real_path)  # shape (m,32,32)
            synth_data = np.load(synth_path)  # shape (m,32,32)
            print(synth_path)

            # Add to lists
            labels.append(real_data)
            data.append(synth_data)

    # Concatenate along the first axis
    data = np.concatenate(data, axis=0)    # shape (n,32,32)
    labels = np.concatenate(labels, axis=0) # shape (n,32,32)
    logger.log(f"Number of samples loaded: {data.shape[0]}")

    logger.log(f"Data shape: {data.shape}")
    logger.log(f"Labels shape: {labels.shape}")



    # Make pytorch Dataset object to pass into DataLoader
    train_data = myDopplerDataset(data,labels)

    print("batch_size", batch_size)
    data= th.utils.data.DataLoader(
        train_data,
        num_workers=16,
        batch_size=batch_size,
        shuffle=True)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=30000,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=40000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        in_ch = 2,
        out_ch = 1,
        out_dir='./train_results/radar_edm_custom_testing',
        dataset_dir = '/home/max/Results/data32x32',
        dataloading_config = "./config/data_loading_config_example.yml",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


class myDopplerDataset(Dataset):
    def __init__(self, data, label = None):

        self.data = data
        self.label = label
        self.len = len(data)

    def __getitem__(self, index):
        x = self.data[index]
        x = th.tensor(x, dtype=th.float32).unsqueeze(0)
        name = index

        if self.label is not None:
            y = self.label[index]
            y = th.tensor(y, dtype=th.float32).unsqueeze(0)
            return x, y

        return x

    def __len__(self):

        return self.len

if __name__ == "__main__":
    main()

