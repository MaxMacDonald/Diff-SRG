import argparse
import os
import numpy as np
import torch as th
from cm import dist_util, logger
from cm.script_util_cond import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    # Create model
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    generator = get_generator(args.generator, args.num_samples, args.seed)

    synth_data_dir = args.synth_data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    synth_files = [f for f in os.listdir(synth_data_dir) if f.endswith('_synth.npy')]
    logger.log(f"Found {len(synth_files)} synth files for inference.")

    for file in synth_files:
        synth_data = np.load(os.path.join(synth_data_dir, file))
        # If each file is (m,32,32), iterate over samples
        for idx, sample_coarse in enumerate(synth_data):
            # Prepare conditioning tensor
            coarse_tensor = th.tensor(sample_coarse, dtype=th.float32).unsqueeze(0).unsqueeze(0).to(dist_util.dev())
            model_kwargs = {"y": coarse_tensor}

            with th.no_grad():
                generated_sample = karras_sample(
                    diffusion=diffusion,
                    model=model,
                    shape=(1, 1, 32, 32),
                    steps=args.steps,
                    model_kwargs=model_kwargs,
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
                    ts=None,
                )

            gen_numpy = generated_sample.squeeze().cpu().numpy()
            gen_numpy_img = Image.fromarray((gen_numpy * 255).astype(np.uint8))
            save_name = os.path.join(output_dir, f"{file[:-9]}_sample_{idx}.png")
            gen_numpy_img.save(save_name)
            logger.log(f"Saved generated sample to {save_name}")

def create_argparser():
    defaults = dict(
        model_path="/home/max/mastersProject/Radar-Diffusion-main/diffusion_consistency_radar/train_results/radar_edm_custom/model000300.pt",
        synth_data_dir="/home/max/Results/data32x32",
        output_dir="./inference_results",
        image_size=32,
        batch_size=1,
        steps=50,
        clip_denoised=True,
        sampler="heun",
        sigma_min=0.002,
        sigma_max=80,
        s_churn=0,
        s_tmin=0,
        s_tmax=float('inf'),
        s_noise=1,
        generator="numpy",
        num_samples=1,
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
