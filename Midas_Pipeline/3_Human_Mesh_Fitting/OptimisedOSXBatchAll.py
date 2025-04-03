import os
import sys
import os.path as osp
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import cv2
from decord import VideoReader, cpu
from alive_progress import alive_bar

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))

from config import cfg
from vid2img import video_to_images
from common.base import Demoer
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import save_obj
from common.utils.human_models import smpl_x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--input_folder', type=str, default='video_data/dkk')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
    return parser.parse_args()

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

cfg.set_additional_args(encoder_setting='osx_l', decoder_setting='normal', pretrained_model_path=args.pretrained_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once
print("Loading models...")
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval().to(device)
detector.half()  # Use FP16 precision for speed

demoer = Demoer()
demoer._make_model()
demoer.model.to(device).eval()
demoer.model.half()  # FP16 precision
print('Loaded checkpoint from {}'.format(args.pretrained_model_path))

transform = transforms.ToTensor()

def process_images(image_folder, output_folder):
    """Process images in a folder and save outputs with a progress bar."""
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    batch_size = 4
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

    with alive_bar(len(image_files), title=f'Processing images in {image_folder}') as bar:
        for batch in batches:
            imgs = []
            orig_sizes = []
            img_paths = []

            # Load images in batch
            for img_file in batch:
                img_path = osp.join(image_folder, img_file)
                original_img = load_img(img_path)
                orig_sizes.append(original_img.shape[:2])
                img_paths.append(img_path)
                imgs.append(original_img)

            # Detect humans in batch
            imgs_tensor = [transform(img.astype(np.float32)) / 255 for img in imgs]
            imgs_tensor = torch.stack(imgs_tensor).to(device).half()

            with torch.no_grad():
                results = detector(imgs_tensor)

            for i, img_path in enumerate(img_paths):
                person_results = results.xyxy[i][results.xyxy[i][:, 5] == 0]
                boxes = [detection.tolist()[:4] for detection in person_results]

                for num, bbox in enumerate(boxes):
                    bbox = process_bbox(bbox, orig_sizes[i][1], orig_sizes[i][0])
                    img, _, _ = generate_patch_image(imgs[i], bbox, 1.0, 0.0, False, cfg.input_img_shape)
                    img = transform(img.astype(np.float32)) / 255
                    img = img.to(device).half()[None, :, :, :]

                    with torch.no_grad():
                        out = demoer.model({'img': img}, {}, {}, 'test')
                    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

                    save_obj(mesh, smpl_x.face, osp.join(output_folder, f'frame_{i:04d}_person_{num}.obj'))
                
                bar()  # Update progress bar

def process_video(video_path, output_folder):
    """Process a single video file with a progress bar."""
    os.makedirs(output_folder, exist_ok=True)

    # Read video using Decord
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = [vr[i].asnumpy() for i in range(len(vr))]

    frame_folder = osp.join(output_folder, "frames")
    os.makedirs(frame_folder, exist_ok=True)

    with alive_bar(len(frames), title=f'Extracting frames from {osp.basename(video_path)}') as bar:
        # Save frames
        for i, frame in enumerate(frames):
            frame_path = osp.join(frame_folder, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            bar()  # Update progress bar
    
    # Process images
    process_images(frame_folder, output_folder)

def worker(task_queue):
    """Worker function for parallel processing."""
    while not task_queue.empty():
        video_path, output_folder = task_queue.get()
        try:
            process_video(video_path, output_folder)
            print("vid processed")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

def process_videos(input_folder, output_folder, num_workers=4):
    """Recursively process videos while preserving the folder structure with progress bars."""
    task_queue = mp.Queue()
    video_list = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_path = osp.join(root, file)
                relative_path = osp.relpath(root, input_folder)
                video_output_folder = osp.join(output_folder, relative_path)
                os.makedirs(video_output_folder, exist_ok=True)
                video_list.append((video_path, video_output_folder))
    
    for video in video_list:
        task_queue.put(video)

    print(f'Found {len(video_list)} videos to process.')

    processes = []
    with alive_bar(len(video_list), title="Processing videos") as bar:
        for _ in range(num_workers):
            p = mp.Process(target=worker, args=(task_queue,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            bar()  # Update progress bar

if __name__ == "__main__":
    process_videos(args.input_folder, args.output_folder, args.num_workers)

