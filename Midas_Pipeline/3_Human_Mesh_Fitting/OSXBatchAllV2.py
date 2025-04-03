import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2

os.environ["ULTRALYTICS_NO_UPDATE"] = "1"
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))

from config import cfg
from vid2img import video_to_images
from common.base import Demoer
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import save_obj
from common.utils.human_models import smpl_x
from alive_progress import alive_bar

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--input_folder', type=str, default='video_data/dkk')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    return parser.parse_args()

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

cfg.set_additional_args(encoder_setting='osx_l', decoder_setting='normal', pretrained_model_path=args.pretrained_model_path)
demoer = Demoer()
demoer._make_model()
demoer.model.eval()

assert osp.exists(args.pretrained_model_path), 'Cannot find model at ' + args.pretrained_model_path
print('Loaded checkpoint from {}'.format(args.pretrained_model_path))

transform = transforms.ToTensor()

def process_images(image_folder, output_folder):
    """Process images in a folder and save outputs."""
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    for img_num, img_file in enumerate(image_files):
        img_path = osp.join(image_folder, img_file)

        # Load image
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
        # detect human bbox with yolov5s

        with torch.no_grad():
            results = detector(original_img)
        person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
        class_ids, confidences, boxes = [], [], []
        for detection in person_results:
            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for num, indice in enumerate(indices):
            bbox = boxes[indice]  # x,y,h,w
            bbox = process_bbox(bbox, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
            mesh = mesh[0]

            # Save mesh
            save_obj(mesh, smpl_x.face, osp.join(output_folder, f'frame_{img_num:04d}_person_{num}.obj'))
            print(f'Processed image {img_num} in {image_folder}')

def process_videos(input_folder, output_folder):
    """Recursively process videos while preserving the folder structure."""
    with alive_bar(66, title="Processing videos") as bar:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith('.mp4'):
                    video_path = osp.join(root, file)

                    # Define corresponding output path
                    relative_path = osp.relpath(root, input_folder)
                    video_output_folder = osp.join(output_folder, relative_path)
                    print(video_output_folder)
                    os.makedirs(video_output_folder, exist_ok=True)

                    # Convert video to images
                    image_folder = osp.join(video_output_folder, 'frames')
                    video_to_images(video_path, image_folder)
                    print("Starting to process images!")
                    # Process extracted images
                    process_images(image_folder, video_output_folder)
                    bar()

if __name__ == "__main__":
    process_videos(args.input_folder, args.output_folder)
