import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch



# Break down video into frames (using Vid2Image)
# For each frame predict using yolo, find humans with confidence of 0.7 or higher (as a starting point) (do this using an array of the frames of the video)
# For each frame i, find frame i - 1. Iterate through boxes, if their positions are 5-10% different in position then assmume they are the same human and add this
# region to an array with index n where n is the index of the human in the scene. If it doesn't exist add a new array with index n+1.

# Perform depth estimation and mesh fitting on each indexed human (lets throw this all in a dictionary for ease of use)

def are_boxes_similar(box1_full,box2_full):
    # To determine the thesholds for similarity take a video with only 1 subject that is always in view e.g. the first midas vid
    # Determine the differences between each frame and find the mean, mode, maximum and minimum differences,
    # go kinda close to this.
    box1 = box1_full.xyxy[0]
    box1 = [box1[0].cpu().numpy(),box1[1].cpu().numpy(),box1[2].cpu().numpy(),box1[3].cpu().numpy()]
    box2 = box2_full.xyxy[0]
    box2 = [box2[0].cpu().numpy(),box2[1].cpu().numpy(),box2[2].cpu().numpy(),box2[3].cpu().numpy()]
    # Compute centroids of each box as x,y pairs
    box1_centroid = [(box1[0] + box1[2])/2,(box1[1]+box1[3])/2]
    box2_centroid = [(box2[0] + box2[2])/2,(box2[1]+box2[3])/2]
    #box1_centroid = box1_centroid.cpu().numpy() #if isinstance(box1_centroid, torch.Tensor) else box1_centroid
    #box2_centroid = box2_centroid.cpu().numpy() #if isinstance(box2_centroid, torch.Tensor) else box2_centroid
    #print("box1 centroid = ", box1_centroid)
    #print("box2 centroid = ", box2_centroid)
    boxes = np.array([box1_centroid, box2_centroid])
    distance = np.linalg.norm(boxes[0]-boxes[1])
    # Normalise distance based on resolution of the image
    x_diff = np.square(box1_centroid[0]/box1_full.orig_shape[0] - box2_centroid[0]/box2_full.orig_shape[0])
    y_diff = np.square(box1_centroid[1]/box1_full.orig_shape[1] - box2_centroid[1]/box2_full.orig_shape[1])
    total_diff = x_diff + y_diff
    if total_diff < 0.005:
        return True
    else:
        return False
        print("frame rejected")



# Load a model
model = YOLO("/home/max/mastersProject/MastersProject/yolo11n.pt")  # load an official model


# Path to the folder containing the images
image_folder = "/home/max/Documents/images/sample_video"

# Get all image paths
frames = []
for image_name in os.listdir(image_folder):
    # Create the full path to the image file
    image_path = os.path.join(image_folder, image_name)
    # Append the path to the list if it's a file
    if os.path.isfile(image_path):
        frames.append(image_path)

#frames = "https://ultralytics.com/images/bus.jpg"

results = model(frames, classes = [0])  # predict only person objects on an image
framesWithBoxes = []
# Process results list
for result in results:
    orignialImage = result.orig_img
    boxesInFrame = []
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    #result.save(filename="result.jpg")  # save to disk

    # Check confidence rating of each box
    for box in boxes:
        if box.conf > 0.8:
            print("found one!")
            print(box.xyxy)
            x1 = int(box.xyxy[0][0])
            y1 = int(box.xyxy[0][1])
            x2 = int(box.xyxy[0][2])
            y2 = int(box.xyxy[0][3])
            croppedImage = orignialImage[y1:y2,x1:x2]
            boxesInFrame.append([box,croppedImage])

    framesWithBoxes.append(boxesInFrame)

#   Iterate through each frame, compare each box to the boxes in the previous frame
#   if they are within tolerance of each other then assign to the same human index
#   if they are not (or its the first frame) add to a new index.
human_region_index = {}
human_index = 0
diffs = []
for i in range(len(framesWithBoxes)):
    if i == 0:
        for box,image in framesWithBoxes[i]:
            name = "human_" + str(human_index)
            human_region_index[name] = [[box,image]]
            human_index += 1
    else:
        currentFrame = framesWithBoxes[i]
        previous_frame = framesWithBoxes[i-1]
        # Check all boxes in each frame, if they are similar to a box in the previous frame add to that human.
        for j in range(len(currentFrame)):
            box = currentFrame[j][0]
            image = currentFrame[j][1]
            noneSimilar = True
            for key,value in human_region_index.items():
                if are_boxes_similar(box,value[-1][0]):
                    noneSimilar = False
                    human_region_index[key].append([box,image])


            if noneSimilar == True:
                name = "human_" + str(human_index)
                human_region_index[name] = [[box,image]]
                human_index += 1

for key,value in human_region_index.items():
    # Create a new folder for the current key if it doesn't exist
    folder_path = os.path.join("/home/max/Documents/visualisations", key)
    os.makedirs(folder_path, exist_ok=True)
    for i in range(len(value)):
        image = value[i][1]
        image_name = f"frame_{i:03d}"
        # Convert the NumPy array to a Pillow Image
        pil_image = Image.fromarray(image)

        # Construct the save path
        save_path = os.path.join(folder_path, f"{image_name}.png")

        # Save the image as a PNG
        pil_image.save(save_path)


# humans = {human_1: [frame1box,frame2box,frame3box]}
# frame1 = [box1,box2 ...] -> [frame1boxes, frame2boxes] -> {human_1: [frame1box,frame1image],[frame2box,frame2image],[frame3box,frame3image]]}


# Need to crop the image when a box is found and just add that image instead. Then we can iterate through each human
# and save all their images into a folder.
