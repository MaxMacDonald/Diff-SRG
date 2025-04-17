import numpy as np
import os
import sys
import argparse
from alive_progress import alive_bar
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='video_data/dkk')
    parser.add_argument('--visualisationFolder', type=str, default='visualisations')
    parser.add_argument('--realDataFolder', type=str, default=None)
    return parser.parse_args()

def plot_data(name,data,path,visPath):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', cmap='viridis', origin='lower')
    #plt.colorbar(label='Intensity')
    #plt.title(name)
    #plt.xlabel('Time/Frequency Bins')
    #plt.ylabel('Radial Velocity (m/s)')

    #num_channels = data.shape[0]  # Number of sensor channels
    #y_ticks = np.linspace(0, num_channels - 1, 5)  # Select 5 tick positions
    #y_labels = np.linspace(-2, 2, 5)  # Corresponding velocity values (-2 to +2)
    #plt.yticks(y_ticks, [f"{label:.1f} m/s" for label in y_labels])  # Set labels
    plt.tight_layout()
    plt.axis('off')
    #plt.savefig(path)
    plt.savefig(visPath)
    plt.close()



args = parse_args()
input_folder = args.input_folder
visualisationFolder = args.visualisationFolder
realDataFolder = args.realDataFolder
with alive_bar(66, title="") as bar:
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('fine_reconstructed.npy'):
                fineDataPath = os.path.join(root, file)
                directory_path = os.path.dirname(fineDataPath)
                fineData = np.load(fineDataPath)
                outputPath = os.path.join(directory_path, "vis_reconstructed.png")
                splitPath = fineDataPath.split('/')[-4:-1]
                visualisationPath = os.path.join(visualisationFolder,f"vis_reconstructed_diffusion_{splitPath[0]}_{splitPath[1]}_{splitPath[2]}.png")
                plot_data("Doppler Intensity Map (Fine)",fineData,outputPath,visualisationPath)
                #print(visualisationPath)
                bar()
            elif file.lower().startswith('synth'):
                coarseDataPath = os.path.join(root, file)
                directory_path = os.path.dirname(coarseDataPath)
                coarseData = np.load(coarseDataPath)
                outputPath = os.path.join(directory_path, "vis_coarse.png")
                splitPath = coarseDataPath.split('/')[-4:-1]
                visualisationPath = os.path.join(visualisationFolder,f"vis_coarse_{splitPath[0]}_{splitPath[1]}_{splitPath[2]}.png")
                plot_data("Doppler Intensity Map (Coarse)",coarseData,outputPath,visualisationPath)
                bar()



