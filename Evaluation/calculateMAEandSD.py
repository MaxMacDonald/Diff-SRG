import numpy as np
import os
import argparse
from alive_progress import alive_bar

def compute_metrics(real, recon):
    """Compute Mean Absolute Error (MAE) and Standard Deviation (STD) between two arrays."""
    mae = np.mean(np.abs(real - recon))
    std = np.std(real - recon)
    return mae, std

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing recon data")
    parser.add_argument("--real_data_folder", type=str, required=True, help="Folder containing real data with the same structure")
    return parser.parse_args()

# Parse input arguments
args = parse_args()
input_folder = args.input_folder
real_data_folder = args.real_data_folder

# Dictionary to store matched real and recon pairs
data_dict = {}

# Scan recon data folder
with alive_bar(title="Processing Recon Files") as bar:
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().startswith('recon'):
                file_path = os.path.join(root, file)
                splitPath = file_path.split('/')[-4:-1]  # Extract key
                key = "_".join(splitPath)

                if key not in data_dict:
                    data_dict[key] = {}

                data_dict[key]['recon'] = np.load(file_path)
                bar()

# Scan real data folder
with alive_bar(title="Processing Real Files") as bar:
    for root, _, files in os.walk(real_data_folder):
        for file in files:
            if file.lower().startswith('dopp'):
                file_path = os.path.join(root, file)
                splitPath = file_path.split('/')[-4:-1]  # Extract key
                key = "_".join(splitPath)

                if key not in data_dict:
                    data_dict[key] = {}

                data_dict[key]['real'] = np.load(file_path)
                bar()

# Compute MAE and STD for each pair
mae_list = []
std_list = []

for key, data_pair in data_dict.items():
    if 'real' in data_pair and 'recon' in data_pair:
        real_data = data_pair['real']
        recon_data = data_pair['recon']

        # Ensure both arrays have the same shape
        if real_data.shape == recon_data.shape:
            mae, std = compute_metrics(real_data, recon_data)
            mae_list.append(mae)
            std_list.append(std)
        else:
            print(f"Skipping {key}: Shape mismatch {real_data.shape} vs {recon_data.shape}")

# Compute overall average metrics
if mae_list and std_list:
    avg_mae = np.mean(mae_list)
    avg_std = np.mean(std_list)

    print(f"\nAverage MAE: {avg_mae:.6f}")
    print(f"Average STD: {avg_std:.6f}")
else:
    print("No valid pairs found for comparison.")

