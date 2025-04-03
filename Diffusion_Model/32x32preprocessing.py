import os
import numpy as np

# Modify these paths
synth_dir = "/home/max/Results/intensitymapping"
gt_dir = "/home/max/mastersProject/Midas/doppler_data/radar_data"
output_dir = "/home/max/Results/data32x32"

person = ['dkk', 'zzh']
angle = ['0', '45', '90']
overlap = 16  # example overlap, modify as needed

# Make sure output directories exist
for p in person:
    for data_type in ['real', 'synth']:
        os.makedirs(os.path.join(output_dir, p, data_type), exist_ok=True)

def get32x32rollingwindow(dop_data, overlap):
    window_size = 32
    step_size = window_size - overlap
    num_windows = (dop_data.shape[1] - window_size) // step_size + 1
    
    rolling_windows = []
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = dop_data[:, start_idx:end_idx]
        rolling_windows.append(window)
    
    rolling_windows = np.array(rolling_windows)  # Shape: (num_windows, 32, 32)
    return rolling_windows

for p in person:
    for a in angle:
        activity_list = sorted(os.listdir(os.path.join(synth_dir, p, a)))
        for active in activity_list:
            print(active)
            # Load real and synth data
            real_data = np.load(os.path.join(gt_dir, p, a, active, "doppler_gt.npy"))
            synth_data = np.load(os.path.join(synth_dir, p, a, active, "synthDopplerData.npy"))
            
            # Apply rolling window
            rolling_real = get32x32rollingwindow(real_data, overlap)
            rolling_synth = get32x32rollingwindow(synth_data, overlap)
            print(f"real ={rolling_real.shape} ")
            print(f"real ={rolling_synth.shape} ")
            # Save results
            filename_real = f"{p}_{a}_{active}_real.npy"
            filename_synth = f"{p}_{a}_{active}_synth.npy"

            np.save(os.path.join(output_dir, p, 'real', filename_real), rolling_real)
            np.save(os.path.join(output_dir, p, 'synth', filename_synth), rolling_synth)

print("Processing and saving complete.")

