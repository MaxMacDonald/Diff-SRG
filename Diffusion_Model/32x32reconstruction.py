import os
import numpy as np

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

def process_npy_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            data = np.load(filepath)
            processed_data = reconstruct_from_rolling_windows(data,16)
            
            new_filename = filename.replace(".npy", "_reconstructed.npy")
            new_filepath = os.path.join(directory, new_filename)
            
            np.save(new_filepath, processed_data)
            print(f"Processed and saved: {new_filename}")

if __name__ == "__main__":
    directory = "/home/max/Results/data32x32/dkk/fine"  # Change this to your actual directory path
    process_npy_files(directory)
