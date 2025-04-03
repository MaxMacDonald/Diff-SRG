import os
import shutil

real_data_root = "/home/max/mastersProject/Midas/doppler_data/radar_data"
synthetic_data_root = "/home/max/Results/generatedData_LOO"

for root, dirs, files in os.walk(real_data_root):
    for file in files:
        real_file_path = os.path.join(root, file)
        relative_path = os.path.relpath(real_file_path, real_data_root)
        synthetic_file_path = os.path.join(synthetic_data_root, relative_path)
        
        # Make sure the destination directory exists
        os.makedirs(os.path.dirname(synthetic_file_path), exist_ok=True)
        
        # Copy the file
        shutil.copy2(real_file_path, synthetic_file_path)

