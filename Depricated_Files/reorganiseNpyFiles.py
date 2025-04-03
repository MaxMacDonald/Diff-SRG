import os
import shutil
import re


def reorganize_files(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    persons = [p for p in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, p))]

    for person in persons:
        person_dir = os.path.join(source_dir, person)
        subfolders = ['fine', 'real', 'synth']

        for subfolder in subfolders:
            subfolder_path = os.path.join(person_dir, subfolder)
            if not os.path.exists(subfolder_path):
                continue

            for filename in os.listdir(subfolder_path):
                match = re.match(rf"{person}_(\d+)_(\d+_r?|\d+)_({subfolder}.*)\.npy", filename)
                if not match:
                    continue

                azimuth, elevation, category = match.groups()

                dest_azi_path = os.path.join(dest_dir, person, azimuth, elevation)
                os.makedirs(dest_azi_path, exist_ok=True)

                src_file = os.path.join(subfolder_path, filename)
                dest_file = os.path.join(dest_azi_path, filename)

                shutil.move(src_file, dest_file)
                print(f"Moved: {src_file} -> {dest_file}")


# Example usage
source_directory = "/home/max/Results/data32x32"  # Update with actual path
destination_directory = "/home/max/Results/diffusion"  # Update with actual path
reorganize_files(source_directory, destination_directory)

