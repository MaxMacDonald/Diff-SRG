import os
import sys
from PIL import Image

def convert_images(input_dir, output_dir):
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # Only process PNG files
            png_path = os.path.join(input_dir, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_dir, jpg_filename)
            
            try:
                # Open PNG file, convert to RGB (required for JPG), and save as JPG
                with Image.open(png_path) as img:
                    img.convert("RGB").save(jpg_path, "JPEG")
                print(f"Converted {filename} to {jpg_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_images.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]  # First argument is the input directory
    output_dir = sys.argv[2]  # Second argument is the output directory

    convert_images(input_dir, output_dir)


