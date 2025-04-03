import trimesh
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import argparse
import glob
import os
from multiprocessing import Pool, cpu_count
from alive_progress import alive_bar

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='input_images')
    args = parser.parse_args()
    return args

def display(values):
    print(f"Shape of input values is {np.shape(values)}")
    doppler_data = np.array(values).T
    plt.figure(figsize=(10, 6))
    plt.imshow(doppler_data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Doppler Intensity Map')
    plt.xlabel('Time/Frequency Bins')
    plt.ylabel('Radial Velocity (m/s)')
    num_channels = doppler_data.shape[0]  # Number of sensor channels
    y_ticks = np.linspace(0, num_channels - 1, 5)  # Select 5 tick positions
    y_labels = np.linspace(-2, 2, 5)  # Corresponding velocity values (-2 to +2)
    plt.yticks(y_ticks, [f"{label:.1f} m/s" for label in y_labels])  # Set labels
    plt.tight_layout()
    plt.savefig("plot.png")
    np.save("synthDopplerData.npy", doppler_data)

def bin_velocities(velocities):
    bins = np.linspace(-2, 2, 33)
    histogram, _ = np.histogram(velocities, bins=bins)
    normalized_histogram = histogram / np.sum(histogram) if np.sum(histogram) > 0 else histogram
    return normalized_histogram

def process_frame(pair):
    mesh1_path, mesh2_path, sensor_position, delta_t = pair
    
    mesh1 = trimesh.load(mesh1_path)
    mesh2 = trimesh.load(mesh2_path)
    
    target_coordinates = np.array([0, 0, 0])
    translation_vector1 = target_coordinates - mesh1.centroid
    translation_vector2 = target_coordinates - mesh2.centroid
    
    mesh1.apply_transform(trimesh.transformations.translation_matrix(translation_vector1))
    mesh2.apply_transform(trimesh.transformations.translation_matrix(translation_vector2))
    
    assert len(mesh1.vertices) == len(mesh2.vertices), "Meshes must have the same number of vertices!"
    mesh1.fix_normals()
    mesh2.fix_normals()
    
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0], [0, 0, 0])
    mesh1.apply_transform(rotation_matrix)
    mesh2.apply_transform(rotation_matrix)
    
    positional_differences = mesh2.vertices - mesh1.vertices
    velocities = np.linalg.norm(positional_differences, axis=1) / delta_t
    directions = np.sign(positional_differences[:, 2])
    signed_velocities = directions * velocities
    
    verticies = [{
        "index": i,
        "position": mesh1.vertices[i],
        "velocityVector": positional_differences[i]
    } for i in range(len(mesh1.vertices))]
    
    origins = np.array([v["position"] + 1e-4 * (sensor_position - v["position"]) for v in verticies])
    directions = np.array([sensor_position - v["position"] for v in verticies])
    is_visible = ~mesh1.ray.intersects_any(origins, directions)
    
    visible_verticies = [verticies[i] for i in range(len(verticies)) if is_visible[i]]
    positions = np.array([v["position"] for v in visible_verticies])
    velocities = np.array([v["velocityVector"] for v in visible_verticies])
    
    los_vectors = (positions - sensor_position) / np.linalg.norm(positions - sensor_position, axis=1)[:, None]
    relative_velocities = np.einsum('ij,ij->i', velocities, los_vectors) * np.linalg.norm(velocities, axis=1)
    scaled_relative_velocities = relative_velocities / delta_t
    
    return bin_velocities(scaled_relative_velocities).tolist()

if __name__ == "__main__":
    args = parse_args()
    obj_files = sorted(glob.glob(os.path.join(args.input_folder, "*.obj")))
    print(len(obj_files))
    print(obj_files)
    
    fps = 30
    delta_t = 1 / fps
    sensor_position = np.array([0, 1, 3])
    
    frame_pairs = [(obj_files[i], obj_files[i+1], sensor_position, delta_t) for i in range(len(obj_files)-1)]
    
    with Pool(processes=cpu_count()) as pool:
        with alive_bar(len(frame_pairs)) as bar:
            doppler_time_data = []
            for result in pool.imap_unordered(process_frame, frame_pairs):
                doppler_time_data.append(result)
                bar()
    
    display(doppler_time_data)

