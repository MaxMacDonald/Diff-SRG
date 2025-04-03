import trimesh
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import argparse
import glob
import os
from alive_progress import alive_bar

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='input_images')
    args = parser.parse_args()
    return args


def display(values):
    print(f"Shape of input values is {np.shape(values)}")
    # Step 2: Reshape to (32, 250)
    doppler_data = np.array(values).T

    # Create an intensity map for the Doppler data
    plt.figure(figsize=(10, 6))
    plt.imshow(doppler_data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Doppler Intensity Map')
    plt.xlabel('Time/Frequency Bins')
    plt.ylabel('Sensor Channels')
    plt.tight_layout()
    plt.show()
    plt.savefig("plot.png")
    np.save("synthDopplerData.npy",doppler_data)

def bin_velocities(velocities):
    # 32 bins between -2 and +2 m/s 
    # total number of velocites in each bin, then normalise these values
    # return an array of size 32 of the normalised number of velocities in each bin.

    # Define bin edges
    bins = np.linspace(-2, 2, 33)  # 33 edges for 32 bins

    # Compute histogram (counts per bin)
    histogram, _ = np.histogram(velocities, bins=bins)

    # Normalize the histogram (convert counts to probabilities)
    normalized_histogram = histogram / np.sum(histogram) if np.sum(histogram) > 0 else histogram

    return normalized_histogram


args = parse_args()
obj_files = sorted(glob.glob(os.path.join(args.input_folder, "*.obj")))  # Get all .obj files
print(len(obj_files))
print(obj_files)

doppler_time_data = []
with alive_bar(len(obj_files)-1) as bar:
    for i in range(len(obj_files)-1):
        
        # Load in the meshes
        mesh1_path = obj_files[i] # Frame 1
        mesh2_path = obj_files[i+1]  # Frame 2
        mesh1 = trimesh.load(mesh1_path)
        mesh2 = trimesh.load(mesh2_path)

        # Specify the target coordinates (x, y, z)
        target_coordinates = np.array([0, 0, 0])

        # Compute the current centroid of the mesh
        current_centroid = mesh1.centroid

        # Compute the translation vector needed to move the centroid to the target position
        translation_vector1 = target_coordinates - mesh1.centroid
        translation_vector2 = target_coordinates - mesh2.centroid


        # Create a translation transformation matrix
        translation_matrix1 = trimesh.transformations.translation_matrix(translation_vector1)
        translation_matrix2 = trimesh.transformations.translation_matrix(translation_vector2)

        # Apply the translation to the mesh
        mesh1.apply_transform(translation_matrix1)
        mesh2.apply_transform(translation_matrix2)

        # Ensure both meshes have the same number of vertices
        assert len(mesh1.vertices) == len(mesh2.vertices), "Meshes must have the same number of vertices!"

        # Fix normals (optional, for consistent rendering)
        mesh1.fix_normals()
        mesh2.fix_normals()

        # Rotate the mesh to align it
        # Example: Rotate around X-axis by 90 degrees to make the body face the camera
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.radians(180),  # Rotate 90 degrees
            direction=[1, 0, 0],  # Rotate around the X-axis
            point=[0, 0, 0]  # Center of rotation
        )
        # Apply rotation to the mesh
        mesh1.apply_transform(rotation_matrix)
        mesh2.apply_transform(rotation_matrix)

        verticies = []
        visible_verticies = []

        # Calculate the positional differences between corresponding vertices
        positional_differences = mesh2.vertices - mesh1.vertices

        # Assume FPS of 30, so time between frames is 1/30 seconds
        fps = 30
        delta_t = 1 / fps

        # Calculate velocity (magnitude of displacement divided by delta_t)
        velocities = np.linalg.norm(positional_differences, axis=1) / delta_t



        # Determine direction: Positive or Negative based on displacement along a specific axis (e.g., Z-axis)
        directions = np.sign(positional_differences[:, 2])  # Assuming Z-axis as vertical

        # Combine direction and velocity
        signed_velocities = directions * velocities

        sensor_position = np.array([0, 1, 3])


        for i in range (len(mesh1.vertices)):
            d = {
                "index": i,
                "position": mesh1.vertices[i],
                "visible": False,
                "velocityVector": positional_differences[i],
                "relativeV": 0,
                "realRelativeV":0
            }
            
            verticies.append(d)


        # Perform raycasting for visibility
        visible_indices = []
        for v in verticies:
            vertex = v["position"]
            ray_direction = sensor_position - vertex
            ray_origin = vertex + 1e-4 * ray_direction
            is_visible = not mesh1.ray.intersects_any([ray_origin], [ray_direction])
            if is_visible:
                v["visible"] = True
                

        fps = 30
        delta_t = 1 / fps
        # Calculate velocity relative to the sensor
        relative_velocities = []
        for i in range(len(verticies)):
            if verticies[i]["visible"]:
                current_v = verticies[i]
                vertex = current_v["position"]
                velocity = current_v["velocityVector"]
                line_of_sight = (vertex - sensor_position) / np.linalg.norm(vertex - sensor_position)
                # How much of the total velocity is facing the sensor.
                relative_velocity = np.dot(velocity, line_of_sight) * np.linalg.norm(velocity)
                # Divide by frame time to get real world velocity of point
                scaled_relative_velocity = relative_velocity/delta_t
                verticies[i]["relativeV"] = relative_velocity
                verticies[i]["realRelativeV"] = scaled_relative_velocity
                visible_verticies.append(verticies[i])

        # Calculate magnitude of each vertex velocity relative to sensor location


        visible_velocities = [item['realRelativeV'] for item in visible_verticies]
        histo = bin_velocities(visible_velocities)
        doppler_time_data.append(histo.tolist())
        bar()
    
display(doppler_time_data)




