import trimesh
import numpy as np

# Load the 3D model
mesh_path = "/home/max/Documents/visualisations/osxTes/person_0.obj"  # Replace with your 3D model file path
mesh = trimesh.load(mesh_path)

# Define Radar Emitter Position (e.g., point source at origin)
emitter_position = np.array([0, 0, 0])  # Emitter at the origin

# Define Receiver Direction (backscatter)
receiver_direction = np.array([0, 0, 1])  # Receiver along +z-axis

# Function to calculate RCS for each vertex
def calculate_vertex_rcs(mesh, emitter_position, receiver_direction):
    # Normalize the receiver direction
    receiver_direction = receiver_direction / np.linalg.norm(receiver_direction)
    
    # Initialize an array to store RCS values for each vertex
    vertex_rcs = np.zeros(len(mesh.vertices))
    
    # Loop through all triangles in the mesh
    for face in mesh.faces:
        # Get the vertices of the triangle
        vertices = mesh.vertices[face]
        
        # Calculate triangle normal
        edge1 = vertices[1] - vertices[0]
        edge2 = vertices[2] - vertices[0]
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal

        # Calculate centroid of the triangle
        centroid = np.mean(vertices, axis=0)
        
        # Vector from emitter to centroid
        emitter_to_centroid = centroid - emitter_position
        emitter_to_centroid = emitter_to_centroid / np.linalg.norm(emitter_to_centroid)
        
        # Check if the triangle faces the emitter
        if np.dot(normal, emitter_to_centroid) > 0:  # Only consider illuminated facets
            # Calculate RCS contribution for the triangle
            # Geometric optics backscatter: |n . d|^2 * Area
            projected_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            rcs_contribution = (np.dot(normal, receiver_direction) ** 2) * projected_area
            
            # Distribute the RCS contribution equally to the triangle's vertices
            for vertex_index in face:
                vertex_rcs[vertex_index] += rcs_contribution / 3  # Equal share to each vertex
    
    return vertex_rcs

# Calculate RCS for each vertex
vertex_rcs = calculate_vertex_rcs(mesh, emitter_position, receiver_direction)

# Print the RCS values for the first few vertices
for i, rcs in enumerate(vertex_rcs[:10]):
    print(f"Vertex {i}: RCS = {rcs:.6f} m^2")

