from raysect.primitive import Intersect, Subtract, Box, Cylinder, Sphere, Mesh
from raysect.optical import World, Node, Point3D, translate, rotate, d65_white, ConstantSF
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.material import Lambert
from raysect.optical.material.emitter import UniformSurfaceEmitter, Checkerboard
from raysect.optical.library import schott
from raysect.primitive import import_obj
from raysect.optical import World, translate, rotate, ConstantSF, Sellmeier, Dielectric
from raysect.optical.material.emitter import UnityVolumeEmitter
import trimesh

import matplotlib.pyplot as plt

# Box defining the ground plane
ground = Box(lower=Point3D(-50, -0.01, -50), upper=Point3D(50, 0, 50), material=Lambert())

# checker board wall that acts as emitter
emitter = Box(lower=Point3D(-10, -10, 10), upper=Point3D(10, 10, 10.1),
              material=Checkerboard(4, d65_white, d65_white, 0.1, 2.0), transform=rotate(45, 0, 0))

# Sphere
# Note that the sphere must be displaced slightly above the ground plane to prevent numerically issues that could
# cause a light leak at the intersection between the sphere and the ground.


mesh_path = "/home/max/Documents/visualisations/osxTes/person_0.obj"
testMesh = trimesh.load("/home/max/Documents/visualisations/osxTes/person_0.obj")

# Access vertex coordinates
vertices = testMesh.vertices  # Nx3 array of vertex positions

# Separate x, y, and z values
x_values = vertices[:, 0]  # All x-coordinates
y_values = vertices[:, 1]  # All y-coordinates
z_values = vertices[:, 2]  # All z-coordinates

# Print results
print("X values:", x_values)
print("Y values:", y_values)
print("Z values:", z_values)

# Calculate mean values
mean_x = x_values.mean()
mean_y = y_values.mean()
mean_z = z_values.mean()

# Calculate the minimum y value
min_y = y_values.min()
max_y = y_values.max()

print(f"Mean X: {mean_x}, Mean Y: {mean_y}, Mean Z: {mean_z}")
print(f"Minimum Y: {min_y}")
print(f"Maximum Y: {max_y}")

x_transform = -10 * mean_x
y_transform = -10 * min_y
z_transform = -10 * mean_z

human_mesh = import_obj(mesh_path, scaling = 10, transform=translate(x_transform, y_transform,z_transform) * rotate(0,0,180), material=schott("N-BK7"))
#human_mesh.material = UniformSurfaceEmitter(ConstantSF(1.0))


sunlight = Box(
    lower=Point3D(-50, -1.51, -50), upper=Point3D(50, -1.5, 50),
    material=UniformSurfaceEmitter(ConstantSF(100.0)),  # Brightness of the sunlight
    transform=translate(0, 50, 0) * rotate(0, 45, 0)  # Angle and position of sunlight
)

sphere = Sphere(radius=1.5, transform=translate(0, 0.0001, 0), material=schott("N-BK7"))
# processing pipeline (human vision like camera response)
rgb = RGBPipeline2D()

# camera
camera = PinholeCamera(pixels=(512, 512), fov=90, pipelines=[rgb], transform=translate(0,(15*max_y), -15) * rotate(0, 0, 0))

# camera - pixel sampling settings
camera.pixel_samples = 250
camera.min_wavelength = 375.0
camera.max_wavelength = 740.0
camera.spectral_bins = 15
camera.spectral_rays = 1


world = World()


#sphere.parent = world
human_mesh.parent = world
ground.parent = world
#emitter.parent = world
camera.parent = world
sunlight.parent = world


print(testMesh.vertices)
#position = human_mesh.world_transform * Point3D(0, 0, 0)
#print(f"World position of the model: {position}")
#print(f"World position of the mesh: {human_mesh.location}")

plt.ion()
camera.observe()
#rgb_data = rgb.frame
#intensity = 0.2126 * rgb_data[:, :, 0] + 0.7152 * rgb_data[:, :, 1] + 0.0722 * rgb_data[:, :, 2]
#print(intensity)
#plt.imshow(intensity, cmap='hot')  # Use a colormap like 'hot'
#plt.colorbar(label="Intensity")
#plt.title("Intensity Map")
#plt.show()
plt.ioff()
rgb.save("render.png")
rgb.display()
