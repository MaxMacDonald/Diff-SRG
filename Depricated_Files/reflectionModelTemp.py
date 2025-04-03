from raysect.primitive import Intersect, Subtract, Box, Cylinder, Sphere
from raysect.optical import World, Node, Point3D, translate, rotate, d65_white, ConstantSF
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.material import Lambert
from raysect.optical.material.emitter import UniformSurfaceEmitter, Checkerboard
from raysect.optical.library import schott

import matplotlib.pyplot as plt


# Box defining the ground plane
ground = Box(lower=Point3D(-50, -1.51, -50), upper=Point3D(50, -1.5, 50), material=Lambert())

# checker board wall that acts as emitter
emitter = Box(lower=Point3D(-10, -10, 10), upper=Point3D(10, 10, 10.1),
              material=Checkerboard(4, d65_white, d65_white, 0.1, 2.0), transform=rotate(45, 0, 0))

# Sphere
# Note that the sphere must be displaced slightly above the ground plane to prevent numerically issues that could
# cause a light leak at the intersection between the sphere and the ground.
sphere = Sphere(radius=1.5, transform=translate(0, 0.0001, 0), material=schott("N-BK7"))


# processing pipeline (human vision like camera response)
rgb = RGBPipeline2D()

# camera
camera = PinholeCamera(pixels=(512, 512), fov=45, pipelines=[rgb], transform=translate(0, 10, -10) * rotate(0, -45, 0))

# camera - pixel sampling settings
camera.pixel_samples = 250
camera.min_wavelength = 375.0
camera.max_wavelength = 740.0
camera.spectral_bins = 15
camera.spectral_rays = 1


world = World()

sphere.parent = world
ground.parent = world
emitter.parent = world
camera.parent = world



plt.ion()
camera.observe()

plt.ioff()
rgb.save("render.png")
rgb.display()