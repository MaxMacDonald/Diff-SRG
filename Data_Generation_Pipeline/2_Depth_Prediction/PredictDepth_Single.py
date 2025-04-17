
from infer import InferenceHelper
from PIL import Image
import matplotlib.pyplot as plt


infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a single pillow image
img = Image.open("/home/max/Documents/visualisations/human_0/frame_002.png")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

plt.imshow(predicted_depth[0][0], cmap='plasma')
plt.show()