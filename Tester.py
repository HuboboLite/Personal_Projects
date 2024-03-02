from roboflow import Roboflow
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import matplotlib.image as mpimg
from roboflow import Roboflow
rf = Roboflow(api_key="LTbPR7zAynPXC51hOJIW")
project = rf.workspace().project("braintumordetection-hxkiz")
model = project.version(2).model

# infer on a local image
# print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/meningioma_tumor/image(4).jpg", confidence=40, overlap=30).save("prediction.jpg")

img = mpimg.imread('prediction.jpg')
imgplot = plt.imshow(img)
plt.show()
