import matplotlib.pyplot as plt
import PIL
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image

img = Image.open(r'C:\\Users\karthik.venkat\PycharmProjects\video_anomaly_detection\data\cifar10\mafia.jpg')

img = TF.resize(img, (32,32))

arr = np.array(img)

plt.imshow(arr)
plt.show()