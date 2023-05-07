import matplotlib.pyplot as plt
import PIL
import numpy as np

from PIL import Image

img = Image.open("C:\\Users\karthik.venkat\Downloads\mafia.JPG")
arr = np.array(img)

plt.imshow(arr)
plt.show()