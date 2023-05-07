import numpy as np
import tensorflow as tf
from tensorflow import keras

dataset = keras.utils.image_dataset_from_directory("C:\\Users\\karthik.venkat\\OneDrive - Accenture\\work\\Learning\\Masters\\Project\\BMG\\11august", batch_size=64, image_size=(200, 200))x - 