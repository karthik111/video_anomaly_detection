import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling3D
from tensorflow.keras import backend as K
import numpy as np
import cv2

# Load the Inception 3D model without the top layers
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling3D()(x)

# Add a fully-connected layer with 1024 units
x = Dense(1024, activation='relu')(x)

# Add a softmax layer for classification
predictions = Dense(num_classes, activation='softmax')(x)
#import
# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Load a video file
cap = cv2.VideoCapture('path/to/video.mp4')

# Initialize an empty feature vector
features = []

# Loop through each frame of the video
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Break the loop if the end of the video is reached
    if not ret:
        break

    # Preprocess the frame
    img = image.img_to_array(frame)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract the features for the frame using the Inception 3D model
    features_for_frame = model.predict(img)

    # Add the features to the feature vector
    features.append(features_for_frame)

# Convert the feature vector to a numpy array
features = np.array(features)

# Save the features to a file
np.save('path/to/features.npy', features)
