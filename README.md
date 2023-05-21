# video_anomaly_detection

07-May-2023
- Added r3d_18_inference.py to perform inference using Inception 3D model on provided video file
- Added load_video_from_dataset.py to obtain a randomly provided video from Data_Loader.py

To do:
- Extract feature from interim layer
- Read up the Inception architecture

08-May-2023
- Coded the feature extractor layer

To do:
- Build a simple classification model based on the feature extractor of a few video clips. 
- Obtain anomaly detection accuracy. 
- Build a UMAP dimensionality reducer for different anomaly types
- Compare current feature extractor to embeddings from a vision transformer

20-May-2023
- Built a CIFAR 10 classifier with custom images to be used for to process the interim layer of video feature extractor