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

24-Jun
-  Video resnet model uses 3D convolutions with variable length output in the penultimate layer
-  This doesn't allow the penultimate Layer 4 to be used for downstream learning tasks that need a fixed length feature input
- used the torch summary library to review output model sizes based on given video input - very useful
- Reverted to last layer of global average pool layer which returns (512) size output
- Questions
  - Will attention be needed given that temporal features are detected in the Video res net model (R3D)