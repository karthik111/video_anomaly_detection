import torch
import torchvision.models.video.resnet as models
import torchvision.transforms as transforms

from PIL import Image
import torchvision.io as io


def load_video_frames(video_path):
    # Load the video frames using torchvision's `read_video` method
    video_frames, audio, info = io.read_video(video_path, pts_unit='sec')

    # Convert the video frames from torch.Tensor to PIL.Image format
    frames_list = []
    for frame in video_frames:
        # Convert from (C, H, W) format to (H, W, C) format
        #frame = frame.permute(1, 2, 0)
        # Convert from torch.Tensor to PIL.Image format
        frame = Image.fromarray(frame.numpy(), 'RGB')
        # Append the frame to the list of frames
        frames_list.append(frame)

    return frames_list


# Load the pre-trained Inception 3D model
inception = models.mc3_18(pretrained=True)

# Set the model to evaluation mode
inception.eval()

# Define the preprocessing transform for the input frames
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and preprocess the video frames
frames = load_video_frames(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\archery.mp4')  # TODO: Implement video loading function
preprocessed_frames = torch.stack([transform(frame) for frame in frames])

# Pass the frames through the model and extract features
with torch.no_grad():
    features = inception(preprocessed_frames)

# Aggregate features (e.g., by averaging)
aggregated_features = features.mean(dim=2)

# Classify the video using a linear layer
linear = torch.nn.Linear(aggregated_features.shape[1], 400)  # 400 action classes
scores = linear(aggregated_features)

# Get the predicted action class
predicted_class = scores.argmax().item()
