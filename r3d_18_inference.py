import torch
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights

vid, _, _ = read_video(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\Assault007_x264.mp4', output_format="TCHW")
vid = vid[:32]  # optionally shorten duration

# Step 1: Initialize model with the best available weights
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
#device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# model.to(device)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(vid).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
label = prediction.argmax().item()
score = prediction[label].item()
category_name = weights.meta["categories"][label]
print(f"{category_name}: {100 * score}%")