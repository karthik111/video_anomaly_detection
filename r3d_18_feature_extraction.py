import torch
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights

vid, _, _ = read_video(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\Assault007_x264.mp4', output_format="TCHW")
vid = vid[:1]  # optionally shorten duration
#vid = v

def hook(module, input, output):
    global features
    features = output

def extract_penultimate_features(vid):
    """
       Extracts Feature Tensor 1*512 shape for each video passed to it.

       Args:
           vid (list): T*C*W*H stack of frames within a video

       Returns:
           penultimate_features: Tensor of extracted features torch.Size([1, 512, 1])
       """

    # Step 1: Initialize model with the best available weights
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    model.eval()

    # Step 2: Define a hook function to extract features from the penultimate layer
    #features = None


    #model.layer4[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    # Step 3: Apply inference preprocessing transforms
    preprocess = weights.transforms()
    batch = preprocess(vid).unsqueeze(0)

    # Step 4: Use the model to extract features from the penultimate layer
    with torch.no_grad():
        _ = model(batch)

    # Step 5: Get the features and print their shape
    penultimate_features = features.squeeze(2).squeeze(2)
    print("Features shape:", penultimate_features.shape)

    return penultimate_features

penultimate_features = extract_penultimate_features(vid)

#
# # Step 1: Initialize model with the best available weights
# weights = R3D_18_Weights.DEFAULT
# model = r3d_18(weights=weights)
# #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# # model.to(device)
# model.eval()
#
# # Step 2: Define a hook function to extract features from the penultimate layer
# features = None
# def hook(module, input, output):
#     global features
#     features = output
#
# model.layer4[-1].register_forward_hook(hook)
#
# # Step 3: Apply inference preprocessing transforms
# preprocess = weights.transforms()
# batch = preprocess(vid).unsqueeze(0)
#
# # Step 4: Use the model to extract features from the penultimate layer
# with torch.no_grad():
#     _ = model(batch)
#
# # Step 5: Get the features and print their shape
# penultimate_features = features.squeeze(2).squeeze(2)
# print("Features shape:", penultimate_features.shape)
