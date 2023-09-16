import torch
from torchvision.models.video import r3d_18, R3D_18_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def get_score_category(model, t):

    t = t.to(device)
    print(model.fc(t.squeeze()).shape)

    prediction = model.fc(t.squeeze()).softmax(0)
    label = prediction.argmax().item()
    score = prediction[label].item()
    category_name = weights.meta["categories"][label]
    print(f"{category_name}: {100 * score}%")

def get_model():
    # Step 1: Initialize model with the best available weights
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.to(get_device())
    model.eval()
    return model

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def main():
    print("Calling get_score_category...")
    t = torch.load(
        r'C:\Users\karthik.venkat\PycharmProjects\video_anomaly_detection\processed\data\explosion\explosion001_x264.pt')
    model = get_model()
    get_score_category(model, t)
    # You can put your main code here

# Check if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    main()