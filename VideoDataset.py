import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime
import logging

current_day = datetime.now().strftime("%d-%m-%y")

logging.basicConfig(filename=f'video_dataset_log_file_{current_day}.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.video_files = [f for f in os.listdir(root_dir) if f.endswith('.mp4')]
        self.transform = transform
        self.classes = [self._find_class(v) for v in self.video_files]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        import decord
        vr = decord.VideoReader(video_path)
        frames = vr.get_batch(range(0, len(vr) - 1, 5))
        print(f"No of frames: {len(vr)}")
        logging.info(f"No of frames: {len(vr)}")
        return frames.asnumpy()


    def _find_class(self, input_string):
        if not 'anomaly' in input_string:
            return 'Normal'
        else:
            class_names = ['Abuse', 'Assault', 'Arrest', 'Arson', 'Burglary', 'Fighting', 'Explosion', 'Robbery', 'RoadAccidents', 'Shooting', 'Vandalism', 'Stealing', 'Shoplifting']
            for substring in class_names:
                if substring in input_string:
                    return substring


# Example of a transform to preprocess each frame (you can customize this)
transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an instance of the custom dataset
#video_folder = os.path.normpath(r'.\\test_videos')
#dataset = VideoDataset(root_dir=video_folder, transform=None)

# Create a data loader
#dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate through the dataloader
# for batch in dataloader:
#     # 'batch' contains a list of tensors, each representing a video's frames
#     # You can process the frames here, pass them through a model, etc.
#     print("Batch size:", len(batch))
#     print("Frame tensor shape:", batch[0].shape)  # Shape of the first video's frames
#     break  # Stop after processing the first batch for demonstration

