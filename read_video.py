import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchvision
from torchvision.datasets.video_utils import VideoClips
import zipfile
import os

zip_file_paths = [os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-1.zip')] #, "C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Testing_Normal_Videos.zip"]
file_to_open = 'Anomaly-Videos-Part-1/Arson/Arson031_x264.mp4'
file_to_open = r'C:\Users\karthik.venkat\PycharmProjects\video_anomaly_detection\Anomaly-Videos-Part-1\Arson\Arson031_x264.mp4'
#file_to_open = 'Arson031_x264.mp4'

# open the zip file
with zipfile.ZipFile(zip_file_paths[0], "r") as zip_file:
    # read the contents of the file you want to open
    with zip_file.open(file_to_open) as file:
        file_contents = file.read()

video_clips = VideoClips([file_to_open], clip_length_in_frames=16, frames_between_clips=1)

stream = "video"
video = torchvision.io.VideoReader(file_to_open, stream)
video.get_metadata()