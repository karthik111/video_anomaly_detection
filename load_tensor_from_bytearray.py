import decord
import torch
import os, io

from Data_Loader import MultiZipVideoDataset

from decord import VideoReader

zip_file_paths = [os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-1.zip'),
                  os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-2.zip'),
                   ]

dataset = MultiZipVideoDataset(zip_file_paths=zip_file_paths, transform=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

samples = dataset.samples
sample = samples[54]
video = sample[0].read_bytes()
video_str = video
file_obj = io.BytesIO(video_str)

decord.bridge.set_bridge('torch')
vr = decord.VideoReader(file_obj)
frames = vr.get_batch(range(0, len(vr) - 1, 5))
print(frames.shape)
print(" No. of frames: " + str(frames.shape[0]))

from matplotlib import pyplot as plt
plt.imshow(frames[150,])
plt.show()