import Data_Loader
import os
import io
from Data_Loader import MultiZipVideoDataset
import torch

samples = []

if samples == []:
    zip_file_paths = [os.path.normpath(
        r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-1.zip'),
                    os.path.normpath(
                        r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-2.zip'),
                    ]

    zip_file_paths = [os.path.normpath(
        r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-3.zip'),
                      os.path.normpath(
                          r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-4.zip'),
                      os.path.normpath(
                          r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Training-Normal-Videos-Part-2.zip'),
                      os.path.normpath(
                          r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Testing_Normal_Videos.zip'),
                      ]

    dataset = MultiZipVideoDataset(zip_file_paths=zip_file_paths, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    samples = dataset.samples

sample = samples[54]
video = sample[0].read_bytes()
file_name = open(sample[0].name, 'wb')
file_name.write(video)
print("Saved file: " + sample[0].name)
