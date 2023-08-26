import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import zipfile

import logging
from datetime import datetime

current_day = datetime.now().strftime("%d-%m-%y")

logging.basicConfig(filename=f'log_file_{current_day}.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class MultiZipVideoDataset(Dataset):
    def __init__(self, zip_file_paths, transform=None):
        self.zip_file_paths = zip_file_paths
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        print(self.classes, self.class_to_idx)
        self.samples = self._make_dataset()

    def read_zip_video(self, p):
        import decord
        import io

        video = p.read_bytes()
        file_obj = io.BytesIO(video)
        vr = decord.VideoReader(file_obj)
        frames = vr.get_batch(range(0, len(vr) - 1, 5))
        print(f"No of frames: {len(vr)}")
        logging.info(f"No of frames: {len(vr)}")
        return frames.asnumpy()

    def __getitem__(self, index):
        video_path, class_idx = self.samples[index]
        print(f"In getitem: {class_idx}, {str(video_path)}")
        logging.info(f"In getitem: {class_idx}, {str(video_path)}")
        video = self.read_zip_video(video_path)
        if self.transform:
            video = self.transform(video)
        return video, class_idx

    def __len__(self):
        return len(self.samples)

    # def _find_classes(self):
    #     classes = []
    #     class_to_idx = {}
    #     for zip_file_path in self.zip_file_paths:
    #         with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
    #             for zip_info in zip_file.infolist():
    #                 if zip_info.is_dir():
    #                     class_name = os.path.join(zip_file_path, zip_info.filename)
    #                     if class_name not in classes:
    #                         classes.append(class_name)
    #                         class_to_idx[class_name] = len(classes) - 1
    #     return classes, class_to_idx

    def _find_classes(self):
        classes = []
        class_to_idx = {}
        for zip_file_path in zip_file_paths:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                for zip_info in zip_file.infolist():
                    if zip_info.is_dir() and zip_info.filename.count('/') >= 1:
                        class_name = zip_info.filename[zip_info.filename.find('/') + 1:zip_info.filename.rfind('/')]
                        if class_name == '':
                            class_name = 'Normal'
                        if class_name not in classes:
                            classes.append(class_name)
                            class_to_idx[class_name] = len(classes) - 1
        return classes, class_to_idx

    # def _make_dataset(self):
    #     samples = []
    #     for zip_file_path in self.zip_file_paths:
    #         with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
    #             for zip_info in zip_file.infolist():
    #                 if not zip_info.is_dir():
    #                     class_name = os.path.join(zip_file_path, os.path.dirname(zip_info.filename))
    #                     print("Name: " + class_name)
    #                     print(self.class_to_idx)
    #                     class_idx = self.class_to_idx[class_name]
    #                     video_path = f'zip://{zip_file_path}#{zip_info.filename}'
    #                     samples.append((video_path, class_idx))
    #     return samples

    def _make_dataset(self):
        samples = []
        for zip_file_path in self.zip_file_paths:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                for zip_info in zip_file.infolist():
                    if not zip_info.is_dir():
                        #print(zip_info.filename)
                        if (zip_info.filename).find('Normal') > 1:
                            class_name = 'Normal'
                        else:
                            class_name = zip_info.filename[zip_info.filename.find('/')+1:zip_info.filename.rfind('/')]
                        #print(class_name)
                        class_idx = self.class_to_idx[class_name]
                        video_path = zipfile.Path(zip_file_path, at=zip_info.filename)
                        samples.append((video_path, class_idx))
        return samples

import os

zip_file_paths = [os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-1.zip'),
                  os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-2.zip'),
                os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Training-Normal-Videos-Part-1.zip'),
                   ]


zip_file_paths_train1 = [os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-1.zip'),
                  os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-2.zip'),
                os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Training-Normal-Videos-Part-1.zip'),
                   ]


zip_file_paths_train2 = [os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-3.zip'),
                  os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-4.zip'),
                os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Training-Normal-Videos-Part-2.zip'),
                        ]

zip_file_paths_test = [os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Testing_Normal_Videos.zip')]

zip_file_paths = zip_file_paths_train1 + zip_file_paths_train2

dataset = MultiZipVideoDataset(zip_file_paths=zip_file_paths, transform=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
#
# import torch
# import torchvision
# from torchvision.datasets.video_utils import VideoClips
#
# video_clips = VideoClips(file_contents, clip_length_in_frames=16, frames_between_clips=1)
#
# for zip_file_path in zip_file_paths:
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
#         for zip_info in zip_file.infolist():
#             if zip_info.is_dir() and zip_info.filename.count('/') > 1:
#                 class_name = zip_info.filename[zip_info.filename.find('/')+1:zip_info.filename.rfind('/')]
#                 print(class_name)
#
# file_to_open = 'Anomaly-Videos-Part-1/Arson/Arson031_x264.mp4'
#
# classes = []
# class_to_idx = {}
# for zip_file_path in zip_file_paths:
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
#         for zip_info in zip_file.infolist():
#             if zip_info.is_dir() and zip_info.filename.count('/') > 1:
#                 class_name = os.path.join(zip_file_path, zip_info.filename)
#                 class_name = zip_info.filename[zip_info.filename.find('/')+1:zip_info.filename.rfind('/')]
#                 if class_name not in classes:
#                     classes.append(class_name)
#                     class_to_idx[class_name] = len(classes) - 1
#
# zip_file_paths = [os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-1.zip'),
#                   os.path.normpath(r'C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-2.zip'),
#                    ]
#
# samples = []
# for zip_file_path in zip_file_paths:
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
#         for zip_info in zip_file.infolist():
#             if not zip_info.is_dir():
#                 # class_name = os.path.join(zip_file_path, os.path.dirname(zip_info.filename))
#                 #print("className: " + zip_info.filename)
#                 class_name = zip_info.filename[zip_info.filename.find('/')+1:zip_info.filename.rfind('/')]
#                 #print("Name: " + class_name)
#                 print(class_to_idx)
#                 class_idx = class_to_idx[class_name]
#                 #video_path = f'zip://{zip_file_path}#{zip_info.filename}'
#                 video_path = zipfile.Path(zip_file_path, at = zip_info.filename)
#                 samples.append((video_path, class_idx))
#
#
# s = f"zip://C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-1.zip" #Anomaly-Videos-Part-1/Abuse/Abuse041_x264.mp4'
# s = f"zip://" + os.path.normpath(r"C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Anomaly-Videos-Part-1.zip") + "#" + r"Anomaly-Videos-Part-1/Abuse/Abuse041_x264.mp4"
# os.scandir(s)