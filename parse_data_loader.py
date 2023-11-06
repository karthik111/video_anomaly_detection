# Uses a Data Loader to read the videos and class names (labels)
# then calls a feature processing network like R3D to extract the features
# and then calls save_file to save the video as a .pt file
# to use this class, find and replace all occurrences of folder_array, with the source location of your videos
# then invoke the parse_video_file_dataset() function

import Data_Loader
import VideoDataset
import torch
import os
import logging
import r3d_18_feature_extraction
from datetime import datetime

current_day = datetime.now().strftime("%d-%m-%y")

logging.basicConfig(filename=f'Parse_data_set_log_file_{current_day}.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

videos_folder = 'test_videos_variant_2_2023-11-04'

def save_file(features, video_name, video_class):
    # folder_array = [os.getcwd(), 'processed', 'data', video_class.lower()]
    # folder_array = [os.getcwd(), 'test_videos', 'processed', video_class.lower()]
    # folder_array = [os.getcwd(), 'test_videos_variant_1', 'processed', video_class.lower()]
    folder_array = [os.getcwd(), videos_folder, 'processed', video_class.lower()]

    folder_path = os.path.join(*folder_array)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = os.path.join(folder_path, video_name.lower() + '.pt')
    torch.save(features, file_name)
    print(f"Saved feature to  folder: {file_name}")
    logging.info(f"Saved feature to  folder: {file_name}")

def is_processed(video_name, video_class):
    folder_array = [os.getcwd(), 'processed', 'data', video_class.lower(), f'{video_name.lower()}.pt']
    folder_array = [os.getcwd(), 'test_videos', 'processed', video_class.lower(), f'{video_name.lower()}.pt']
    folder_array = [os.getcwd(), 'test_videos_variant_1', 'processed', video_class.lower(), f'{video_name.lower()}.pt']
    folder_array = [os.getcwd(), videos_folder, 'processed', video_class.lower(), f'{video_name.lower()}.pt']

    folder_path = os.path.join(*folder_array)
    #print(f'In is_processed: {folder_path} : {os.path.exists(folder_path)}')
    if os.path.exists(folder_path):
        return True
    else:
        return False

def parse_multi_zip_dataset():
    classes = Data_Loader.dataset.classes
    samples = Data_Loader.dataset.samples

    n = len(Data_Loader.dataset.samples)

    for i in range(n):
        video_name = samples[i][0].stem
        video_class = samples[i][0].filename.parent.name
        if not is_processed(video_name, video_class):
            try:
                video, class_idx = Data_Loader.dataset.__getitem__(i)
                input_vid = torch.tensor(video.transpose(0,3,1,2))
                features = r3d_18_feature_extraction.extract_penultimate_features(input_vid)
                save_file(features, video_name, video_class)
            except Exception as e:
                logging.error(f"An error occurred with file: {video_name}", exc_info=True)
        else:
            print(f"Feature processed earlier: {video_name}")
            logging.info(f"Feature processed earlier: {video_name}")

    print(n)

def parse_video_file_dataset():
    # Create an instance of the custom dataset
    #video_folder = os.path.normpath(r'.\\test_videos')
    #video_folder = os.path.normpath(r'.\\test_videos_variant_1')
    #video_folder = os.path.normpath(f'.\\{videos_folder}')

    dataset = VideoDataset.VideoDataset(root_dir=videos_folder, transform=None)

    classes = dataset.classes
    samples = dataset.video_files

    n = len(samples)

    for i in range(n):
        video_name = samples[i]
        video_class = classes[i]
        if not is_processed(video_name, video_class):
            try:
                video = dataset.__getitem__(i)
                input_vid = torch.tensor(video.transpose(0, 3, 1, 2))
                features = r3d_18_feature_extraction.extract_penultimate_features(input_vid)
                save_file(features, video_name, video_class)
            except Exception as e:
                print(e)
                logging.error(f"An error occurred with file: {video_name}", exc_info=True)
        else:
            print(f"Feature processed earlier: {video_name}")
            logging.info(f"Feature processed earlier: {video_name}")

    print(n)
