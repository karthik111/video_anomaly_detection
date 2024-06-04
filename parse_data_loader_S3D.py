import Data_Loader
import traceback
from omegaconf import OmegaConf
from tqdm import tqdm
from omegaconf import OmegaConf
from tqdm import tqdm


from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check

from datetime import datetime
import os
import logging
import parse_data_loader
import Data_Loader
import VideoDataset
import torch
import sys
import logging
import r3d_18_feature_extraction
from datetime import datetime

videos_folder = 'test_videos_variant_2_2023-11-04'
class ParseDataLoader():
    def __init__(self, args) -> None:
       print('In')

def save_file(features, video_name, video_class):
    # folder_array = [os.getcwd(), 'processed', 'data', video_class.lower()]
    # folder_array = [os.getcwd(), 'test_videos', 'processed', video_class.lower()]
    # folder_array = [os.getcwd(), 'test_videos_variant_1', 'processed', video_class.lower()]
    #folder_array = [os.getcwd(), videos_folder, 'processed', video_class.lower()]
    folder_array = [os.getcwd(), videos_folder, 'processed_s3d', video_class.lower()]

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
    folder_array = [os.getcwd(), videos_folder, 'processed_s3d', video_class.lower(), f'{video_name.lower()}.pt']

    folder_path = os.path.join(*folder_array)
    #print(f'In is_processed: {folder_path} : {os.path.exists(folder_path)}')
    if os.path.exists(folder_path):
        return True
    else:
        return False
def parse_multi_zip_dataset(dataset, dataloader, extractor):


    classes = dataset.classes
    samples = dataset.samples

    n = len(dataset.samples)

    for i in range(n):
        video_name = samples[i][0].stem
        video_class = samples[i][0].filename.parent.name
        if not is_processed(video_name, video_class):
            try:
                video, class_idx = dataset.__getitem__(i)
                input_vid = torch.tensor(video.transpose(0,3,1,2))
                print(video_name, video.shape)
                features = extractor.extract(video)
                print("Extracted features: ", type(features))
                #import numpy as np
                #features = np.mean(features[None], axis=0)
                #return feature
                #features = r3d_18_feature_extraction.extract_penultimate_features(input_vid)
                save_file(features[None], video_name, video_class)
            except Exception as e:
                traceback.print_exc()
                logging.error(f"An error occurred with file: {video_name}", exc_info=True)
        else:
            print(f"Feature processed earlier: {video_name}")
            logging.info(f"Feature processed earlier: {video_name}")

    print(n)

def main(args_cli):
    dataset, dataloader = Data_Loader.get_dataset_and_loader()

    from models.s3d.extract_s3d import ExtractS3D as Extractor

    args = args_cli
    extractor = Extractor(args)

    current_day = datetime.now().strftime("%d-%m-%y")

    logging.basicConfig(filename=f'Parse_data_set_log_file_{current_day}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    videos_folder = 'test_videos_variant_2_2023-11-04'

    parse_multi_zip_dataset(dataset, dataloader, extractor)

if __name__ == '__main__':
    args_cli = OmegaConf.from_cli()
    main(args_cli)
