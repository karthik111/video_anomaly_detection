import Data_Loader

import os


def create_or_move_file(class_name, file_path):
    folder_path = class_name.lower()

    # Check if the folder exists
    if os.path.exists(folder_path):
        # Folder already exists, move the file to it
        new_file_path = os.path.join(folder_path, os.path.basename(file_path))
        os.rename(file_path, new_file_path)
        print(f"Moved file to existing folder: {new_file_path}")
    else:
        # Folder doesn't exist, create a new one and move the file
        os.makedirs(folder_path)
        new_file_path = os.path.join(folder_path, os.path.basename(file_path))
        os.rename(file_path, new_file_path)
        print(f"Created new folder and moved file: {new_file_path}")

classes = Data_Loader.dataset.classes
samples = Data_Loader.dataset.samples

n = len(Data_Loader.dataset.samples)

for i in range(n):
    video_name = samples[i][0].stem
    video_class = samples[i][0].filename.parent.name
    video, class_idx = Data_Loader.dataset.__getitem__(i)
    # Now save this video to a tensor and then to a file

print(n)