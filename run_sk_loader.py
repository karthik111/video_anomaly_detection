# For a given folder path where the processed features are present, this class sets up the X_train, y_train, X_val and Y_val.
# Update the folder_array to use this class

import os
from torchvision import datasets
import sklearn.datasets
import torch
import numpy as np
import pandas as pd

import load_train_list


def pt_loader(path):
    sample = torch.load(path)
    return sample


folder_array = [os.getcwd(), 'processed', 'data']
folder_path = os.path.join(*folder_array)

dataset = datasets.DatasetFolder(
    root=folder_path,
    loader=pt_loader,
    extensions=['.pt']
)

sk_data = sklearn.datasets.load_files(container_path=f'{folder_path}')
df = pd.DataFrame(sk_data.filenames)
df1 = df[0].apply(lambda x: x[x.rfind('\\') + 1:-3].capitalize())

df1 = df1.str.replace('_', '').str.lower()

# get the train and test data set from the data set specification
df_train, df_test = load_train_list.get_train_test_data()

df_train[1] = df_train[1].str.replace('_', '').str.lower()
df_test[1] = df_test[1].str.replace('_', '').str.lower()

indices_train = df1[df1.isin(df_train[1])].index.to_list()
indices_test = df1[df1.isin(df_test[1])].index.to_list()

# df_test is the list of test video names load from the published test list in
# df1 is the full list of .pt file names which is present in the specified folder

labels = sk_data.target_names

classes_dict = {value: index for index, value in enumerate(sk_data.target_names)}
dict_idx_to_class = {index: value for index, value in enumerate(sk_data.target_names)}

##
num_classes = dict(zip(sk_data.target_names, np.zeros(len(sk_data.target_names))))

idx_to_classes = {value: key for key, value in dataset.class_to_idx.items()}

for i in range(len(sk_data.target)):
    #num_classes[ idx_to_classes[ samples[i][1] ] ] += 1
    num_classes[ dict_idx_to_class[sk_data.target[i] ] ] += 1
##

target_label_list = [0 if (value==12 or value==13 or value==14) else 1 for value in sk_data.target]

X_train = [feature_array_list[i] for i in indices_train]
X_val = [feature_array_list[i] for i in indices_test]

y_train = [target_label_list[i] for i in indices_train]
y_val = [target_label_list[i] for i in indices_test]
