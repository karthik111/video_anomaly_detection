# For a set of .pt feature files, runs the R3d model and obtains the final 400 element scores and the predicted classs
# run naive_classifier_all_files.py before this to obtain the list of the videos which which passed or failed
# or use the generated txt files where these are stored

import pandas as pd
from pandas import DataFrame as df
import run_model_with_weights
import torch

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# pd.DataFrame(file_names).to_csv('incorrect_class_videos_full_list.txt')
# pd.DataFrame(file_names_pass).to_csv('correct_class_videos_full_list.txt', header=False)

df_pass = pd.read_csv('correct_class_videos_full_list.txt', header=None)
df_fail = pd.read_csv('incorrect_class_videos_full_list.txt', header=None)

df_fail['predicted_class'] = df_fail[1].apply(lambda t: run_model_with_weights.get_score_category1(torch.load(t)))
split_series = df_fail.predicted_class.apply(lambda x: pd.Series(x))
df_fail = pd.concat([df_fail, split_series], axis=1)
df_fail.to_csv('df_fail.csv', header=False)

df_pass['predicted_class'] = df_pass[1].apply(lambda t: run_model_with_weights.get_score_category1(torch.load(t)))
split_series = df_pass.predicted_class.apply(lambda x: pd.Series(x))
df_pass = pd.concat([df_pass, split_series], axis=1)
df_pass.to_csv('df_pass.csv', header=False)

