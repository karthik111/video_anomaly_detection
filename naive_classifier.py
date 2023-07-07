import os
from torchvision import datasets
import sklearn.datasets
import torch
import numpy as np


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

from sklearn.model_selection import train_test_split

import io
tensorObject = torch.load(io.BytesIO(sk_data.data[0]))

feature_array_list = [torch.load(io.BytesIO(byte_obj)).numpy() for byte_obj in sk_data.data]

labels = sk_data.target_names

target_label_list = [0 if value == 7 else 1 for value in sk_data.target]
#target_label_list = sk_data.target


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(feature_array_list, target_label_list, test_size=0.2, random_state=42)

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Create and train an SVM classifier
classifier = svm.SVC(decision_function_shape='ovo')
#classifier = RandomForestClassifier()
#classifier = DummyClassifier(strategy='most_frequent')
classifier = GradientBoostingClassifier()

classifier.fit(np.squeeze(X_train), y_train)

# Predict using the trained classifier
predictions = classifier.predict(np.squeeze(X_val))

from sklearn.metrics import accuracy_score, roc_auc_score

accuracy = accuracy_score(y_val, predictions)
# Print the accuracy
print("Accuracy:", accuracy)

# Compute the AUC score
auc_score = roc_auc_score(y_val, predictions)

# Print the AUC score
print("AUC Score:", auc_score)

from sklearn.model_selection import cross_val_score

# Perform cross-validation with 5 folds
scores = cross_val_score(classifier, np.squeeze(feature_array_list), target_label_list, cv=5)

# Print the accuracy scores for each fold
print("Accuracy scores:", scores)

# Print the average accuracy across all folds
print("Average accuracy:", scores.mean())
