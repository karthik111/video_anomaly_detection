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

classes_dict = {value: index for index, value in enumerate(sk_data.target_names)}
dict_idx_to_class = {index: value for index, value in enumerate(sk_data.target_names)}

##
num_classes = dict(zip(sk_data.target_names, np.zeros(len(sk_data.target_names))))

idx_to_classes = {value: key for key, value in dataset.class_to_idx.items()}

for i in range(len(sk_data.target)):
    #num_classes[ idx_to_classes[ samples[i][1] ] ] += 1
    num_classes[ dict_idx_to_class[sk_data.target[i] ] ] += 1
##

target_label_list = [0 if (value==12 or value==13) else 1 for value in sk_data.target]
#target_label_list = sk_data.target

indices = np.arange(len(feature_array_list))
# Split the data into training and validation sets
# X_train, X_val, y_train, y_val, indices_train, indices_test = train_test_split(feature_array_list, target_label_list, indices, test_size=0.2, random_state=42)

## TO DO: RUN run_sk_loader.py

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
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

probs = classifier.predict_proba(np.squeeze(X_val))
auc_score = roc_auc_score(y_val, probs[:, 1])
RocCurveDisplay.from_predictions(y_val, probs[:, 1])
# Print the AUC score
print("AUC Score:", auc_score)
plt.show()

from sklearn.model_selection import cross_val_score

# Perform cross-validation with 5 folds
scores = cross_val_score(classifier, np.squeeze(feature_array_list), target_label_list, cv=5)

# Print the accuracy scores for each fold
print("Accuracy scores:", scores)

# Print the average accuracy across all folds
print("Average accuracy:", scores.mean())

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Assuming you have true labels (y_true) and predicted labels (y_pred)
cm = confusion_matrix(y_val, predictions)

# If you have multiclass classification, you can use normalize='true' or 'all' to get normalized values in the heatmap.
# For binary classification, you can omit the normalize parameter.
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')

plt.title("Confusion Matrix")
plt.show()

# find false predictions

mask = y_val != predictions

indices_fail = np.where(mask)[0]

for i in indices_fail:
    print(sk_data.filenames[indices_test[i]])

file_names = [sk_data.filenames[indices_test[i]] for i in indices_fail]