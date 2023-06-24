# counts the number of videos in the dataset

num_classes = dict(zip(classes, np.zeros(len(classes))))

idx_to_classes = {value: key for key, value in dataset.class_to_idx.items()}

for i in range(len(samples)):
    num_classes[ idx_to_classes[ samples[i][1] ] ] += 1

print('Number of videos in each class: ' + str(num_classes))