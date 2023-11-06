search_value = 'Testing_Normal_Videos_Anomaly'
replacement_value = 'Normal'

# Replace the values in the specified column
df_test[0] = df_test[0].replace(search_value, replacement_value)

plt.hist(df_test[0], alpha=0.7, rwidth=0.8)
plt.xticks(rotation=45)
plt.figure(figsize=(12, 4))  # Adjust the width and height as needed
plt.savefig('histogram.png')
plt.show()

class_counts = df_test[0].value_counts()
# Create a bar plot
class_counts.plot(kind='bar', color='skyblue')
plt.title('Test Data Class Distribution')
plt.xlabel('Class Labels')
plt.ylabel('Number of Samples')
plt.xticks(rotation=25)
plt.show()

# list enumeration
indices = [i for i, x in enumerate(target_label_list) if x == 1]

[sk_data.filenames[i] for i in indices]

[sk_data.filenames[j] for j in [i for i, x in enumerate(target_label_list) if x == 1]]

# saving to file
pd.DataFrame(file_names_pass).to_csv('correct_class_videos_full_list.txt', header=False)

# finding files not present in the feature extracted list of pt files
data1 = df_test[1]
len(data1)
290

data2 = (df1[df1.isin(df_test[1])])
len(data2)
288
# Find items in Series 1 that are not in Series 2
items_not_in_series2 = data1[~data1.isin(data2)]
len(items_not_in_series2)
2
items_not_in_series2
normalvideos924x264
normalvideos935x264
Name: 1, dtype: object

#names of all video files with a given target label in sk_loader
sk_data.filenames[np.where(sk_data.target==11)[0]]

# save a classifier to disk
from joblib import dump
dump(classifier, 'gradient_boost_clf_all_files.joblib')

# Load the classifier from the saved file
classifier = load('gradient_boost_clf_all_files.joblib')

