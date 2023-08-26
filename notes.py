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