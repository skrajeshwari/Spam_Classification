import pandas as pd

# Provide the path to the dataset in the GitHub Codespaces environment
df = pd.read_csv("spam.csv", encoding='latin-1")

# prompt: Display the head section of the dataset. Display top 7 records.
df.head(7)

# prompt: Give a description of the dataset. Number of features. Check the number of unique values in the features. Put the entire code in the function and call it.

def describe_dataset():
  """
  This function provides a description of the dataset, including the number of features,
  and the number of unique values in each feature.
  """
  print("Dataset Description:")
  print(f"Number of features: {df.shape[1]}")
  print("\nUnique values in each feature:")
  for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

describe_dataset()

# prompt: Create the final dataset by only considering features v1 and v2. Name the dataframe as 'df'.

df = df[['v1', 'v2']]

# prompt: Display the first 7 records of the updated dataframe.

df.head(7)

# prompt: Create a new variable labels by taking the 'v1' feature from the dataframe and values variable from the 'v2' feature of the dataframe.

labels = df['v1']
values = df['v2']

# prompt: Visualize labels by making use of appropriate plots.

# Count the occurrences of each label
label_counts = labels.value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(label_counts.index, label_counts.values)
plt.xlabel("Labels")
plt.ylabel("Count")
plt.title("Distribution of Labels")
plt.show()

# prompt: encode the categorical variable and replace in the dataframe. Map as follows, {'ham': 0, 'spam': 1}

# Create a mapping dictionary
mapping = {'ham': 0, 'spam': 1}

# Replace the values in the 'v1' column using the mapping
df['v1'] = df['v1'].map(mapping)

# prompt: Display the first 7 records of the updated dataframe.

df.head(7)


