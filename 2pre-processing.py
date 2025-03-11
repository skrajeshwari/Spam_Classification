
# Text cleaning standardizes the text data, removing noise.
# Tokenization converts text to numerical representations for model input.
# Padding ensures all sequences have the same length, required for many ML models.

# prompt: Clean the text present in values. Create a function to do it and return the clean text. Update the dataframe with the cleaned text. Update the below code and also remove white spaces from the code.

import re

def clean_text(text):
  """
  This function cleans the text by removing punctuation, converting to lowercase,
  and removing extra whitespace.
  """
  text = text.lower()  # Convert to lowercase
  text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
  text = " ".join(text.split())  # Remove extra whitespace
  return text

# Apply the clean_text function to the 'v2' column
df['v2'] = df['v2'].apply(clean_text)

# prompt: Display few records of df['v2']

print(df['v2'].head(10))

# prompt: Split the dataframe into training and testing sets.

from sklearn.model_selection import train_test_split

# Split the dataframe into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['v2'], df['v1'], test_size=0.2, random_state=42
)
