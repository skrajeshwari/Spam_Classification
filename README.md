# Spam_Classification

Generating a prompt/prompts using no code based LLMs to solve a text classification problem.
Dataset: Spam / ham classification

# Objectives: 
1) Create text vectorization layer.
2) Create embeddings of words in the message.
3) Visualize the different categories of words.
4) Build a model to classify the messages as Spam or Ham.
5) Get the evaluation score of the model.

# Dataset

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

# Use the SMS Spam collection dataset

* Based on your understanding of the steps required to do text classification, generate the code using LLM to accomplish the task.
* There is no restriction on the number of prompts or the choice of LLM.
* Do not use a pre-trained model.
* Use any LLM to pass prompts and generate code from it.
* All the sections are to be solved by making use of prompts.
* Feel free to add/delete code cells as you may require.
* Use any number of code cells/prompts to solve any particular section.

# Parameters of evaluations

* Accuracy on Test data set
* Choice of preprocessing steps
* Model Architecture
* Design parameters choice
* Number of experiments conducted
* Parameter tuning
* Visualizations
    ->Train validation loss & other metrics
    ->Word clouds of words in each category
    ->Embeddings visualizations

# Step 1: Install kaggle
Ensure you have the kaggle package installed in your Codespaces environment. Use the terminal or add it to your requirements.txt file.

To install it directly in the terminal:
pip install kaggle

# Step 2: Set Up Kaggle API Key
Go to Kaggle and log in.

Navigate to your account settings by clicking on your profile icon and selecting Account.

Scroll to the API section and click Create New API Token. This will download a kaggle.json file.

Upload this file to your Codespaces environment. You can do this by dragging it into your project workspace in the Codespaces file explorer.

Alternatively, move it to the ~/.kaggle/ directory and ensure proper permissions:
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Step 3: Use Python Code to Download Dataset
Now you can use Python code to download the dataset. Here’s an example for downloading the SMS Spam Collection Dataset (uciml/sms-spam-collection-dataset):

import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# Define dataset identifier and download path
dataset = "uciml/sms-spam-collection-dataset"
download_path = "datasets"

# Download the dataset
api.dataset_download_files(dataset, path=download_path, unzip=True)

print(f"Dataset downloaded and unzipped in: {os.path.abspath(download_path)}")

Step 4: Verify and Explore the Dataset
After running the script, the dataset files will be downloaded and unzipped in the datasets/ directory. You can explore the directory using the file explorer or by listing the files programmatically:

import os

# List downloaded files
files = os.listdir(download_path)
print("Downloaded files:", files)

# Initial Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.decomposition import PCA

