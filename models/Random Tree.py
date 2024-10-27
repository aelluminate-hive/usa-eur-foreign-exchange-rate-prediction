"""Import"""
# Import Libraries
import os
import pandas as pd
import numpy as np

# Import preprocessed dataset
preprocessed = "datasets/preprocessed.csv"

# Load the dataset
fexchange = pd.read_csv(preprocessed)

print(fexchange.head())
