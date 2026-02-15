import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_PATH = os.path.join(BASE_DIR, 'cleaned_data copy.csv')
TEST_PATH = os.path.join(BASE_DIR, 'test_data.csv')
TRAIN_PATH = os.path.join(BASE_DIR, 'train_data.csv')

print('Loading dataset...')
df = pd.read_csv(DATA_PATH)

print('Splitting dataset...')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f'Saving test set to {TEST_PATH}')
test_df.to_csv(TEST_PATH, index=False)
print(f'Saving train set to {TRAIN_PATH}')
train_df.to_csv(TRAIN_PATH, index=False)

print('Done!')
