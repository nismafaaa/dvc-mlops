from scipy.io import arff
import pandas as pd
import os

data, meta = arff.loadarff('data/dataset_31_credit-g.arff')
df = pd.DataFrame(data)

for col in df.columns:
  if df[col].dtype == 'object':
    df[col] = df[col].str.decode('utf-8')

cat_features = df.select_dtypes(include=['object']).columns
cat_features = cat_features.drop('class')

ordinal = ['checking_status', 'savings_status', 'employment']

checking_status_mapping = {
    'no checking': 0,
    '<0': 1,
    '0<=X<200': 2,
    '>=200': 3
}

savings_status_mapping = {
    'no known savings': 0,
    '<100': 1,
    '100<=X<500': 2,
    '500<=X<1000': 3,
    '>=1000': 4
}

employment_mapping = {
    'unemployed': 0,
    '<1': 1,
    '1<=X<4': 2,
    '4<=X<7': 3,
    '>=7': 4
}

df['checking_status'] = df['checking_status'].map(checking_status_mapping)
df['savings_status'] = df['savings_status'].map(savings_status_mapping)
df['employment'] = df['employment'].map(employment_mapping)

num_features = df.select_dtypes(include=['float64']).columns

bins = [18, 30, 40, 50, 60, 70, 80]
labels = ['0', '1', '2', '3', '4', '5']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
df.drop('age', axis=1, inplace=True)

categorical = df.select_dtypes(include=['object']).columns
categorical = categorical.drop('class')
df_encoded = pd.get_dummies(df, columns=categorical)

folder_path = 'data'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

file_path = os.path.join(folder_path, 'processed_data.csv')

df_encoded.to_csv(file_path, index=False)

print(f"DataFrame saved to {file_path}")