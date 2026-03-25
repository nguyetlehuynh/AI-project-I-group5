import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

'''
# data_df = pd.DataFrame(columns=["input", "output"])

# row = data_df._append({
#     "input": 'data/images/paper_001.jpg',
#     "output": 'data/images/paper_001.jpg'
# })

# data_df = pd.concat([data_df, row])
# print(data_df)
'''

dataset = pd.read_csv(r"data\CSVs\dataset.csv")


train_data, val_data = train_test_split(dataset, test_size=0.3)
train_data.to_csv(r"data\CSVs\train_df.csv", index=False)
val_data.to_csv(r"data\CSVs\val_df.csv", index=False)