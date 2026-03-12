import pandas as pd
df = pd.read_parquet('./Data/train/train.parquet')
print(df['color_label'].value_counts())
