import pandas as pd

path = '/root/autodl-tmp/xingmei/RecSysChallenge23/predictions/LightGBM/2023-05-16-14-11-11.csv'
df = pd.read_csv(path, sep='\t')
df['is_installed'] = df['is_installed'].map(lambda x: 0 if x < 0 else 1 if x > 1 else x)
df.to_csv(path, sep='\t', index=False)