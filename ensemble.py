import pandas as pd
import numpy as np

pred_path = [
    "./predictions/DCNv2/2023-06-21-00-28-24is_installed.csv",
    "./predictions/HardShareSEnet/2023-06-21-22-59-11['is_clicked', 'is_installed'].csv",
    "./predictions/PLE/2023-06-19-11-52-21['is_clicked', 'is_installed'].csv",
    "./predictions/PLEMLPSEnet/2023-06-21-22-32-26['is_clicked', 'is_installed'].csv",
    "./predictions/PLESEnet/2023-06-21-22-24-10['is_clicked', 'is_installed'].csv",
]
df = {}
for i, pth in enumerate(pred_path):
    tmp = pd.read_csv(pth, sep='\t')
    df[i] = tmp['is_installed_prob']
    
df = pd.DataFrame(df)

final_pred = pd.DataFrame({
    'RowId': pd.read_csv('./data/tst_rowid.csv')['f_0'].to_list(), 
    'is_clicked': 0, 
    'is_installed': np.sort(df[-160973:].copy(), axis=1)[:, 1:-1].mean(axis=1)
})
final_pred.to_csv(f'./predictions/prediction.csv', sep='\t', index=False)