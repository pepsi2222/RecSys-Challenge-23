import os
import re
import pandas as pd
from collections import defaultdict

config = defaultdict(list)
ops = ['combination', 'num_experts', 'num_layers', 'embed_dim', 'mlp_layer', 'dropout', 
        'scheduler', 'learning_rate', 'weight_decay']

log_dir = '/root/autodl-tmp/xingmei/RecSysChallenge23/log/DCNv2/finetune'
for file_name in os.listdir(log_dir):
    if 'csv' in file_name:
        continue
    # if file_name <= '2023-05-18-07-33-31.log':
    #     continue
    # if file_name <= '2023-05-19-02-52-25.log':
    #     continue
    if file_name <= '2023-05-19-18-55-36.log':
        continue
    f = open(os.path.join(log_dir, file_name))
    lines = f.readlines()
    for i, line in enumerate(lines):
        for op in ops:
            match_ = re.search(rf'\t{op}=(.*)$', line)
            if match_ is not None:
                config[op].append(match_.group(1))
        if 'The best score of logloss is' in line:
            match_ = re.search(r'The best score of logloss is (.*) on epoch (.*)', line)
            epoch = match_.group(2)
            config['logloss'].append(match_.group(1))
            config['epoch'].append(epoch)
            break
    while True:
        i = i - 1
        line = lines[i]
        if f'INFO Training: Epoch= {epoch}' in line or \
            f'INFO Training: Epoch=  {epoch}' in line:
            config['auc'].append(re.search(r'auc=(.*) ', line).group(1))
            break
df = pd.DataFrame(config)
df.sort_values(by=['combination', 'num_experts', 'num_layers', 'embed_dim'])
df = df[['combination', 'num_experts', 'num_layers', 'embed_dim', 
         'mlp_layer', 'dropout', 
         'scheduler', 'learning_rate', 'weight_decay',
         'epoch', 'logloss', 'auc'
         ]]
# print(df)    
df.to_csv(os.path.join(log_dir, 'summary.csv'), sep='\t', mode='a')            
                   