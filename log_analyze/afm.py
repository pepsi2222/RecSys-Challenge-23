import os
import re
import pandas as pd
from collections import defaultdict

config = defaultdict(list)
ops = ['embed_dim', 'attention_dim', 'dropout', 'learning_rate', 'weight_decay']

log_dir = '/root/autodl-tmp/xingmei/RecSysChallenge23/log/AFM/finetune'
for file_name in os.listdir(log_dir):
    flag=False
    if 'csv' in file_name or file_name == '2023-06-05-17-15-41.log':
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
            flag=True
            break
    if not flag:
        print(file_name)
    while True:
        i = i - 1
        try:
            line = lines[i]
        except:
            print(file_name, epoch)
            break
        if f'INFO Training: Epoch= {epoch}' in line or \
            f'INFO Training: Epoch=  {epoch}' in line or \
            f'INFO Training: Epoch={epoch}' in line    :
            config['auc'].append(re.search(r'auc=(.*) ', line).group(1))
            break
for k, v in config.items():
    print(k, len(v))
df = pd.DataFrame(config)
df.sort_values(by=['embed_dim', 'attention_dim', 'dropout', 'learning_rate', 'weight_decay'])
df = df[['embed_dim', 'attention_dim', 'dropout', 'learning_rate', 'weight_decay',
         'epoch', 'logloss', 'auc'
         ]]
# print(df)    
df.to_csv(os.path.join(log_dir, 'summary.csv'), sep='\t', mode='a')            
                   