It's an intro of the *.csv under the directory.

for all *.csv
- sep='\t' 
- header=0 
- dir='/root/data01/xingmei/Sharechat-RecSys-Challenge-23/data'

# Raw Data

- train.csv
- test.csv

# Split Day 66 As Validation

The remaining of train.csv is a customized train dataset.

Which are **got rid of RowId(f_0)**.

- customized_trn.csv
- customized_val.csv
- customized_trn_val.csv

Notice: the nums of entries for trn and val are 3387880, 97972, respectively.

## Seperate the Customized Ones to Two Tasks

Which can be used in recstudio directly.

Which are **got rid of RowId(f_0) and the other label**.

- click_trn_val.csv
- install_trn_val.csv

## Sample 30% For Tuning Hyper-Parameters

sample from customized_*.csv

sample 30% from each day

the last two *.csv are for single task

- sub_trn_val.csv
- click_sub_trn_val.csv
- install_sub_trn_val.csv

Notice: the nums of entries for trn and val are 1016364, 29392, respectively.


# Simplified Data

all simplified_*.csv

delete f7 (unique_num=1), f27, f28, f29 (pearson corr=1 with f26)

## Fine tune

- For task-specific fine tune
  - simplified_click_sub_trn_val.csv
  - simplified_install_sub_trn_val.csv
- For multitask fine tune
  - simplified_sub_trn_val.csv

Notice: trn/val split_ratio is [1016364, 29392]

## Test
- For task-specific test
  - simplified_click_trn_val_tst.csv
  - simplified_install_trn_val_tst.csv
- For multitask test
  - simplified_trn_val_test.csv

Notice: trn/val/tst split_ratio is [3387880, 97972, 160973]
