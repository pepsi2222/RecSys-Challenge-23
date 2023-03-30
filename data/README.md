It's an intro of the *.csv under the directory.

sep is '\t' and header is 0 for all *.csv

# Concated Raw Data

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
