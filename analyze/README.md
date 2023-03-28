# Task

Two TimeSeries binary classification tasks.

Given 80 features, predict click or not and install or not.

Notice that there is no click but there is an install.

## Analyses

- Nulls of rows

    ||# = 0|# = 2<br>f_30, f_31|# = 11<br>f_43, f_51, f_58, f_59, f_64,<br> f_65, f_66, f_67, f_68, f_69, f_70|# = 13<br>f_30, f_31<br> f_43, f_51, f_58, f_59, f_64,<br> f_65, f_66, f_67, f_68, f_69, f_70|
    |:-:|:-:|:-:|:-:|:-:|
    |train|0.50|0.45|0.03|0.03|
    |test|0.68|0.26|0.05|0.02|

- Features
  - see feature.ipynb

- Correlation
    - see corr.ipynb

## 


