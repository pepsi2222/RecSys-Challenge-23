# Task

Two TimeSeries binary classification tasks.

Given 80 features, predict click or not and install or not.

Notice that in the case of online advertising, it is possible that a user has viewed an ad and then didn’t click on the ad and directly installed the underlying application.

- Tasks: to predict is_clicked and is_installed for the records in the test set

- Train data and Test data
  - 22 consecutive dates
  - train data: the first 21 days
  - test data: the 22nd day data

- Data Types

  |RowId|Date|Categorical features|Binary features|Numerical features|Labels|
  |:-:|:-:|:-:|:-:|:-:|:-:|
  |f_0|f_1|f_2 ... f_32|f_33 ... f_41|f_42 ... f_79|is_clicked<br>is_installed|

## Analyses

- Nulls of rows

    ||#Nulls = 0|#Nulls = 2<br>f_30, f_31|#Nulls = 11<br>f_43, f_51, f_58, f_59, f_64,<br> f_65, f_66, f_67, f_68, f_69, f_70|#Nulls = 13<br>f_30, f_31<br> f_43, f_51, f_58, f_59, f_64,<br> f_65, f_66, f_67, f_68, f_69, f_70|
    |:-:|:-:|:-:|:-:|:-:|
    |train|0.50|0.45|0.03|0.03|
    |test|0.68|0.26|0.05|0.02|

- Features
  - see feature.ipynb

- Correlation
    - see corr.ipynb
