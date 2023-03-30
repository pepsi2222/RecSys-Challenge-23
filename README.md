# About Submodule 

To clone submodules, execute it in the outside directory:
~~~
$ git submodule update --init --recursive
$ cd RecStudio
$ git checkout -b recsys
~~~

To pull submodules, go to the submodule directory:
~~~
git pull
~~~

To push submodules, just the same with normal modules in the vscode.

# TODO
- 用deepctr/fuxictr处理数据
- Single task
  - FM, MLP
- Two tasks
  - PLM 等
- 主从任务
- trick
  - automl