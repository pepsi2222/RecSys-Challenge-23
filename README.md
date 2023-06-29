# RecSys Challenge 23

## Data

run `analyze/process.ipynb`, `data/split_sample.ipynb` and `feature_engineer/weekday.ipynb` to generate *.csv

## Path

fix url in `RecStudio/recstudio/data/config/*.yaml`

## Predict

~~~
python RecStudio/run.py -m=DCNv2 -c=./saved/DCNv2/2023-06-21-00-28-24.ckpt
~~~

~~~
python RecStudio/run.py -m=HardShareSEnet -c=./saved/HardShareSEnet/2023-06-21-22-59-11.ckpt
~~~

~~~
python RecStudio/run.py -m=PLE -c=./saved/PLE/2023-06-19-11-52-21.ckpt -d=multi
~~~

~~~
python RecStudio/run.py -m=PLEMLPSEnet -c=./saved/PLEMLPSEnet/2023-06-21-22-32-26.ckpt
~~~

~~~
python RecStudio/run.py -m=PLESEnet -c=./saved/PLESEnet/2023-06-21-22-24-10.ckpt
~~~

Then, concat them and run `ensemble.ipynb` to generate the final prediction.

## Try more

Please refer to [RecStudio](https://github.com/ustcml/RecStudio), which is a unified, highly-modularized and recommendation-efficient recommendation library based on PyTorch. We welcome more effient models from you.