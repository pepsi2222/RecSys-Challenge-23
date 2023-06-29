python RecStudio/run.py -m=DCNv2 -c=./saved/DCNv2/2023-06-21-00-28-24.ckpt
python RecStudio/run.py -m=HardShareSEnet -c=./saved/HardShareSEnet/2023-06-21-22-59-11.ckpt
python RecStudio/run.py -m=PLE -c=./saved/PLE/2023-06-19-11-52-21.ckpt -d=multi
python RecStudio/run.py -m=PLEMLPSEnet -c=./saved/PLEMLPSEnet/2023-06-21-22-32-26.ckpt
python RecStudio/run.py -m=PLESEnet -c=./saved/PLESEnet/2023-06-21-22-24-10.ckpt
python ensemble.py