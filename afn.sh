file='/data2/home/xingmei/RecSys23/RecStudio/recstudio/model/fm/config/afn.yaml'
for embed_dim in 10 20
do
    sed -i -r "6s/embed_dim: [0-9.]+/embed_dim: $embed_dim/" $file
    for log_hidden_size in 200 500
    do
        sed -i -r "2s/log_hidden_size: [0-9.]+/log_hidden_size: $log_hidden_size/" $file

        sed -i -r "5s/dropout: [0-9.]+/dropout: 0/" $file
        sed -i -r "15s/weight_decay: [0-9](e-[0-9])?/weight_decay: 0/" $file
        {   python RecStudio/run.py --gpu 0
        }&
        sleep 10s
        sed -i -r "15s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-6/" $file
        {   python RecStudio/run.py --gpu 0
        }&
        sleep 10s

        sed -i -r "5s/dropout: [0-9.]+/dropout: 0.3/" $file
        sed -i -r "15s/weight_decay: [0-9](e-[0-9])?/weight_decay: 0/" $file
        {   python RecStudio/run.py --gpu 1
        }&
        sleep 10s
        sed -i -r "15s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-6/" $file
        {   python RecStudio/run.py --gpu 1
        }&
        sleep 10s


        sed -i -r "5s/dropout: [0-9.]+/dropout: 0.5/" $file
        sed -i -r "15s/weight_decay: [0-9](e-[0-9])?/weight_decay: 0/" $file
        {   python RecStudio/run.py --gpu 2
        }&
        sleep 10s
        sed -i -r "15s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-6/" $file
        {   python RecStudio/run.py --gpu 2
        }&

        wait
    done
done