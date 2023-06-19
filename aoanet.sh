file='/data2/home/xingmei/RecSys23/RecStudio/recstudio/model/fm/config/aoanet.yaml'
for embed_dim in 10 20
do
    sed -i -r "7s/embed_dim: [0-9.]+/embed_dim: $embed_dim/" $file
    for num_interaction_layers in 2 3
    do
        sed -i -r "6s/num_interaction_layers: [0-9.]+/num_interaction_layers: $num_interaction_layers/" $file

        sed -i -r "4s/dropout: [0-9.]+/dropout: 0/" $file
        sed -i -r "11s/weight_decay: [0-9](e-[0-9])?/weight_decay: 0/" $file
        {   python RecStudio/run.py --gpu 3 --model AOANet 
        }&
        sleep 5s
        sed -i -r "11s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-6/" $file
        {   python RecStudio/run.py --gpu 4 --model AOANet 
        }&
        sleep 5s

        sed -i -r "4s/dropout: [0-9.]+/dropout: 0.3/" $file
        sed -i -r "11s/weight_decay: [0-9](e-[0-9])?/weight_decay: 0/" $file
        {   python RecStudio/run.py --gpu 5 --model AOANet 
        }&
        sleep 5s
        sed -i -r "11s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-6/" $file
        {   python RecStudio/run.py --gpu 6 --model AOANet 
        }&
        sleep 5s


        sed -i -r "4s/dropout: [0-9.]+/dropout: 0.5/" $file
        sed -i -r "11s/weight_decay: [0-9](e-[0-9])?/weight_decay: 0/" $file
        {   python RecStudio/run.py --gpu 7 --model AOANet 
        }&
        sleep 5s
        sed -i -r "11s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-6/" $file
        {   python RecStudio/run.py --gpu 8 --model AOANet 
        }&

        wait
    done
done