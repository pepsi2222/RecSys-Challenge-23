file='/root/autodl-tmp/xingmei/RecSysChallenge23/RecStudio/recstudio/model/fm/config/dcnv2.yaml'
for combination in 'parallel' 'stacked'
do
    sed -i -r "2s/combination: [a-z]+/combination: $combination/" $file
    for num_experts in 3 4 5
    do
        sed -i -r "4s/num_experts: [0-9]+/num_experts: $num_experts/" $file
        for num_layers in 2 3 4
        do
            sed -i -r "5s/num_llayers: [0-9]+/num_layers: $num_layers/" $file
            for embed_dim in 20 30 40
            do
                sed -i -r "6s/embed_dim: [0-9]+/embed_dim: $embed_dim/" $file
                python RecStudio/run.py
                wait
            done
        done
    done
done
# for scheduer in 'onplateau' 'exponential'
# do
#     sed -i -r "14s/scheduler: [a-z]+/scheduler: $scheduler/" $file
# done