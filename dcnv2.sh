file='/root/autodl-tmp/xingmei/RecSysChallenge23/RecStudio/recstudio/model/fm/config/dcnv2.yaml'
for dropout in 0 0.2 0.3
do
    sed -i -r "10s/dropout: [0-9.]+/dropout: $dropout/" $file
    for scheduler in 'onplateau' 'exponential'
    do
        sed -i -r "14s/scheduler: [a-z]+/scheduler: $scheduler/" $file
        for learning_rate in '1e-4' '1e-3'
        do
            sed -i -r "15s/learning_rate: 1e-[0-9]/learning_rate: $learning_rate/" $file


            sed -i -r "16s/weight_decay: [0-9](e-[0-9])?/weight_decay: 0/" $file
            python RecStudio/run.py
            sleep 1m

            sed -i -r "16s/weight_decay: [0-9](e-[0-9])?/weight_decay: 3e-5/" $file
            python RecStudio/run.py
            sleep 1m

            sed -i -r "16s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-4/" $file
            python RecStudio/run.py

            wait
            # for weight_decay in '0' '3e-5' '1e-4' 
            # do
            #     sed -i -r "16s/weight_decay: [0-9](e-[0-9])?/weight_decay: $weight_decay/" $file
            #     python RecStudio/run.py
            #     wait
            # done
        done
    done
done