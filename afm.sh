file='/root/autodl-tmp/xingmei/RecSysChallenge23/RecStudio/recstudio/model/fm/config/afm.yaml'
for embed_dim in 64 128
do
    sed -i -r "3s/embed_dim: [0-9.]+/embed_dim: $embed_dim/" $file
    for attention_dim in 20 40
    do
        sed -i -r "2s/attention_dim: [0-9.]+/attention_dim: $attention_dim/" $file
        for dropout in 0 0.3 0.5
        do
            sed -i -r "4s/dropout: [0-9.]+/dropout: $dropout/" $file
            for learning_rate in '1e-4' '1e-3'
            do
                sed -i -r "7s/learning_rate: 1e-[0-9]/learning_rate: $learning_rate/" $file


                # sed -i -r "8s/weight_decay: [0-9](e-[0-9])?/weight_decay: 0/" $file
                # python RecStudio/run.py
                # sleep 1m

                # sed -i -r "8s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-6/" $file
                # python RecStudio/run.py
                # sleep 1m

                # sed -i -r "8s/weight_decay: [0-9](e-[0-9])?/weight_decay: 1e-5/" $file
                # python RecStudio/run.py

                # wait
                for weight_decay in '0' '1e-6' '1e-5' 
                do
                    sed -i -r "8s/weight_decay: [0-9](e-[0-9])?/weight_decay: $weight_decay/" $file
                    python RecStudio/run.py
                    wait
                done
            done
        done
    done
done