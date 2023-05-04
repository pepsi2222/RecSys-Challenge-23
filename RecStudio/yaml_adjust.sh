
if [ "$(basename "$PWD")" == 'RecStudio' ];
then
    file='./recstudio/data/config/recsys.yaml'
else
    file='RecStudio/recstudio/data/config/recsys.yaml'
fi

if [ "${3}" == "START" ];
then
    if [ "${1}" == "True" ];
    then
        sed -i "8s/# //" $file
        sed -i "22s/# //" $file
        if [ "${4}" == "True" ];        # fine tune
        then
            sed -i "27s/# //" $file
        else
            sed -i "14s/# //" $file
        fi
    else
        if [ "${2}" == "is_clicked" ];
        then
            sed -i "5s/# //" $file
            if [ "${4}" == "True" ];        # fine tune
            then 
                sed -i "25s/# //" $file
            else
                sed -i "12s/# //" $file
            fi
        else
            sed -i "6s/# //" $file
            if [ "${4}" == "True" ];        # fine tune
            then
                sed -i "26s/# //" $file
            else
                sed -i "13s/# //" $file
            fi
        fi
        sed -i "20s/# //" $file
    fi
else
    if [ "${1}" == "True" ];
    then
        sed -i "8s/^/# /" $file
        sed -i "22s/^/# /" $file
        if [ "${4}" == "True" ];        # fine tune
        then
            sed -i "27s/^/# /" $file
        else
            sed -i "14s/^/# /" $file
        fi
    else
        if [ "${2}" == "is_clicked" ];
        then
            sed -i "5s/^/# /" $file
            if [ "${4}" == "True" ];        # fine tune
            then
                sed -i "25s/^/# /" $file
            else
                sed -i "12s/^/# /" $file
            fi
        else
            sed -i "6s/^/# /" $file
            if [ "${4}" == "True" ];        # fine tune
            then
                sed -i "26s/^/# /" $file
            else
                sed -i "13s/^/# /" $file
            fi
        fi
        sed -i "20s/^/# /" $file
    fi
fi

if [ "$(basename "$PWD")" == 'RecStudio' ];
then
    file='./recstudio/model/basemodel/basemodel.yaml'
else
    file='RecStudio/recstudio/model/basemodel/basemodel.yaml'
fi

if [ "${4}" == "True" ];
then
    if [ "${3}" == "START" ];
    then
        sed -i "24s/# //" $file
    else
        sed -i "24s/^/# /" $file
    fi
else
    if [ "${3}" == "START" ];
    then
        sed -i "25s/# //" $file
    else
        sed -i "25s/^/# /" $file
    fi
fi

