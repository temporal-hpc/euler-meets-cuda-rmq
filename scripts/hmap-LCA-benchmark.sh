#!/bin/bash
if [ "$#" -ne 10 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <rea> <reps>   <n1> <n2>  <q>  <lr1> <lr2>   <filename>\n\n"
    printf "e.g: ${0}  0     8     8     10     16   26    26     1    15     RTX3090Ti\n"
    printf "\nnote:\n"
    printf "  - the *.csv extension will be placed automatically\n"
    printf "  - prefix (perf) and suffix 'ALG7' will be added to filename\n"
    printf "  - n,q,lr values are exponents of 2^x\n\n"
    exit
fi
dev=${1}
nt=${2}
rea=${3}
reps=${4}
n1=${5}
n2=${6}
q=$((2**${7}))
lr1=${8}
lr2=${9}
outfile_path=data/hmap-${10}-ALG7.csv
binary=./rmq.e

# change to bin directory
cd ../

printf "args:\nLCA-GPU dev=${dev}  nt=${nt} rea=${rea} reps=${reps}  n=${n1}-${n2} q=${q}  lr=${lr1}-${lr2}  outfile_path=${outfile_path}\n\n"
[ ! -f ${outfile_path} ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}

DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATE}"


for(( n=$n1; n<=$n2; n++ ))
do
    for(( lr=$lr1; lr<=$n && lr<=$lr2; lr++ ))
    do
        nv=$((2**$n))
        lrdiv=$((2**$lr))
        lrv=$(( ($nv/$lrdiv) ))
        for(( R=1; R<=$rea; R++ ))
        do
            SEED=${RANDOM}
            printf "\n\n\n\n\n\n"
            printf "REALIZATION $R -> n=$nv lr=${lrv} lrfrac=1/$lrdiv\n"
            printf "${binary} $nv $q $lrv --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n"
                    ${binary} $nv $q $lrv --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}
        done
    done
done
# come back to scripts directory
cd scripts
DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "FINISH #DATE = ${DATE}"
