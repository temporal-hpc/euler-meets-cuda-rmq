#!/bin/bash
if [ "$#" -ne 10 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <rea> <reps>   <n1> <n2>  <q1> <q2> <lr> <filename>\n\n"
    printf "e.g: ${0}  0     8     8     10      16   26    10   26   10   RTX3090Ti\n"
    printf "\nnote:\n"
    printf "  - the *.csv extension will be placed automatically\n"
    printf "  - prefix (perf) and suffix (alg) will be added to filename\n"
    printf "  - n,q,lr values are exponents of 2^x\n\n"
    exit
fi
dev=${1}
nt=${2}
alg=7
rea=${3}
reps=${4}
n1=${5}
n2=${6}
q1=${7}
q2=${8}
lr=${9}
outfile_path=data/perf-${10}-ALG${alg}.csv
binary=./rmq.e

# change to bin directory
cd ../

printf "args:\ndev=${dev} nt=${nt} rea=${rea} reps=${reps} n=${n1}-${n2} q=${q1}-${q2} lr=${lr}   outfile_path=${outfile_path}\n\n"
[ ! -f ${outfile_path} ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}
DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"


for(( n=$n1; n<=$n2; n++ ))
do
    for(( q=$q1; q<=$q2; q++ ))
    do
        for(( R=1; R<=$rea; R++ ))
        do
            printf "\n\n\n\n\n\n\n\n"
            SEED=${RANDOM}
            printf "REALIZATION $R -> n=$((2**$n)) q=$((2**$q))\n"
		    printf "${binary} $((2**$n)) $((2**$q)) ${lr} --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n"
		            ${binary} $((2**$n)) $((2**$q)) ${lr} --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n
        done
    done
done
# come back to scripts directory
cd scripts
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "perf-LCA-benchmark.sh FINISHED:\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
