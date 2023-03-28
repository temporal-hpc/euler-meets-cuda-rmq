#!/bin/bash
printf "ARGS ${#}\n"
if [ "$#" -ne 12 ]; then
    echo "Run as"
    printf "    ${0} <dev> <nt> <rea> <reps>  <n1>         <n2>        <dn>       <q1>        <q2>         <dq>   <lr> <name>\n\n"
    printf "e.g ${0}  0     8     8     10  \$((10**6)) \$((10**8))  \$((10**6)) \$((2**26)) \$((2**26))   100    -1  RTX3090Ti\n"
    printf "\nnote:\n"
    printf "  - the *.csv extension will be placed automatically\n"
    printf "  - prefix (perf) and suffix (alg) will be added to filename\n\n"
    exit
fi
dev=${1}
nt=${2}
alg=7
rea=${3}
reps=${4}
n1=${5}
n2=${6}
dn=${7}
q1=${8}
q2=${9}
dq=${10}
lr=${11}
outfile_path=data/perf-${12}-ALG${alg}.csv
binary=./rmq.e

# change to bin directory
cd ../

printf "args:\ndev=${dev} nt=${nt} rea=${rea} reps=${reps} n=${n1}-${n2} (dn=${dn}) q=${q1}-${q2} (dq=${dq}) lr=${lr}   outfile_path=${outfile_path}\n\n"
[ ! -f ${outfile_path} ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}
DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"

for(( n=$n1; n<=$n2; n+=${dn} ))
do
    for(( q=$q1; q<=$q2; q+=${dq} ))
    do
        for(( R=1; R<=$rea; R++ ))
        do
            printf "\n\n\n\n\n\n\n\n"
            SEED=${RANDOM}
            printf "REALIZATION $R -> n=$n q=$q\n"
		    printf "${binary} $n $q ${lr} --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n"
		            ${binary} $n $q ${lr} --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}
        done
    done
done
# come back to scripts directory
cd scripts
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "perf-LCA-benchmark.sh FINISHED:\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
