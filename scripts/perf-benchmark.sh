#!/bin/bash
if [ "$#" -ne 13 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <alg> <rea> <reps>   <n1> <n2>  <q1> <q2> <bs-or-nb> <bsize/#blocks> <lr> <filename>\n\n"
    printf "e.g: ${0}  0     8     5     8     10      16   26    10   26      bs            15        10   RTX3090Ti\n"
    printf "\nnote:\n"
    printf "  - the *.csv extension will be placed automatically\n"
    printf "  - prefix (perf) and suffix (alg) will be added to filename\n"
    printf "  - n,q,bsize,#blocks values are exponents of 2^x\n\n"
    exit
fi
dev=${1}
nt=${2}
alg=${3}
rea=${4}
reps=${5}
n1=${6}
n2=${7}
q1=${8}
q2=${9}
bs_or_nb=${10}
bsize=${11}
lr=${12}
outfile_path=data/perf-${13}-ALG${alg}.csv
binary=./rmq.e

printf "args:\ndev=${dev} nt=${nt} alg=${alg} rea=${rea} reps=${reps} n=${n1}-${n2} q=${q1}-${q2} bsize=${bsize} lr=${lr}   outfile_path=${outfile_path}\n\n"
[ ! -f ${outfile_path} ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}
DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"

# change to bin directory
cd ../

for(( n=$n1; n<=$n2; n++ ))
do
    for(( q=$q1; q<=$q2; q++ ))
    do
        for(( R=1; R<=$rea; R++ ))
        do
            printf "\n\n\n\n\n\n\n\n"
            SEED=${RANDOM}
            printf "REALIZATION $R -> n=$((2**$n)) q=$((2**$q))\n"
	    if [ "${bs_or_nb}" = "bs" ]; then
		    printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --bs $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n"
		    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --bs $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n
            else
		    printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n"
		    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED}\n
	    fi
        done
    done
done
# come back to scripts directory
cd ../scripts
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "perf-benchmark.sh FINISHED:\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
