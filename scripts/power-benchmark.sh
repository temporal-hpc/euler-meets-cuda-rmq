#!/bin/bash
if [ "$#" -ne 12 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <alg> <rea> <reps>   <n> <q> <bsize> <lr> <filename>\n\n"
    printf "e.g: ${0}  0     8     5     8     10      26  26   15     20   RTX3090Ti\n"
    printf "\nnote:\n"
    printf "  - the *.csv extension will be placed automatically\n"
    printf "  - prefix (power) and suffix (alg) will be added to filename\n"
    printf "  - n,q,bsize,lr values are exponents of 2^x\n"
    printf "  - lr is the actual query range size\n\n"
    exit
fi
dev=${1}
nt=${2}
alg=${3}
rea=${4}
reps=${5}
n=${6}
q=${7}
bsize=${8}
lr=${9}
outfile_path=data/power-${10}-n${n}-q${q}-lr${lr}-ALG${alg}.csv
binary=./rmq.e

printf "args:\ndev=${dev} nt=${nt} alg=${alg} rea=${rea} reps=${reps} n=${n} q=${q} bsize=${bsize} lr=${lr} outfile_path=${outfile_path}\n\n"
#[ ! -f ${outfile_path} ] && echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > ${outfile_path}
DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATE}"

# change to bin directory
cd ../

SEED=${RANDOM}
printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --bs ${bsize} --reps $reps --nt $nt --dev $dev --save-power=${outfile_path} --seed ${SEED}\n"
        ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --bs ${bsize} --reps $reps --nt $nt --dev $dev --save-power=${outfile_path} --seed ${SEED}\n
# come back to scripts directory
cd ../scripts
DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "FINISH #DATE = ${DATE}"
