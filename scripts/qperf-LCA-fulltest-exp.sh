#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt>  <filename>\n"
    printf "e.g: ${0}   0     8    RTX3090Ti\n"
    printf "\nnote:\n"
    printf "  - dev   : GPU device ID\n"
    printf "  - nt    : number of CPU threads (for generating queries in parallel)\n"
    exit
fi
dev=${1}
nt=${2}
name=${3}

DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "LCA-GPU BENCHMARK START #DATE = ${DATEBEGIN}"

printf "args dev=${dev} nt=${nt} name=${name}\n\n"
for lr in {-1..-3}
do
    #./perf-LCA-benchmark-exp.sh    <dev> <nt>   <rea> <reps>   <n1> <n2>  <q1> <q2>   <lr> <filename>
     ./perf-LCA-benchmark-exp.sh  ${dev} ${nt}    16     32       26   26    0   26   ${lr}  ${name}-QPERF
done
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "FULL LCA-GPU BENCHMARK FINISHED: args dev=${dev} nt=${nt} name=${name}\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
