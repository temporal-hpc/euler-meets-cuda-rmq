#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <filename>\n"
    printf "e.g: ${0}   0     8   RTX4090\n"
    printf "\nnote:\n"
    printf "  - dev   : GPU device ID\n"
    printf "  - nt    : number of CPU threads (relevant for CPU methods)\n"
    exit
fi
dev=${1}
nt=${2}
name=${3}

N1=$((10**6))
N2=$((10**8))
DN=$((10**6))

Q1=$((2**26))
Q2=$((2**26))
DQ=$((100))

DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"

printf "args dev=${dev} nt=${nt} name=${name}, N1=${N1}, N2=${N2}, DN=${DN}, Q1=${Q1}, Q2=${Q2}, DQ=${DQ}\n\n"

for lr in {-1..-3}
do
    #./perf-LCA-benchmark.sh     <dev>  <nt>  <rea> <reps>  <n1>   <n2>   <dn>      <q1>  <q2>   <dq>    <lr>   <name>
    ./perf-LCA-benchmark.sh     ${dev} ${nt}   16    16    ${N1} ${N2}   ${DN}     ${Q1} ${Q2}  ${DQ}    ${lr}  ${name}
done
DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "FULL LCA BENCHMARK FINISHED: args dev=${dev} nt=${nt} name=${name}\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"
