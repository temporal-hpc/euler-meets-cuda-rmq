if [ "$#" -ne 5 ]; then
    echo "Run as"
    printf "     ${0} <dev> <nt> <alg>\n"
    printf "e.g: ${0}   0     8    5\n"
    printf "\nnote:\n"
    printf "  - dev   : GPU device ID\n"
    printf "  - nt    : number of CPU threads (relevant for CPU methods)\n"
    printf "  - alg   : algorithm (check ./rtxrmq help message)\n"
    exit
fi
dev=${1}
nt=${2}
alg=${3}
name="3090Ti"

printf "args dev=${dev} nt=${nt} alg=${alg} bsize=${bsize} name=${name}\n\n"

#TODO
#    - put efficient `bsize` values at each n/lr (obtained from results of hmap)

# UNIFORM DISTRIBUTION (large values)
#./perf-benchmark.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bsize>   <lr> <filename>
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      10   10     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      11   11     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      12   12     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      13   13     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      14   14     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      15   15     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      16   16     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      17   17     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      18   18     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      19   19     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      20   20     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      21   21     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      22   22     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      23   23     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      24   24     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      25   25     1   26   ${bsize}   -1   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      26   26     1   26   ${bsize}   -1   ${name}

# LOGNORMAL DISTRIBUTION (medium values)
#./perf-benchmark.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bsize>   <lr> <filename>
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      10   10     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      11   11     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      12   12     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      13   13     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      14   14     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      15   15     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      16   16     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      17   17     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      18   18     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      19   19     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      20   20     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      21   21     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      22   22     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      23   23     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      24   24     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      25   25     1   26   ${bsize}   -2   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      26   26     1   26   ${bsize}   -2   ${name}

# LOGNORMAL DISTRIBUTION (small values)
#./perf-benchmark.sh <dev> <nt>  <alg>  <rea> <reps>   <n1> <n2>  <q1> <q2>  <bsize>   <lr> <filename>
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      10   10     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      11   11     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      12   12     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      13   13     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      14   14     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      15   15     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      16   16     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      17   17     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      18   18     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      19   19     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      20   20     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      21   21     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      22   22     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      23   23     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      24   24     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      25   25     1   26   ${bsize}   -3   ${name}
./perf-benchmark.sh ${dev} ${nt} ${alg}  16     32      26   26     1   26   ${bsize}   -3   ${name}
