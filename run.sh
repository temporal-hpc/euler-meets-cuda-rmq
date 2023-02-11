dev=0
nt=1
bs=0
rep=10

[ -e "data_rmq/data.csv" ] || echo "dev,alg,reps,n,bs,q,lr,t,q/s,ns/q,construction" > data_rmq/data.csv

cd build/
for alg in {7..7}
do
	for n in {16..26..2}
	do
		for q in {10..25..3}
		do
			for lr in {5..25..4}
			do
				if [ $lr -lt $n ]
				then
					./rmq.e $rep $RANDOM $dev $((2**$n)) $bs $((2**$q)) $((2**$lr)) $nt $alg
				fi
			done
			./rmq.e $rep $RANDOM $dev $((2**$n)) $((2**$q)) -1 $nt $alg
		done
	done
done

