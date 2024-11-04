# euler-meets-cuda

Fork from [stobis/euler-meets-cuda](https://github.com/stobis/euler-meets-cuda/tree/master) to solve RMQ queries with LCA
queries over a cartesian tree and compare its results with [temporal-hpc/rtxrmq](https://github.com/temporal-hpc/rtxrmq)


## Cloning and building instructions
To clone the repository together with 3rd party dependencies
```shell
git clone git@github.com:temporal-hpc/euler-meets-cuda-rmq.git
git submodule init
git submodule update
```

You may wish to update Makefile variables: CUDA, NVCC and you GPU's computing capability (NVCCSM) to match your system before building.

In case of stack overflow problems (e.g. segfaults when generating tests)
```shell
    ulimit -s unlimited
```


### Build and run RMQ
```shell
make rmq
```
and run the executable `rmq.e` with the same arguments as the rtxrmq program.

```
./rtxrmq <n> <q> <lr>

n   = num elements
q   = num RMQ querys
lr  = length of range; min 1, max n
  >0 -> value
  -1 -> uniform distribution (big values)
  -2 -> lognormal distribution (medium values)
  -3 -> lognormal distribution (small values)
Options:
   --reps <repetitions>      RMQ repeats for the avg time (default: 10)
   --dev <device ID>         device ID (default: 0)
   --nt  <thread num>        number of CPU threads
   --seed <seed>             seed for PRNG
   --check                   check correctness
   --save-time=<file>
   --save-power=<file>
```


