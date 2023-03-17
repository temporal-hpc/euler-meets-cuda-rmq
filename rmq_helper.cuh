#pragma once

#include <curand_kernel.h>
#include <unistd.h>
#include <random>
#include <cmath>
#include <omp.h>
#include <getopt.h>

#define BSIZE 1024

#define AC_RESET   "\033[0m"
#define AC_BLACK   "\033[30m"      /* Black */
#define AC_RED     "\033[31m"      /* Red */
#define AC_GREEN   "\033[32m"      /* Green */
#define AC_YELLOW  "\033[33m"      /* Yellow */
#define AC_BLUE    "\033[34m"      /* Blue */
#define AC_MAGENTA "\033[35m"      /* Magenta */
#define AC_CYAN    "\033[36m"      /* Cyan */
#define AC_WHITE   "\033[37m"      /* White */
#define AC_BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define AC_BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define AC_BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define AC_BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define AC_BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define AC_BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define AC_BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define AC_BOLDWHITE   "\033[1m\033[37m"      /* Bold White */


#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    if (error != cudaSuccess)                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << cudaGetErrorString(error) << "'\n";                         \
  }

#define CUDA_DRIVER_CHECK(error)                                               \
  {                                                                            \
    if (error != CUDA_SUCCESS) {                                               \
      const char *error_str = nullptr;                                         \
      cuGetErrorString(error, &error_str);                                     \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << error_str << "'\n";                                         \
    }                                                                          \
  }

#define ARG_BS 1
#define ARG_NB 2
#define ARG_REPS 3
#define ARG_DEV 4
#define ARG_NT 5
#define ARG_SEED 6
#define ARG_CHECK 7
#define ARG_TIME 8
#define ARG_POWER 9

struct CmdArgs {
    int n, q, lr, alg, bs, nb, reps, dev, nt, seed, check, save_time, save_power;
    std::string time_file, power_file;
};

#define NUM_REQUIRED_ARGS 10
void print_help(){
    fprintf(stderr, AC_BOLDGREEN "run as ./rtxrmq <n> <q> <lr> <alg>\n\n" AC_RESET
                    "n   = num elements\n"
                    "q   = num RMQ querys\n"
                    "lr  = length of range; min 1, max n\n"
                    "  >0 -> value\n"
                    "  -1 -> uniform distribution (big values)\n"
                    "  -2 -> lognormal distribution (medium values)\n"
                    "  -3 -> lognormal distribution (small values)\n"
                    "alg = algorithm (always LCA)\n"
                    "Options:\n"
                    "   --bs <block size>         block size for RTX_blocks (default: 2^15)\n"
                    "   --reps <repetitions>      RMQ repeats for the avg time (default: 10)\n"
                    "   --dev <device ID>         device ID (default: 0)\n"
                    "   --nt  <thread num>        number of CPU threads\n"
                    "   --seed <seed>             seed for PRNG\n"
                    "   --check                   check correctness\n"
                    "   --save-time=<file>        \n"
                    "   --save-power=<file>       \n"
                );
}


CmdArgs get_args(int argc, char *argv[]) {
    if (argc < 5) {
        print_help();
        exit(EXIT_FAILURE);
    }

    CmdArgs args;
    args.n = atoi(argv[1]);
    args.q = atoi(argv[2]);
    args.lr = atoi(argv[3]);
    args.alg = atoi(argv[4]);
    if (!args.n || !args.q || !args.lr) {
        print_help();
        exit(EXIT_FAILURE);
    }
    if (args.lr > args.n) {
        fprintf(stderr, "Error: lr=%i > n=%i  (lr must be between '1' and 'n')\n", args.lr, args.n);
        exit(EXIT_FAILURE);
    }

    args.bs = 1<<15;
    args.nb = args.n / args.bs;
    args.reps = 10;
    args.seed = time(0);
    args.dev = 0;
    args.check = 0;
    args.save_time = 0;
    args.save_power = 0;
    args.nt = 1;
    args.time_file = "";
    args.power_file = "";
    
    static struct option long_option[] = {
        // {name , has_arg, flag, val}
        {"bs", required_argument, 0, ARG_BS},
        {"nb", required_argument, 0, ARG_NB},
        {"reps", required_argument, 0, ARG_REPS},
        {"dev", required_argument, 0, ARG_DEV},
        {"nt", required_argument, 0, ARG_NT},
        {"seed", required_argument, 0, ARG_SEED},
        {"check", no_argument, 0, ARG_CHECK},
        {"save-time", optional_argument, 0, ARG_TIME},
        {"save-power", optional_argument, 0, ARG_POWER},
    };
    int opt, opt_idx;
    while ((opt = getopt_long(argc, argv, "12345", long_option, &opt_idx)) != -1) {
        if (isdigit(opt))
                continue;
        switch (opt) {
            case ARG_BS:
                args.bs = min(args.n, atoi(optarg));
                args.nb = args.n / args.bs;
                break;
            case ARG_NB:
                args.nb = min(args.n, atoi(optarg));
                args.bs = args.n / args.nb;
                break;
            case ARG_REPS:
                args.reps = atoi(optarg);
                break;
            case ARG_DEV:
                args.dev = atoi(optarg);
                break;
            case ARG_NT: 
                args.nt = atoi(optarg);
                break;
            case ARG_SEED:
                args.seed = atoi(optarg);
                break;
            case ARG_CHECK:
                args.check = 1;
                break;
            case ARG_TIME:
                args.save_time = 1;
                if (optarg != NULL)
                    args.time_file = optarg;
                break;
            case ARG_POWER:
                args.save_power = 1;
                if (optarg != NULL)
                    args.power_file = optarg;
                break;
            default:
                break;
        }
    }

    args.bs = 0;
    args.nb = 0;

    printf( "Params:\n"
            "   reps = %i\n"
            "   seed = %i\n"
            "   dev  = %i\n"
            AC_GREEN "   n    = %i (~%f GB, float)\n" AC_RESET
            "   bs   = %i\n"
            AC_GREEN "   q    = %i (~%f GB, int2)\n" AC_RESET
            "   lr   = %i\n"
            "   nt   = %i CPU threads\n"
            "   alg  = %i (%s)\n\n",
            args.reps, args.seed, args.dev, args.n, sizeof(float)*args.n/1e9, args.bs, args.q,
            sizeof(int2)*args.q/1e9, args.lr, args.nt, args.alg, "LCA");

    return args;
}

__global__ void kernel_setup_prng(int n, int seed, curandState *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    if(id <= n){
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void kernel_random_array(int n, curandState *state, float *array){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= n){ return; }
    float x = curand_uniform(&state[id]);
    //array[id] = x*1000.0f;
    array[id] = x;
}

std::pair<float*, curandState*> create_random_array_dev(int n, int seed){
    // cuRAND states
    curandState *devStates;
    cudaMalloc((void **)&devStates, n * sizeof(curandState));

    // data array
    float* darray;
    cudaMalloc(&darray, sizeof(float)*n);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_setup_prng<<<grid, block>>>(n, seed, devStates);
    cudaDeviceSynchronize();

    // gen random numbers
    kernel_random_array<<<grid,block>>>(n, devStates, darray);
    cudaDeviceSynchronize();

    return std::pair<float*, curandState*>(darray,devStates);
}

__global__ void kernel_random_array(int n, int max, int lr, curandState *state, int2 *array){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= n){ return; }
    int y = lr > 0 ? lr : curand_uniform(&state[id]) * (max/100);
    int x = curand_uniform_double(&state[id]) * (max-y-1);
    array[id].x = x;
    array[id].y = x+y;
}

std::pair<int2*, curandState*> create_random_array_dev2(int n, int max, int lr, int seed){
    // cuRAND states
    curandState *devStates;
    cudaMalloc((void **)&devStates, n * sizeof(curandState));

    // data array
    int2* darray;
    cudaMalloc(&darray, sizeof(int2)*n);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_setup_prng<<<grid, block>>>(n, seed, devStates);
    cudaDeviceSynchronize();

    // gen random numbers
    kernel_random_array<<<grid,block>>>(n, max, lr, devStates, darray);
    cudaDeviceSynchronize();

    return std::pair<int2*, curandState*>(darray,devStates);
}

__global__ void kernel_rmq_basic(int n, int q, float *x, int2 *rmq, float *out){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= q){
        return;
    }
    // solve the tid-th RMQ query in the x array of size n
    int l = rmq[tid].x;
    int r = rmq[tid].y;
    float min = x[l];
    float val;
    for(int i=l; i<=r; ++i){
        val = x[i]; 
        if(val < min){
            min = val;
        }
    }
    //printf("thread %i accessing out[%i] putting min %f\n", tid, tid, min);
    out[tid] = min;
}


bool is_equal(float a, float b) {
    float epsilon = 1e-4f;
    return abs(a - b) < epsilon;
}

bool check_result(float *hA, int2 *hQ, int q, float *expected, float *result){
    bool pass = true;
    for (int i = 0; i < q; ++i) {
        //if (expected[i] != result[i]) { // RT-cores don't introduce floating point errors
        if (!is_equal(expected[i], result[i])) {
            printf("Error on %i-th query: got %f, expected %f\n", i, result[i], expected[i]);
            printf("  [%i,%i]\n", hQ[i].x, hQ[i].y);
            pass = false;
            //for (int j = hQ[i].x; j <= hQ[i].y; ++j) {
            //    printf("%f ", hA[j]);
            //}
            //printf("\n");
            //return false;
        }
    }
    //for (int j = 0; j <= 1<<24; ++j) {
    //    printf("%f\n", hA[j]);
    //}
    return pass;
}

bool check_result(float *hA, int2 *hQ, int q, int *expected, int *result){
    for (int i = 0; i < q; ++i) {
        if (expected[i] != result[i]) {
            printf("Error on %i-th query: got %i, expected %i\n", i, result[i], expected[i]);
            //printf("[%i,%i]\n", hQ[i].x, hQ[i].y);
            //for (int j = hQ[i].x; j <= hQ[i].y; ++j) {
            //    printf("%f ", hA[j]);
            //}
            //printf("\n");
            //return false;
        }
    }
    return true;
}

bool check_parameters(int argc){
    if(argc < NUM_REQUIRED_ARGS){
        fprintf(stderr, AC_YELLOW "missing arguments\n" AC_RESET);
        print_help();
        return false;
    }
    return true;
}

void print_gpu_specs(int dev){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Memory:                       %f GB\n", prop.totalGlobalMem/(1024.0*1024.0*1024.0));
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %s\n", prop.concurrentKernels == 1? "yes" : "no");
    printf("  Memory Clock Rate:            %d MHz\n", prop.memoryClockRate);
    printf("  Memory Bus Width:             %d bits\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth:        %f GB/s\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}


// GPU RMQ basic approach
float* gpu_rmq_basic(int n, int q, float *devx, int2 *devrmq){
    dim3 block(BSIZE, 1, 1);
    dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
    float *hout, *dout;
    printf("Creating out array........................"); fflush(stdout);
    //Timer timer;
    hout = (float*)malloc(sizeof(float)*q);
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)*q));
    //printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    printf(AC_BOLDCYAN "Computing RMQs (%-11s).............." AC_RESET, "GPU BASE"); fflush(stdout);
    //timer.restart();
    kernel_rmq_basic<<<grid, block>>>(n, q, devx, devrmq, dout);
    CUDA_CHECK(cudaDeviceSynchronize());
    //timer.stop();
    //float timems = timer.get_elapsed_ms();
    //printf(AC_BOLDCYAN "done: %f secs: [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, (double)q/(timems/1000.0), (double)timems*1e6/q);
    printf("Copying result to host...................."); fflush(stdout);
    //timer.restart();
    CUDA_CHECK(cudaMemcpy(hout, dout, sizeof(float)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dout));
    //printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    //write_results(timems, q, 0);
    return hout;
}

void print_array(int n, float *m, const char* msg){
    printf("%s\n", msg);
    for(int i=0; i<n; ++i){
            printf("%.2f ", m[i]);
    }
    printf("\n");
}

void print_array(int n, int *m, const char* msg){
    printf("%s\n", msg);
    for(int i=0; i<n; ++i){
            printf("%4i ", m[i]);
    }
    printf("\n");
}

__global__ void print_gpu_array(int n, float *m){
    for(int i=0; i<n; ++i){
            printf("%.2f ", m[i]);
    }
    printf("\n");
}

__global__ void print_gpu_array(int n, int *m){
    for(int i=0; i<n; ++i){
            printf("%4i ", m[i]);
    }
    printf("\n");
}

void write_results(int dev, int alg, int n, int bs, int q, int lr, int reps, CmdArgs args) {
    if (!args.save_time) return;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    char *device = prop.name;

    FILE *fp;
    fp = fopen(args.time_file.c_str(), "a");
    fprintf(fp, "%s,%s,%i,%i,%i,%i,%i",
            device,
            "LCA",
            reps,
            n,
            bs,
            q,
            lr);
    fclose(fp);
}

/*
void write_results(float time_ms, int q, float construction_time, int reps) {
    if (!SAVE) return;
    float time_it = time_ms/reps;
    FILE *fp;
    fp = fopen(SAVE_FILE, "a");
    fprintf(fp, ",%f,%f,%f,%f\n",
            time_ms/1000.0,
            (double)q/(time_it/1000.0),
            (double)time_it*1e6/q,
            construction_time);
    fclose(fp);
}
*/

int gen_lr(int n, int lr, std::mt19937 gen) {
    if (lr > 0) {
        return lr;
    } else if (lr == -1) {
        std::uniform_int_distribution<int> dist(0, n-1);
        return dist(gen);
    } else if (lr == -2) {
        std::lognormal_distribution<double> dist(1.5, 1);
        int x = (int)(dist(gen) * pow(n, 0.7));
        //printf("after x  %i\n", x); fflush(stdout);
        while (x < 0 || x > n-1) {
            x = (int)(dist(gen) * pow(n, 0.7));
            //printf("in loop x  %i\n", x); fflush(stdout);
        }
        return x;
    } else if (lr == -3) {
        std::lognormal_distribution<double> dist(1.5, 1);
        int x = (int)(dist(gen) * pow(n, 0.3));
        while (x < 0 || x > n-1)
            x = (int)(dist(gen) * pow(n, 0.3));
        return x;
    }
    return 0;
}

int2* random_queries(int q, int lr, int n, int seed) {
    int2 *query = new int2[q];
    std::mt19937 gen(seed);

    for (int i = 0; i < q; ++i) {
        int qsize = gen_lr(n, lr, gen);
        //printf("qsize  %i\n", qsize); fflush(stdout);
        std::uniform_int_distribution<int> lrand(0, n - qsize-1);
        int l = lrand(gen);
        query[i].x = l;
        query[i].y = l + qsize;
    }
    return query;
}

void fill_queries_constant(int2 *query, int q, int lr, int n, int nt, int seed){
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(seed*tid);
        int chunk = (q+nt-1)/nt;
        int begin = chunk*tid;
        int end   = begin + chunk;
        int qsize = lr;
        for(int i=begin; i<q && i<end; ++i){
            std::uniform_int_distribution<int> lrand(0, n-1 - (qsize-1));
            int l = lrand(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
            //printf("thread %i (l,r) -> (%i, %i)\n\n", tid, query[i].x, query[i].y);
        }
    }
}

void fill_queries_uniform(int2 *query, int q, int lr, int n, int nt, int seed){
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(seed*tid);
        std::uniform_int_distribution<int> dist(1, n);
        int chunk = (q+nt-1)/nt;
        int begin = chunk*tid;
        int end   = begin + chunk;
        for(int i = begin; i<q && i<end; ++i){
            int qsize = dist(gen);
            std::uniform_int_distribution<int> lrand(0, n-1 - (qsize-1));
            int l = lrand(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
            //printf("(l,r) -> (%i, %i)\n\n", query[i].x, query[i].y);
        }
    }
}

void fill_queries_lognormal(int2 *query, int q, int lr, int n, int nt, int seed, int scale){
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(seed*tid);
        std::lognormal_distribution<double> dist(log(scale), 0.3);
        int chunk = (q+nt-1)/nt;
        int begin = chunk*tid;
        int end   = begin + chunk;
        //printf("fill_queries_lognormal: n=%i q=%i lr=%i  scale=%i\n", n, q, lr, scale);
        for(int i = begin; i<q && i<end; ++i){
            int qsize;
            do{ qsize = (int)dist(gen);  /*printf("dist gen! qsize=%i\n", qsize);*/ }
            while (qsize <= 0 || qsize > n);
            std::uniform_int_distribution<int> lrand(0, n-1 - (qsize-1));
            int l = lrand(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
            //printf("qsize=%i (l,r) -> (%i, %i)  thread %i\n\n", qsize, query[i].x, query[i].y, tid);
        }
    }
}

int2* random_queries_par_cpu(int q, int lr, int n, int nt, int seed) {
    omp_set_num_threads(nt);
    int2 *query = new int2[q];
    if(lr>0){
        fill_queries_constant(query, q, lr, n, nt, seed);
    }
    else if(lr == -1){
        fill_queries_uniform(query, q, lr, n, nt, seed);
    }
    else if(lr == -2){
        fill_queries_lognormal(query, q, lr, n, nt, seed, (int)pow((double)n,0.7));
    }
    else if(lr == -3){
        fill_queries_lognormal(query, q, lr, n, nt, seed, (int)pow((double)n,0.4));
    }
    else if(lr == -4){
        fill_queries_lognormal(query, q, lr, n, nt, seed, (int)max(1,n/(1<<8)));
    }
    else if(lr == -5){
        fill_queries_lognormal(query, q, lr, n, nt, seed, (int)max(1,n/(1<<15)));
    }
    return query;
}


