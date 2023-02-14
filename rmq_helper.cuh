#pragma once

#include <curand_kernel.h>
#include <unistd.h>

#define SAVE_FILE "data_rmq/data.csv"

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


#define NUM_REQUIRED_ARGS 10
void print_help(){
    fprintf(stderr, AC_BOLDGREEN "run as ./rtxrmq <rep> <seed> <dev> <n> <bs> <q> <lr> <nt> <alg>\n\n" AC_RESET
                    "rep  = repetitions for avg time"
                    "seed = seed for PRNG\n"
                    "dev  = device ID\n"
                    "n    = num elements\n"
                    "bs   = block size for RTX_blocks\n"
                    "q    = num RMQ querys\n"
                    "lr   = size of range (-1: randomized)\n"
                    "nt   = num CPU threads\n"
                    "alg  = algorithm\n"
                );
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

void write_results(int dev, int alg, int n, int bs, int q, int lr, int reps) {
    if (!SAVE) return;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    char *device = prop.name;

    FILE *fp;
    fp = fopen(SAVE_FILE, "a");
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
