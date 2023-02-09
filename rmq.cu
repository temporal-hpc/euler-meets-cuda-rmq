#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <iomanip>
#include <fstream>
#include <stdio.h>

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

#include "lca.h"
#include "tree.h"
#include "utils.h"
#include "timer.hpp"

#include <chrono>
#include <thread>

#define BSIZE 1024

#include "rmq_helper.cuh"


using namespace std;
using namespace emc;


int build_tree(int *out_parents, int *out_idx, float *out_val, float *a, int idx, int p, int l, int r) {
	//printf("%i %i\n", l, r);
	if (l > r)
		return idx;

	idx++;
	int min_arg = l;
	for (int i = l+1; i <= r; ++i) {
		if  (a[i] < a[min_arg])
			min_arg = i;
	}
	out_parents[idx] = p;
	out_val[idx] = a[min_arg];
	out_idx[min_arg] = idx;
	//printf("    %i %.2f %i\n", idx, a[min_arg], p);

	int new_idx = build_tree(out_parents, out_idx, out_val, a, idx, idx, l, min_arg-1);

	new_idx = build_tree(out_parents, out_idx, out_val, a, new_idx, idx, min_arg+1, r);

	return new_idx;
}

void build_tree(int *out_parents, int *out_idx, float *out_val, float *a, int n) {
    build_tree(out_parents, out_idx, out_val, a, -1, -1, 0, n-1);
}

__global__ void transform_queries(int *Q_lca, int2 *Q_rmq, int n, int *idx) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;

  int l = Q_rmq[tid].x;
  int r = Q_rmq[tid].y;
  Q_lca[2*tid] = idx[l];
  Q_lca[2*tid+1] = idx[r];
}

__global__ void get_vals(float* rmq_ans, int* lca_ans, int n, float *vals) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;

  //printf("tid: %i  lca: %i\n", tid, lca_ans[tid]);
  rmq_ans[tid] = vals[lca_ans[tid]];
}

int main(int argc, char *argv[]) {
  std::ios_base::sync_with_stdio(false);

  if(!check_parameters(argc)){
      exit(EXIT_FAILURE);
  }
  int seed = atoi(argv[1]);
  int dev = atoi(argv[2]);
  int n = atoi(argv[3]);
  int bs = atoi(argv[4]);
  int q = atoi(argv[5]);
  int lr = atoi(argv[6]);
  int nt = atoi(argv[7]);
  int alg = atoi(argv[8]);
  if (lr >= n) {
      fprintf(stderr, "Error: lr can not be bigger than n\n");
      return -1;
  }

  printf( "Params:\n"
          "   seed = %i\n"
          "   dev = %i\n"
          AC_GREEN "   n   = %i (~%f GB, float)\n" AC_RESET
          "   bs = %i\n"
          AC_GREEN "   q   = %i (~%f GB, int2)\n" AC_RESET
          "   lr  = %i\n"
          "   nt  = %i CPU threads\n"
          "   alg = %i (%s)\n\n",
          seed, dev, n, sizeof(float)*n/1e9, bs, q, sizeof(int2)*q/1e9, lr, nt, alg, "LCA");
  cudaSetDevice(dev);
  print_gpu_specs(dev);

  // 1) data on GPU, result has the resulting array and the states array
  //Timer timer;
  //printf(AC_YELLOW "Generating n=%i values..............", n); fflush(stdout);
  //std::pair<float*, curandState*> p = create_random_array_dev(n, seed);
  //printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
  //timer.restart();
  //printf(AC_YELLOW "Generating q=%i queries.............", q); fflush(stdout);
  //std::pair<int2*, curandState*> qs = create_random_array_dev2(q, n, lr, seed+7); //TODO use previous states
  //printf("done: %f secs\n" AC_RESET, timer.get_elapsed_ms()/1000.0f);

  // gen array
  printf("Creating arrays\n");
  float *a = new float[n];
  int2 *hq = new int2[q];
  //cudaMemcpy(a, p.first, sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(hq, qs.first, sizeof(int2), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i)
    a[i] = (rand() % 1000) / 1000.0f;
  for (int i = 0; i < q; ++i) {
    int length = lr > 0 ? lr : rand() % (n/100);
    int l = rand() % (n - length-1);
    hq[i].x = l;
    hq[i].y = l + length;
  }
  int2 *Q_rmq;
  CUDA_CHECK( cudaMalloc(&Q_rmq, sizeof(int2)*q) );
  CUDA_CHECK( cudaMemcpy(Q_rmq, hq, sizeof(int2)*q, cudaMemcpyHostToDevice) ); 

  int *parents = new int[n];
  int *idx = new int[n];
  float *vals = new float[n];
  
  // build cartesian tree
  printf("Building cartesian tree\n");
  build_tree(parents, idx, vals, a, n);

  //print_array(n, a, "A:");
  //print_array(n, parents, "Parents:");
  //print_array(n, idx, "Idx:");
  //print_array(n, vals, "Vals:");

  int *d_parents, *d_idx;
  float *d_vals;
  CUDA_CHECK( cudaMalloc(&d_parents, sizeof(int)*n) );
  cudaMalloc(&d_idx, sizeof(int)*n);
  cudaMalloc(&d_vals, sizeof(float)*n);
  CUDA_CHECK( cudaMemcpy(d_parents, parents, sizeof(int)*n, cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(d_idx, idx, sizeof(int)*n, cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(d_vals, vals, sizeof(float)*n, cudaMemcpyHostToDevice) );

  //print_gpu_array<<<1,1>>>(n, d_parents);
  //print_gpu_array<<<1,1>>>(n, d_idx);
  //print_gpu_array<<<1,1>>>(n, d_vals);
  CUDA_CHECK( cudaDeviceSynchronize() );

  //transform queries
  int *Q_lca;
  CUDA_CHECK( cudaMalloc(&Q_lca, sizeof(int)*2*q) );

  int grid = (q + BSIZE - 1) / BSIZE;
  transform_queries<<<grid, BSIZE>>>(Q_lca, Q_rmq, q, d_idx);
  //print_gpu_array<<<1,1>>>(2*q, Q_lca);
  CUDA_CHECK( cudaDeviceSynchronize() );


  // solve LCA
  printf("Solving LCA\n");
  mgpu::standard_context_t context(0);
  int *d_answers; 
  CUDA_CHECK( cudaMalloc(&d_answers, sizeof(int)*q) );
  //void cuda_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers, int batchSize,
  //                    mgpu::context_t &context) {
  // use batchsize = to num of queries
  cuda_lca_inlabel(n, d_parents, q, Q_lca, d_answers, q, context);
  CUDA_CHECK( cudaDeviceSynchronize() );

  //print_gpu_array<<<1,1>>>(q, d_answers);

  float *rmq_ans;
  CUDA_CHECK (cudaMalloc(&rmq_ans, sizeof(float)*q) );
  get_vals<<<grid, BSIZE>>>(rmq_ans, d_answers, q, d_vals);
  CUDA_CHECK( cudaDeviceSynchronize() );

  float *out = new float[q];
  CUDA_CHECK( cudaMemcpy(out, rmq_ans, sizeof(float)*q, cudaMemcpyDeviceToHost) );


  int *lca_queries = new int[2*q];
  CUDA_CHECK( cudaMemcpy(lca_queries, Q_lca, sizeof(float)*q*2, cudaMemcpyDeviceToHost) );
  //for (int i = 0; i < q; ++i)
    //printf("RMQ query: (%i, %i)   LCA query: (%i, %i)   ans: %f\n", hq[i].x, hq[i].y, lca_queries[2*i], lca_queries[2*i+1], out[i]);

  printf("\nchecking result:\n");
  float *d_a;
  cudaMalloc(&d_a, sizeof(float)*n);
  cudaMemcpy(d_a, a, sizeof(float)*n, cudaMemcpyHostToDevice);
  float *expected = gpu_rmq_basic(n, q, d_a, Q_rmq);
  printf(AC_YELLOW "\nchecking result..........................." AC_YELLOW); fflush(stdout);
  int pass = check_result(a, hq, q, expected, out);
  printf(AC_YELLOW "%s\n" AC_RESET, pass ? "pass" : "failed");


  printf("Benchmark Finished\n");
  return 0;
}
