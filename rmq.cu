#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <stack>
#include <tuple>

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

#define SAVE 0
#define MEASURE_POWER 0
#define CHECK 1

#include "rmq_helper.cuh"

#include "lca.h"
#include "tree.h"
#include "utils.h"
#include "timer.hpp"

#include <chrono>
#include <thread>



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

struct Node {
	Node *parent;
	Node *left;
	Node *right;
	float val;
	int idx;
};

void insert_val(Node *&r, Node *&b, float val, int idx) {
        //printf("%i  %.2f\n", idx, val);
	Node *q = new Node;
	q->idx = idx;
	q->val = val;
        q->parent = nullptr;
        q->left = nullptr;
	q->right = nullptr;

	if (b == nullptr) {
		q->parent = nullptr;
		q->left = nullptr;
		r = q;
		b = q;
		return;
	}

	while (b->parent != nullptr && b->parent->val > val) {
		b = b->parent;
	}
	if (b->val < val) {
		q->parent = b;
		b->right = q;
		b = q;
	} else if (b->parent == nullptr) {
		q->left = b;
		b->parent = q;
		r = q;
		b = q;
	} else {
		q->parent = b->parent;
		q->left = b;
		b->parent->right = q;
		b->parent = q;
		b = q;
	}
	return;
}

void tree_to_array(int *out_parents, int *out_idx, float *out_val, Node *r) {
  int cont = 0;
  std::stack<std::pair<Node*, int>> *node_stack = new std::stack<std::pair<Node*, int>>();
  node_stack->push({r,-1});
  while (!node_stack->empty()) {
    auto pc = node_stack->top();
    node_stack->pop();
    Node *p = pc.first;
    int parent = pc.second;
    //printf("%.2f  %i  %i\n", p->val, p->idx, parent);
    out_parents[cont] = parent;
    out_val[cont] = p->val;
    out_idx[p->idx] = cont;
    if (p->left != nullptr)
      node_stack->push({p->left, cont});
    if (p->right != nullptr)
      node_stack->push({p->right, cont});
    cont++;
  }

  return;
}

void print_tree(Node *r, int depth) {
	int indent = 4;
	if (r->right != nullptr)
		print_tree(r->right, depth+1);

	for (int i = 0; i < indent*depth; ++i)
		printf(" ");
	printf("%.2f  %i\n", r->val, r->idx);

	if (r->left != nullptr)
		print_tree(r->left, depth+1);
}

void build_tree_online(int *out_parents, int *out_idx, float *out_val, float *a, int n) {
	Node *r = nullptr;
	Node *b = r;
	//printf("Building tree\n");
	for (int i = 0; i < n; ++i) {
		insert_val(r, b, a[i], i);
            //print_tree(r, 0);
	}
	//printf("Tree built\n");
	//print_tree(r, 0);
	tree_to_array(out_parents, out_idx, out_val, r);
	//printf("Array built\n"); fflush(stdout);
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
  int reps = atoi(argv[1]);
  int seed = atoi(argv[2]);
  int dev = atoi(argv[3]);
  int n = atoi(argv[4]);
  int bs = atoi(argv[5]);
  int q = atoi(argv[6]);
  int lr = atoi(argv[7]);
  int nt = atoi(argv[8]);
  int alg = atoi(argv[9]);
  if (lr >= n) {
      fprintf(stderr, "Error: lr can not be bigger than n\n");
      return -1;
  }

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
          reps, seed, dev, n, sizeof(float)*n/1e9, bs, q, sizeof(int2)*q/1e9, lr, nt, alg, "LCA");
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
  srand(seed);
  double t1,t2;
  printf("Creating arrays............"); fflush(stdout);
  t1 = omp_get_wtime();
  float *a = new float[n];
  int2 *hq;
  //cudaMemcpy(a, p.first, sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(hq, qs.first, sizeof(int2), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) {
    a[i] = (float)rand() / (float)RAND_MAX;
    //a[i] = (float)(n-1-i) / (float)n;
  }
  hq = random_queries(q, lr, n, seed);
  int2 *Q_rmq;
  CUDA_CHECK( cudaMalloc(&Q_rmq, sizeof(int2)*q) );
  CUDA_CHECK( cudaMemcpy(Q_rmq, hq, sizeof(int2)*q, cudaMemcpyHostToDevice) ); 
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);

  int *parents = new int[n];
  int *idx = new int[n];
  float *vals = new float[n];
  
  // build cartesian tree
  printf("Building cartesian tree...."); fflush(stdout);
  t1 = omp_get_wtime();
  build_tree_online(parents, idx, vals, a, n);
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);

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


  write_results(dev, alg, n, bs, q, lr, reps);

  // solve LCA
  printf("Solving LCA:\n"); fflush(stdout);
  mgpu::standard_context_t context(0);
  int *d_answers; 
  CUDA_CHECK( cudaMalloc(&d_answers, sizeof(int)*q) );
  //void cuda_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers, int batchSize,
  //                    mgpu::context_t &context) {
  // use batchsize = to num of queries
  t1 = omp_get_wtime();
  cuda_lca_inlabel(n, d_parents, q, Q_lca, d_answers, q, context, reps, SAVE, MEASURE_POWER, dev);
  CUDA_CHECK( cudaDeviceSynchronize() );
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);

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

  if (CHECK) {
    printf("\nchecking result:\n");
    float *d_a;
    cudaMalloc(&d_a, sizeof(float)*n);
    cudaMemcpy(d_a, a, sizeof(float)*n, cudaMemcpyHostToDevice);
    float *expected = gpu_rmq_basic(n, q, d_a, Q_rmq);
    printf(AC_YELLOW "\nchecking result..........................." AC_YELLOW); fflush(stdout);
    int pass = check_result(a, hq, q, expected, out);
    printf(AC_YELLOW "%s\n" AC_RESET, pass ? "pass" : "failed");
  }


  printf("Benchmark Finished\n");
  return 0;
}
