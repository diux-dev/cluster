// To run
// g++ -std=c++0x memcpy_slow.cc -pthread -march=native -O6
// ./a.out <num_iters> <num_threads>
//
// c5
// numactl --cpunodebind 0 --membind 0 ./memcpy_classic 100 
// memcpy: 23.0 ms, 4.35 GB/sec
// memcpy: 23.0 ms, 4.36 GB/sec
// memcpy: 23.0 ms, 4.35 GB/sec

// p3
// numactl --cpunodebind 0 --membind 0 ./memcpy_classic 100
// memcpy: 12.8 ms, 7.82 GB/sec
// memcpy: 12.7 ms, 7.84 GB/sec
// memcpy: 12.8 ms, 7.84 GB/sec







#include <immintrin.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>
#include <cstring>

using namespace std;

#define ALIGN 64

void *aligned_malloc(int size) {
    void *mem = malloc(size+ALIGN+sizeof(void*));
    void **ptr = (void**)((uintptr_t)(mem+ALIGN+sizeof(void*)) & ~(ALIGN-1));
    ptr[-1] = mem;
    return ptr;
}

void aligned_free(void *ptr) {
    free(((void**)ptr)[-1]);
}

//using namespace std;

int main(int argc, char *argv[]) {
  printf("Hello world\n");
  int size = 100*1000*1000; // 1000 MB
  double* s1 = (double *) aligned_malloc(size);
  double* s2 = (double *) aligned_malloc(size);
  memcpy(s1, s2, 1);

  int iters = 100;
  int num_threads = 1;
  if (argc > 1) {
    iters = atoi(argv[1]);
  }
  
  if (argc > 2) {
    num_threads = atoi(argv[2]);
  }
  
  for (int i = 0; i < size/sizeof(double); i++) {
    s1[i] = 0;
    s2[i] = 0;
  }

  __m256i* s1_ = reinterpret_cast<__m256i*>(s1);
  __m256i* s2_ = reinterpret_cast<__m256i*>(s2);


  for (int i = 0; i < iters; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    memcpy(s1, s2, size);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("memcpy: %.1f ms, %.2lf GB/sec\n", nSec*1000, size / nSec/1000/1000/1000);
  };
  
  return 0;
}
