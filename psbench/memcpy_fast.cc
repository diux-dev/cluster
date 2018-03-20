// To run
// g++ -std=c++0x memcpy_fast.cc -pthread -march=native -O6 -o memcpy_fast
// numactl --cpunodebind 0 --membind 0 ./memcpy_fast 100 1
//
//
// c5.18xlarge
// Stream copy 1 threads: 15.0 ms, 6.65 GB/sec
// Stream copy 2 threads: 8.0 ms, 12.47 GB/sec
// Stream copy 4 threads: 5.0 ms, 19.81 GB/sec
// Stream copy 8 threads: 3.2 ms, 31.71 GB/sec
// Stream copy 16 threads: 2.6 ms, 37.78 GB/sec

//
// p3.16xlarge
// Stream copy 1 threads: 11.8 ms, 8.51 GB/sec
// Stream copy 2 threads: 6.0 ms, 16.78 GB/sec
// Stream copy 4 threads: 3.8 ms, 26.30 GB/sec
// Stream copy 8 threads: 3.8 ms, 26.16 GB/sec
// Stream copy 16 threads: 4.0 ms, 24.74 GB/sec
//
// borrowed from https://stackoverflow.com/a/44948720/419116
//




#include <immintrin.h>
#include <cstdint>
#include <stdlib.h>
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>

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

/* ... */
void fastMemcpy(void *pvDest, void *pvSrc, size_t nBytes) {
  assert(nBytes % 32 == 0);
  assert((intptr_t(pvDest) & 31) == 0);
  assert((intptr_t(pvSrc) & 31) == 0);
  const __m256i *pSrc = reinterpret_cast<const __m256i*>(pvSrc);
  __m256i *pDest = reinterpret_cast<__m256i*>(pvDest);
  int64_t nVects = nBytes / sizeof(*pSrc);
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256i loaded = _mm256_stream_load_si256(pSrc);
    _mm256_stream_si256(pDest, loaded);
  }
  _mm_sfence();
}

void AsyncStreamCopy(__m256i *pDest, const __m256i *pSrc, int64_t nVects) {
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256i loaded = _mm256_stream_load_si256(pSrc);
    _mm256_stream_si256(pDest, loaded);
  }
}

void BenchmarkMultithreadStreamCopy(double *gpdOutput, const double *gpdInput, const int64_t cnDoubles, int maxThreads) {
  // std::cout << "cnDoubles " << cnDoubles << "\n";
  // std::cout << "sizeof(double) " << sizeof(double) << "\n";
  // std::cout << "sizeof(__m256i) " << sizeof(__m256i) << "\n";
  // std::cout << "offset " << ((cnDoubles * sizeof(double)) % sizeof(__m256i)) << "\n";
  assert((cnDoubles * sizeof(double)) % sizeof(__m256i) == 0);
  //  const uint32_t maxThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> thrs;
  thrs.reserve(maxThreads + 1);

  const __m256i *pSrc = reinterpret_cast<const __m256i*>(gpdInput);
  __m256i *pDest = reinterpret_cast<__m256i*>(gpdOutput);
  const int64_t nVects = cnDoubles * sizeof(*gpdInput) / sizeof(*pSrc);

  for (uint32_t nThreads = maxThreads; nThreads <= maxThreads; nThreads++) {
    auto start = std::chrono::high_resolution_clock::now();
    div_t perWorker = div((long long)nVects, (long long)nThreads);
    int64_t nextStart = 0;
    for (uint32_t i = 0; i < nThreads; i++) {
      const int64_t curStart = nextStart;
      nextStart += perWorker.quot;
      if ((long long)i < perWorker.rem) {
	nextStart++;
      }
      thrs.emplace_back(AsyncStreamCopy, pDest + curStart, pSrc+curStart, nextStart-curStart);
    }
    for (uint32_t i = 0; i < nThreads; i++) {
      thrs[i].join();
    }
    _mm_sfence();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("Stream copy %d threads: %.1f ms, %.2lf GB/sec\n", (int)nThreads, nSec*1000, cnDoubles * sizeof(double) / nSec/1000/1000/1000);

    thrs.clear();
  }
};

int main(int argc, char *argv[]) {
  printf("Hello world\n");
  int size = 100*1000*1000; // 1000 MB
  double* s1 = (double *) aligned_malloc(size);
  double* s2 = (double *) aligned_malloc(size);

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

  //  const __m256i loaded = _mm256_stream_load_si256(s2_);
  //  _mm256_stream_si256(s1_, loaded);

  //  AsyncStreamCopy(s1_, s2_, 64);
  for (int i = 0; i < iters; ++i) {
    BenchmarkMultithreadStreamCopy(s1, s2, size/sizeof(double), num_threads);
  };
  
  return 0;
}
