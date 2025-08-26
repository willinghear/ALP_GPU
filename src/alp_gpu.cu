/*
 * ============================================
 *  ALP-GPU 压缩/解压（优化版）
 *  主要优化：
 *    1) 并行化所有kernel，移除强制串行化
 *    2) 两阶段压缩策略：先计算元数据，再并行写入
 *    3) 使用共享内存和高效归约
 *    4) 每个线程独立处理向量，提高GPU利用率
 * ============================================
 */

#include "alp_gpu.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <climits>
#include <cstdlib>
#include <vector>

using std::uint8_t; using std::uint32_t; using std::uint64_t;

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << __FILE__ << ":" << __LINE__ \
                      << " " << cudaGetErrorString(error) << std::endl; \
        } \
    } while(0)

namespace alp_gpu {

// ===================== 两级采样配置 =====================
namespace sampling_config {
    static constexpr int ROWGROUP_SIZE = 100000;
    static constexpr int ROWGROUP_VECTOR_SAMPLES = 8;
    static constexpr int SAMPLES_PER_VECTOR = 32;
    static constexpr int MAX_K_COMBINATIONS = 5;
    static constexpr int EARLY_EXIT_THRESHOLD = 2;
}

template<typename T> struct SamplingConstants {
    static constexpr size_t RD_SIZE_THRESHOLD_LIMIT = 
        sizeof(T) == 8 ? (48 * sampling_config::SAMPLES_PER_VECTOR) 
                       : (22 * sampling_config::SAMPLES_PER_VECTOR);
};

// ===================== 常量 =====================
__device__ __constant__ double D_EXP_ARR[19] = {
  1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
  100000000.0, 1000000000.0, 10000000000.0, 100000000000.0, 1000000000000.0,
  10000000000000.0, 100000000000000.0, 1000000000000000.0,
  10000000000000000.0, 100000000000000000.0, 1000000000000000000.0
};
__device__ __constant__ double D_FRAC_ARR[20] = {
  1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
  0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.000000000001,
  0.0000000000001, 0.00000000000001, 0.000000000000001, 0.0000000000000001,
  0.00000000000000001, 0.000000000000000001
};

static constexpr int   DICT_BW  = 3;
static constexpr int   DICT_SZ  = 1 << DICT_BW;
static constexpr int   CUT_LIM  = 16;
static constexpr int   MAX_VEC  = 4096;

// ===================== 安全的设备端位流 Writer/Reader =====================
struct SafeBitWriter {
    uint8_t* buf;
    uint64_t bitpos;
    uint64_t max_bits;  // 最大可写位数
    
    __device__ SafeBitWriter(uint8_t* buffer, uint64_t start_bit, uint64_t buffer_size_bits)
        : buf(buffer), bitpos(start_bit), max_bits(start_bit + buffer_size_bits) {}
    
    __device__ bool put1(int b){
        if (bitpos >= max_bits) {
            printf("[DEVICE-ERROR] BitWriter overflow at bit %llu (max %llu)\n", bitpos, max_bits);
            return false;
        }
        if (!b) { ++bitpos; return true; }
        
        uint64_t byte = bitpos >> 3;
        int off = 7 - int(bitpos & 7ULL);
        
        // 方法1：使用简单的非原子写入（如果每个向量由单独的线程处理，不会有竞争）
        buf[byte] |= (uint8_t(1u) << off);
        
        // 方法2：如果确实需要原子操作，使用32位对齐的原子操作
        // uint32_t* aligned = (uint32_t*)(buf + (byte & ~3ULL));
        // int byte_offset = byte & 3;
        // int bit_offset = byte_offset * 8 + off;
        // atomicOr(aligned, uint32_t(1u) << bit_offset);
        
        ++bitpos;
        return true;
    }
    
    __device__ bool putN(uint64_t v, int bits){
        if (bits <= 0 || bits > 64) return false;
        if (bitpos + bits > max_bits) {
            printf("[DEVICE-ERROR] BitWriter would overflow: pos=%llu + bits=%d > max=%llu\n", 
                   bitpos, bits, max_bits);
            return false;
        }
        for(int i=bits-1;i>=0;--i) {
            if (!put1( (v>>i) & 1ULL )) return false;
        }
        return true;
    }
    
    __device__ uint64_t get_pos() const { return bitpos; }
    __device__ uint64_t remaining_bits() const { return max_bits - bitpos; }
};

struct BitReader {
    const uint8_t* buf;
    uint64_t bitpos;
    __device__ int get1(){
        uint64_t byte = bitpos >> 3;
        int off = 7 - int(bitpos & 7ULL);
        int b = (buf[byte] >> off) & 1;
        ++bitpos; return b;
    }
    __device__ uint64_t getN(int bits){
        uint64_t v=0;
        for(int i=0;i<bits;++i){ v = (v<<1) | get1(); }
        return v;
    }
    __device__ uint64_t get_pos() const { return bitpos; }
};

// ===================== 工具函数 =====================
__device__ __forceinline__ int width_needed_unsigned(unsigned long long range){
    if (range==0ULL) return 1;
    int c=0; while(range){ ++c; range>>=1ULL; } return c;
}

__device__ inline long long fast_round_double(double x){
    const double SWEET = double((1ULL<<51) + (1ULL<<52));
    return (long long)(x + SWEET) - (long long)SWEET;
}

__device__ inline uint32_t mask_lo(int bits){
    return (bits >= 32) ? 0xFFFFFFFFu : ((1u<<bits) - 1u);
}

// ===================== ALP 判断与分析 =====================
template<typename T>
__device__ inline bool alp_exact_equal(T v, uint8_t e, uint8_t f){
    if constexpr (std::is_same_v<T,double>) {
        double enc = v * D_EXP_ARR[e] * D_FRAC_ARR[f];
        long long I = fast_round_double(enc);
        double dec = double(I) * (1.0 / D_FRAC_ARR[f]) * D_FRAC_ARR[e];
        return dec==v;
    } else {
        float enc = v * float(D_EXP_ARR[e]) * float(D_FRAC_ARR[f]);
        int   I   = __float2int_rn(enc);
        float dec = float(I) * (1.0f/float(D_FRAC_ARR[f])) * float(D_FRAC_ARR[e]);
        return dec==v;
    }
}

template<typename T>
__device__ inline void alp_vector_analyze(const T* v, int n, uint8_t e, uint8_t f,
                                          short& bitw, long long& FOR,
                                          int& exc_cnt){
    long long mn=LLONG_MAX, mx=LLONG_MIN;
    exc_cnt=0;
    for(int i=0;i<n;++i){
        double enc = double(v[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
        long long I = fast_round_double(enc);
        double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
        if (dec==double(v[i])) { mn=(mn<I?mn:I); mx=(mx>I?mx:I); }
        else ++exc_cnt;
    }
    unsigned long long range = (mn==LLONG_MAX)? 0ULL : (unsigned long long)(mx - mn);
    bitw = (short)width_needed_unsigned(range);
    FOR  = (mn==LLONG_MAX?0:mn);
}

template<typename T>
__device__ inline void alp_vector_choose_best_bits(
    const T* v, int n,
    uint8_t& best_e, uint8_t& best_f,
    short& bitw, long long& FOR, int& exc)
{
    const int val_bits = std::is_same_v<T,double> ? 64 : 32;
    double best_score = 1e300;
    best_e=0; best_f=0; bitw=0; FOR=0; exc=0;

    for(uint8_t e=0;e<=18;++e){
        for(uint8_t f=0;f<=e;++f){
            short _bw; long long _FOR; int _exc;
            alp_vector_analyze<T>(v, n, e, f, _bw, _FOR, _exc);
            double score = double(n)*_bw + double(_exc)*(val_bits + 16);
            if (score < best_score){
                best_score = score;
                best_e = e; best_f = f; bitw = _bw; FOR = _FOR; exc = _exc;
            }
        }
    }
}

template<typename T>
__device__ inline uint64_t alp_vector_size_bits_safe(int n, uint8_t e, uint8_t f,
                                                     short bitw, int exc_cnt){
    const int val_bits = std::is_same_v<T,double> ? 64 : 32;
    return 145ULL + uint64_t(n) * uint64_t(bitw) + uint64_t(exc_cnt) * (val_bits + 16);
}

template<typename T>
__device__ inline bool alp_vector_write_safe(SafeBitWriter& bw, const T* v, int n,
                                            uint8_t e, uint8_t f, short bitw, long long FOR){
    assert(n <= MAX_VEC);

    if (!bw.put1(1)) return false; // useALP = 1
    if (!bw.putN((uint64_t)e, 8)) return false;
    if (!bw.putN((uint64_t)f, 8)) return false;
    if (!bw.putN((uint64_t)bitw, 16)) return false;
    if (!bw.putN((uint64_t)FOR, 64)) return false;
    if (!bw.putN((uint64_t)n, 32)) return false;

    int exc_cnt=0;
    int      exc_pos[MAX_VEC];
    uint64_t exc_val[MAX_VEC];

    for(int i=0;i<n;++i){
        double enc = double(v[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
        long long I = fast_round_double(enc);
        double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
        if (dec==double(v[i])) {
            uint64_t packed = (uint64_t)(I - FOR);
            if (!bw.putN(packed, bitw)) return false;
        } else {
            if (!bw.putN(0, bitw)) return false;
            if constexpr (std::is_same_v<T,double>) {
                uint64_t raw = *reinterpret_cast<const uint64_t*>(&v[i]);
                exc_val[exc_cnt] = raw;
            } else {
                uint32_t raw = *reinterpret_cast<const uint32_t*>(&v[i]);
                exc_val[exc_cnt] = raw;
            }
            exc_pos[exc_cnt] = i;
            ++exc_cnt;
        }
    }
    if (!bw.putN((uint64_t)exc_cnt, 16)) return false;
    for(int k=0;k<exc_cnt;++k){
        if constexpr (std::is_same_v<T,double>) {
            if (!bw.putN(exc_val[k], 64)) return false;
        } else {
            if (!bw.putN(exc_val[k], 32)) return false;
        }
        if (!bw.putN((uint64_t)exc_pos[k], 16)) return false;
    }
    return true;
}

// ===================== ALPrd 结构与函数 =====================
template<typename T> struct ALPrdDict {
    uint8_t rightBW;
    uint8_t leftBW;
    uint32_t dict[DICT_SZ];
};

template<typename T>
__device__ inline void alprd_find_best(const uint64_t* in, int n, ALPrdDict<T>& D){
    double best_score = 1e100; int best_rbw = int(sizeof(T)*8) - 1;
    uint32_t best_dict[DICT_SZ] = {0};

    for(int lbw=1; lbw<=CUT_LIM; ++lbw){
        int rbw = int(sizeof(T)*8) - lbw;
        uint32_t lmask = mask_lo(lbw);

        uint32_t uniq_left[MAX_VEC]; int cnt[MAX_VEC];
        int u = 0;

        for(int i=0;i<n;++i){
            uint32_t left = (uint32_t)((in[i] >> rbw) & lmask);
            int j=0; for(; j<u; ++j) if (uniq_left[j]==left) { ++cnt[j]; break; }
            if (j==u){ uniq_left[u]=left; cnt[u]=1; ++u; }
        }
        
        uint32_t dict[DICT_SZ]={0};
        int used = (DICT_SZ < u ? DICT_SZ : u);
        for(int k=0;k<used;++k){
            int best=-1, id=-1;
            for(int j=0;j<u;++j){
                bool taken=false;
                for(int t=0;t<k;++t) if (dict[t]==uniq_left[j]) { taken=true; break; }
                if (taken) continue;
                if (cnt[j]>best){ best=cnt[j]; id=j; }
            }
            dict[k] = uniq_left[id];
        }
        
        int keep=0;
        for(int k=0;k<used;++k){
            for(int j=0;j<u;++j) if (uniq_left[j]==dict[k]) { keep += cnt[j]; break; }
        }
        int exc = n - keep;

        double bits = 1 + 32 + 8 + double(n)*(DICT_BW + rbw) + double(DICT_SZ)*lbw + 16.0*exc + double(lbw)*exc;

        if (bits < best_score){
            best_score = bits;
            best_rbw   = rbw;
            for(int k=0;k<DICT_SZ;++k) best_dict[k]=dict[k];
        }
    }
    D.rightBW = (uint8_t)best_rbw;
    D.leftBW  = (uint8_t)(int(sizeof(T)*8) - best_rbw);
    for(int k=0;k<DICT_SZ;++k) D.dict[k]=best_dict[k];
}

template<typename T>
__device__ inline uint64_t alprd_vector_size_bits_safe(int n, const ALPrdDict<T>& D, int exc_cnt){
    uint64_t base = 57ULL + uint64_t(n)*(DICT_BW + D.rightBW) + DICT_SZ*D.leftBW + uint64_t(exc_cnt)*(D.leftBW+16);
    return base;
}

template<typename T>
__device__ inline bool alprd_vector_write_safe(SafeBitWriter& bw, const uint64_t* in, int n,
                                              const ALPrdDict<T>& D){
    assert(n <= MAX_VEC);
    
    if (!bw.put1(0)) return false; // useALP=0
    if (!bw.putN((uint64_t)n, 32)) return false;
    if (!bw.putN((uint64_t)D.rightBW, 8)) return false;

    int exc_cnt=0; uint16_t exc_pos[MAX_VEC]; uint32_t exc_left[MAX_VEC];
    uint64_t right_mask = (D.rightBW==64)? ~0ULL : ((1ULL<<D.rightBW)-1ULL);
    uint32_t left_mask  = mask_lo(D.leftBW);

    for(int i=0;i<n;++i){
        uint64_t right = in[i] & right_mask;
        uint32_t left  = (uint32_t)((in[i] >> D.rightBW) & left_mask);
        short idx = DICT_SZ;
        for(int k=0;k<DICT_SZ;++k){ if (D.dict[k]==left){ idx=(short)k; break; } }
        if (idx<DICT_SZ){
            if (!bw.putN((uint64_t)idx, DICT_BW)) return false;
            if (!bw.putN(right, D.rightBW)) return false;
        }else{
            if (!bw.putN(0, DICT_BW)) return false;
            if (!bw.putN(right, D.rightBW)) return false;
            exc_pos[exc_cnt]  = (uint16_t)i;
            exc_left[exc_cnt] = left;
            ++exc_cnt;
        }
    }
    
    for(int k=0;k<DICT_SZ;++k) {
        if (!bw.putN((uint64_t)D.dict[k], D.leftBW)) return false;
    }
    
    if (!bw.putN((uint64_t)exc_cnt, 16)) return false;
    for(int i=0;i<exc_cnt;++i){
        if (!bw.putN((uint64_t)exc_left[i], D.leftBW)) return false;
        if (!bw.putN((uint64_t)exc_pos[i], 16)) return false;
    }
    return true;
}

// ===================== 采样函数 =====================
template<typename T>
__device__ void rowgroup_sample_and_find_k_combinations(
    const T* rowgroup_data, 
    int rowgroup_size,
    int vectorSize,
    EFCombination* best_k_combinations,
    int& k_actual,
    CompressionMode& mode
) {
    int total_vectors = (rowgroup_size + vectorSize - 1) / vectorSize;
    int sample_stride = max(1, total_vectors / sampling_config::ROWGROUP_VECTOR_SAMPLES);
    
    struct LocalStats {
        int count;
        double total_score;
    } stats[19][19];
    
    for(int e=0; e<=18; e++) {
        for(int f=0; f<=e; f++) {
            stats[e][f].count = 0;
            stats[e][f].total_score = 0;
        }
    }
    
    double best_overall_compression_size = 1e30;
    int samples_taken = 0;
    
    for(int v = 0; v < total_vectors && samples_taken < sampling_config::ROWGROUP_VECTOR_SAMPLES; 
        v += sample_stride) {
        
        int vec_start = v * vectorSize;
        int vec_size = min(vectorSize, rowgroup_size - vec_start);
        if(vec_size <= 0) break;
        
        T samples[32];
        int sample_count = min(sampling_config::SAMPLES_PER_VECTOR, vec_size);
        int sample_step = max(1, vec_size / sample_count);
        
        for(int i = 0; i < sample_count; i++) {
            samples[i] = rowgroup_data[vec_start + i * sample_step];
        }
        
        uint8_t best_e = 0, best_f = 0;
        short bitw; long long FOR; int exc;
        alp_vector_choose_best_bits<T>(samples, sample_count, best_e, best_f, bitw, FOR, exc);
        
        int val_bits = std::is_same_v<T,double> ? 64 : 32;
        double compression_size = sample_count * bitw + exc * (val_bits + 16);
        
        stats[best_e][best_f].count++;
        stats[best_e][best_f].total_score += compression_size;
        
        if(compression_size < best_overall_compression_size) {
            best_overall_compression_size = compression_size;
        }
        
        samples_taken++;
    }
    
    if(best_overall_compression_size >= SamplingConstants<T>::RD_SIZE_THRESHOLD_LIMIT) {
        mode = CompressionMode::ALPrd;
        k_actual = 0;
        return;
    }
    
    mode = CompressionMode::ALP;
    
    EFCombination all_combinations[361];  
    int num_combinations = 0;
    
    for(int e = 0; e <= 18; e++) {
        for(int f = 0; f <= e; f++) {
            if(stats[e][f].count > 0) {
                all_combinations[num_combinations].e = e;
                all_combinations[num_combinations].f = f;
                all_combinations[num_combinations].count = stats[e][f].count;
                all_combinations[num_combinations].score = 
                    stats[e][f].total_score / stats[e][f].count;
                num_combinations++;
            }
        }
    }
    
    for(int i = 0; i < num_combinations - 1; i++) {
        for(int j = i + 1; j < num_combinations; j++) {
            bool swap = false;
            if(all_combinations[j].count > all_combinations[i].count) {
                swap = true;
            } else if(all_combinations[j].count == all_combinations[i].count) {
                if(all_combinations[j].score < all_combinations[i].score) {
                    swap = true;
                }
            }
            
            if(swap) {
                EFCombination tmp = all_combinations[i];
                all_combinations[i] = all_combinations[j];
                all_combinations[j] = tmp;
            }
        }
    }
    
    k_actual = min(sampling_config::MAX_K_COMBINATIONS, num_combinations);
    for(int i = 0; i < k_actual; i++) {
        best_k_combinations[i] = all_combinations[i];
    }
}

template<typename T>
__device__ void vector_choose_from_k_combinations(
    const T* vec_data,
    int vec_size,
    const EFCombination* k_combinations,
    int k,
    uint8_t& best_e,
    uint8_t& best_f,
    short& bitw,
    long long& FOR,
    int& exc
) {
    if(k == 1) {
        best_e = k_combinations[0].e;
        best_f = k_combinations[0].f;
        alp_vector_analyze<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
        return;
    }
    
    T samples[32];
    int sample_count = min(sampling_config::SAMPLES_PER_VECTOR, vec_size);
    int sample_step = max(1, vec_size / sample_count);
    
    for(int i = 0; i < sample_count; i++) {
        samples[i] = vec_data[i * sample_step];
    }
    
    double best_score = 1e30;
    int worse_count = 0;
    
    for(int kid = 0; kid < k; kid++) {
        uint8_t e = k_combinations[kid].e;
        uint8_t f = k_combinations[kid].f;
        
        short test_bitw;
        long long test_FOR;
        int test_exc;
        alp_vector_analyze<T>(samples, sample_count, e, f, test_bitw, test_FOR, test_exc);
        
        int val_bits = std::is_same_v<T,double> ? 64 : 32;
        double score = sample_count * test_bitw + test_exc * (val_bits + 16);
        
        if(score < best_score) {
            best_score = score;
            best_e = e;
            best_f = f;
            worse_count = 0;
        } else {
            worse_count++;
            if(worse_count >= sampling_config::EARLY_EXIT_THRESHOLD) {
                break;
            }
        }
    }
    
    alp_vector_analyze<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
}

// ===================== 优化的并行Kernels =====================
static constexpr int THREADS_PER_BLOCK = 128;

// 优化的测量kernel - 增加并行度
template<typename T>
__global__ void kernel_measure_parallel(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    int numBlocks,
    int vectorSize,
    uint64_t* out_bits,
    uint8_t* out_mode
) {
    int blockId = blockIdx.x;
    if (blockId >= numBlocks) return;
    
    const T* blk = data + blk_starts[blockId];
    int n = (int)blk_sizes[blockId];
    int numVec = (n + vectorSize - 1) / vectorSize;
    
    // 采样阶段（仍需串行，但只是小部分工作）
    __shared__ CompressionMode sh_mode;
    __shared__ EFCombination sh_k_combinations[5];
    __shared__ int sh_k_actual;
    
    if(threadIdx.x == 0) {
        EFCombination k_combinations[5];
        int k_actual;
        CompressionMode mode;
        
        rowgroup_sample_and_find_k_combinations<T>(
            blk, n, vectorSize,
            k_combinations, k_actual, mode
        );
        
        sh_mode = mode;
        sh_k_actual = k_actual;
        for(int i = 0; i < k_actual; i++) {
            sh_k_combinations[i] = k_combinations[i];
        }
    }
    __syncthreads();
    
    // 使用共享内存累加总位数
    __shared__ uint64_t sh_partial_sums[THREADS_PER_BLOCK];
    sh_partial_sums[threadIdx.x] = 0;
    
    if(sh_mode == CompressionMode::ALP) {
        // 每个线程并行处理多个向量
        for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = (vectorSize < rem ? vectorSize : rem);
            
            uint8_t e, f;
            short bw;
            long long FOR;
            int exc;
            
            vector_choose_from_k_combinations<T>(
                blk + beg, len,
                sh_k_combinations, sh_k_actual,
                e, f, bw, FOR, exc
            );
            
            uint64_t bits = alp_vector_size_bits_safe<T>(len, e, f, bw, exc);
            sh_partial_sums[threadIdx.x] += bits;
        }
    } else {
        // ALPrd模式的并行处理
        for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = (vectorSize < rem ? vectorSize : rem);
            
            uint64_t tmp[MAX_VEC];
            for(int i = 0; i < len; i++) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else 
                    tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            
            ALPrdDict<T> D;
            alprd_find_best<T>(tmp, len, D);
            
            int exc = 0;
            for(int i = 0; i < len; i++) {
                uint32_t left = (uint32_t)((tmp[i] >> D.rightBW) & mask_lo(D.leftBW));
                bool inDict = false;
                for(int k = 0; k < DICT_SZ; k++) {
                    if(D.dict[k] == left) {
                        inDict = true;
                        break;
                    }
                }
                if(!inDict) exc++;
            }
            
            uint64_t bits = alprd_vector_size_bits_safe<T>(len, D, exc);
            sh_partial_sums[threadIdx.x] += bits;
        }
    }
    
    __syncthreads();
    
    // 归约求和
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            sh_partial_sums[threadIdx.x] += sh_partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        out_bits[blockId] = 8 + sh_partial_sums[0];  // 8位是行组头
        out_mode[blockId] = (sh_mode == CompressionMode::ALPrd) ? 1 : 0;
    }
}

// 计算每个向量元数据的kernel（添加调试信息）
template<typename T>
__global__ void kernel_compute_vector_metadata(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const uint8_t* modes,
    int numBlocks,
    int vectorSize,
    uint64_t* vec_bit_sizes,
    uint8_t* vec_e,
    uint8_t* vec_f,
    uint16_t* vec_bitw,
    int64_t* vec_FOR,
    uint32_t* vec_exc_cnt
) {
    int blockId = blockIdx.x;
    if (blockId >= numBlocks) return;
    
    const T* blk = data + blk_starts[blockId];
    int n = (int)blk_sizes[blockId];
    int numVec = (n + vectorSize - 1) / vectorSize;
    CompressionMode mode = (modes[blockId] ? CompressionMode::ALPrd : CompressionMode::ALP);
    
    // 重新采样获取k个组合
    __shared__ EFCombination sh_k_combinations[5];
    __shared__ int sh_k_actual;
    __shared__ CompressionMode sh_mode;
    
    if(threadIdx.x == 0) {
        EFCombination k_combinations[5];
        int k_actual;
        CompressionMode mode_check;
        rowgroup_sample_and_find_k_combinations<T>(
            blk, n, vectorSize,
            k_combinations, k_actual, mode_check
        );
        sh_k_actual = k_actual;
        sh_mode = mode;
        for(int i = 0; i < k_actual; i++) {
            sh_k_combinations[i] = k_combinations[i];
        }
        
        // 调试：打印第一个块的信息
        // if(blockId == 0) {
        //     printf("[DEVICE-METADATA] Block0: n=%d, numVec=%d, mode=%s, k_actual=%d\n",
        //            n, numVec, mode == CompressionMode::ALP ? "ALP" : "ALPrd", k_actual);
        // }
    }
    __syncthreads();
    
    // 计算全局向量索引基址
    uint64_t vec_base = 0;
    for(int b = 0; b < blockId; b++) {
        vec_base += (blk_sizes[b] + vectorSize - 1) / vectorSize;
    }
    
    // 每个线程并行处理不同的向量
    for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
        uint64_t global_vec_idx = vec_base + v;
        int beg = v * vectorSize;
        int rem = n - beg;
        int len = (vectorSize < rem ? vectorSize : rem);
        
        if(sh_mode == CompressionMode::ALP) {
            uint8_t e, f;
            short bitw;
            long long FOR;
            int exc;
            
            vector_choose_from_k_combinations<T>(
                blk + beg, len,
                sh_k_combinations, sh_k_actual,
                e, f, bitw, FOR, exc
            );
            
            uint64_t bits = alp_vector_size_bits_safe<T>(len, e, f, bitw, exc);
            
            // 调试：检查异常位数
            // if(blockId == 0 && v == 0) {
            //     printf("[DEVICE-METADATA] Vec0: e=%d f=%d bitw=%d exc=%d bits=%llu\n",
            //            e, f, bitw, exc, bits);
            // }
            
            vec_bit_sizes[global_vec_idx] = bits;
            vec_e[global_vec_idx] = e;
            vec_f[global_vec_idx] = f;
            vec_bitw[global_vec_idx] = bitw;
            vec_FOR[global_vec_idx] = FOR;
            vec_exc_cnt[global_vec_idx] = exc;
            
        } else {
            // ALPrd模式
            uint64_t tmp[MAX_VEC];
            for(int i = 0; i < len; i++) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else 
                    tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            
            ALPrdDict<T> D;
            alprd_find_best<T>(tmp, len, D);
            
            int exc = 0;
            for(int i = 0; i < len; i++) {
                uint32_t left = (uint32_t)((tmp[i] >> D.rightBW) & mask_lo(D.leftBW));
                bool inDict = false;
                for(int k = 0; k < DICT_SZ; k++) {
                    if(D.dict[k] == left) {
                        inDict = true;
                        break;
                    }
                }
                if(!inDict) exc++;
            }
            
            uint64_t bits = alprd_vector_size_bits_safe<T>(len, D, exc);
            
            // // 调试：检查ALPrd位数
            // if(blockId == 0 && v == 0) {
            //     printf("[DEVICE-METADATA] ALPrd Vec0: rbw=%d lbw=%d exc=%d bits=%llu\n",
            //            D.rightBW, D.leftBW, exc, bits);
            // }
            
            // 检查异常值：如果位数太大，可能有问题
            if(bits > 100000) {
                printf("[DEVICE-WARNING] Block%d Vec%d: ALPrd bits=%llu too large! exc=%d\n",
                       blockId, v, bits, exc);
            }
            
            vec_bit_sizes[global_vec_idx] = bits;
            vec_e[global_vec_idx] = 0xFF;  // 标记为ALPrd
            vec_f[global_vec_idx] = 0xFF;
        }
    }
}

// 写入行组头的kernel
__global__ void kernel_write_rowgroup_headers(
    const uint64_t* blk_sizes,
    const uint64_t* blk_offsets,
    int numBlocks,
    int vectorSize,
    uint8_t* out_bytes
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= numBlocks) return;
    
    int numVec = (blk_sizes[blockId] + vectorSize - 1) / vectorSize;
    uint64_t bit_offset = blk_offsets[blockId];
    
    // 写入8位行组头
    SafeBitWriter bw(out_bytes, bit_offset, 8);
    bw.putN((uint64_t)numVec, 8);
}

// 并行写入kernel
template<typename T>
__global__ void kernel_parallel_emit(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const uint8_t* modes,
    const uint64_t* vec_bit_offsets,
    const uint8_t* vec_e,
    const uint8_t* vec_f,
    const uint16_t* vec_bitw,
    const int64_t* vec_FOR,
    int numBlocks,
    int vectorSize,
    uint8_t* out_bytes
) {
    int blockId = blockIdx.x;
    if (blockId >= numBlocks) return;
    
    const T* blk = data + blk_starts[blockId];
    int n = (int)blk_sizes[blockId];
    int numVec = (n + vectorSize - 1) / vectorSize;
    CompressionMode mode = (modes[blockId] ? CompressionMode::ALPrd : CompressionMode::ALP);
    
    // 计算全局向量索引基址
    uint64_t vec_base = 0;
    for(int b = 0; b < blockId; b++) {
        vec_base += (blk_sizes[b] + vectorSize - 1) / vectorSize;
    }
    
    // 每个线程并行写入自己负责的向量
    for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
        uint64_t global_vec_idx = vec_base + v;
        int beg = v * vectorSize;
        int rem = n - beg;
        int len = (vectorSize < rem ? vectorSize : rem);
        
        uint64_t bit_offset = vec_bit_offsets[global_vec_idx];
        SafeBitWriter bw(out_bytes, bit_offset, 100000);
        
        if(mode == CompressionMode::ALP) {
            uint8_t e = vec_e[global_vec_idx];
            uint8_t f = vec_f[global_vec_idx];
            short bitw = vec_bitw[global_vec_idx];
            long long FOR = vec_FOR[global_vec_idx];
            
            alp_vector_write_safe<T>(bw, blk + beg, len, e, f, bitw, FOR);
            
        } else {
            // ALPrd需要重新计算
            uint64_t tmp[MAX_VEC];
            for(int i = 0; i < len; i++) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else 
                    tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            
            ALPrdDict<T> D;
            alprd_find_best<T>(tmp, len, D);
            alprd_vector_write_safe<T>(bw, tmp, len, D);
        }
    }
}

// 解压kernel
template<typename T>
__global__ void kernel_decompress_debug(const uint8_t* bytes,
                                        const uint64_t* blk_starts_bits,
                                        const uint64_t* blk_bits,
                                        const uint64_t* out_starts,
                                        const int vectorSize,
                                        T* out_data, 
                                        int numBlocks) {
    int blockId = blockIdx.x;
    if (blockId >= numBlocks) return;

    uint64_t bit_offset = blk_starts_bits[blockId];
    BitReader br{bytes, bit_offset};
    
    int numVec = (int)br.getN(8);
    
    // if (blockId == 0) {
    //     printf("[DEVICE-DECOMP] Block0: bit_offset=%llu, numVec=%d\n", bit_offset, numVec);
    // }

    if (numVec <= 0 || numVec > 10000) {
        if (blockId == 0) {
            printf("[DEVICE-ERROR] Invalid numVec: %d\n", numVec);
        }
        return;
    }

    uint64_t out_pos = out_starts[blockId];
    
    for(int v = 0; v < numVec; v++) {
        int useALP = br.get1();
        
        // if (blockId == 0 && v == 0) {
        //     printf("[DEVICE-DECOMP] Vec0: useALP=%d\n", useALP);
        // }

        if (useALP) {
            uint8_t e = (uint8_t)br.getN(8);
            uint8_t f = (uint8_t)br.getN(8);
            short bitw = (short)br.getN(16);
            long long FOR = (long long)br.getN(64);
            int n = (int)br.getN(32);
            
            // if (blockId == 0 && v == 0) {
            //     printf("[DEVICE-DECOMP] Vec0 ALP: e=%d f=%d bitw=%d n=%d\n", e, f, bitw, n);
            // }

            if (n <= 0 || n > MAX_VEC) return;

            for(int k = 0; k < n; k++) {
                uint64_t enc = br.getN(bitw);
                long long I = FOR + (long long)enc;
                double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
                out_data[out_pos + k] = (T)dec;
            }

            int exc = (int)br.getN(16);
            for(int t = 0; t < exc; t++) {
                uint64_t raw = std::is_same_v<T,double> ? br.getN(64) : br.getN(32);
                int pos = (int)br.getN(16);
                
                if (pos < n) {
                    if constexpr (std::is_same_v<T,double>) {
                        double val = *reinterpret_cast<double*>(&raw);
                        out_data[out_pos + pos] = (T)val;
                    } else {
                        uint32_t rv = (uint32_t)raw;
                        float val = *reinterpret_cast<float*>(&rv);
                        out_data[out_pos + pos] = (T)val;
                    }
                }
            }
            out_pos += n;
        } else {
            int n = (int)br.getN(32);
            if (n <= 0 || n > MAX_VEC) return;

            uint8_t rbw = (uint8_t)br.getN(8);
            uint64_t right[MAX_VEC]; 
            uint16_t leftIdx[MAX_VEC];
            
            for(int k = 0; k < n; k++) {
                leftIdx[k] = (uint16_t)br.getN(DICT_BW);
                right[k] = br.getN(rbw);
            }

            uint8_t lbw = uint8_t(sizeof(T)*8 - rbw);
            uint64_t dict[DICT_SZ];
            for(int k = 0; k < DICT_SZ; k++) {
                dict[k] = br.getN(lbw);
            }

            int exc = (int)br.getN(16);
            uint16_t exc_pos[MAX_VEC]; 
            uint64_t exc_left[MAX_VEC];
            for(int t = 0; t < exc; t++) {
                exc_left[t] = br.getN(lbw);
                exc_pos[t] = (uint16_t)br.getN(16);
            }

            for(int k = 0; k < n; k++) {
                uint64_t left = (leftIdx[k] < DICT_SZ) ? dict[leftIdx[k]] : 0ULL;
                uint64_t raw = (left << rbw) | right[k];
                
                if constexpr (std::is_same_v<T,double>) {
                    double val = *reinterpret_cast<double*>(&raw);
                    out_data[out_pos + k] = (T)val;
                } else {
                    uint32_t r32 = (uint32_t)raw;
                    float val = *reinterpret_cast<float*>(&r32);
                    out_data[out_pos + k] = (T)val;
                }
            }

            for(int t = 0; t < exc; t++) {
                int p = exc_pos[t];
                uint64_t raw = (exc_left[t] << rbw) | right[p];
                
                if (p < n) {
                    if constexpr (std::is_same_v<T,double>) {
                        double val = *reinterpret_cast<double*>(&raw);
                        out_data[out_pos + p] = (T)val;
                    } else {
                        uint32_t r32 = (uint32_t)raw;
                        float val = *reinterpret_cast<float*>(&r32);
                        out_data[out_pos + p] = (T)val;
                    }
                }
            }
            out_pos += n;
        }
    }
}

// ===================== 优化的主要API实现 =====================
template<typename T>
static Compressed compress_impl_optimized(const T* h_data, size_t n, const Params& p) {
    // std::cout << "[DEBUG] 开始优化压缩，数据量: " << n << std::endl;
    
    Compressed c;
    if (n == 0) { c.vectorSize = p.vectorSize; return c; }
    
    const int V = p.vectorSize;
    const int B = p.blockSize > 0 ? p.blockSize : int(n);
    const int numBlocks = int((n + B - 1) / B);
    
    // std::cout << "[DEBUG] 块数: " << numBlocks << ", 向量大小: " << V << std::endl;
    
    // 计算总向量数
    uint64_t total_vectors = 0;
    std::vector<uint64_t> h_starts(numBlocks), h_sizes(numBlocks);
    size_t pos = 0;
    for(int i = 0; i < numBlocks; i++) {
        h_starts[i] = pos;
        uint64_t sz = std::min<uint64_t>(B, n - pos);
        h_sizes[i] = sz;
        total_vectors += (sz + V - 1) / V;
        pos += sz;
    }
    
    // std::cout << "[DEBUG] 总向量数: " << total_vectors << std::endl;
    
    // 上传数据到GPU
    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
    
    uint64_t *d_starts = nullptr, *d_sizes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_starts, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sizes, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_starts, h_starts.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // 第一阶段：并行测量
    uint64_t* d_bits = nullptr;
    uint8_t* d_mode = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_mode, numBlocks * sizeof(uint8_t)));
    
    dim3 grid1(numBlocks);
    dim3 block1(THREADS_PER_BLOCK);
    
    kernel_measure_parallel<T><<<grid1, block1>>>(
        d_data, d_starts, d_sizes, numBlocks, V,
        d_bits, d_mode
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<uint64_t> h_bits(numBlocks);
    std::vector<uint8_t> h_mode(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_bits.data(), d_bits, numBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mode.data(), d_mode, numBlocks * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    // 调试输出
    // std::cout << "[DEBUG] 各块压缩位数: ";
    // for (int i = 0; i < std::min(5, numBlocks); i++) {
    //     std::cout << h_bits[i] << "(" << (h_mode[i] ? "ALPrd" : "ALP") << ") ";
    // }
    // std::cout << std::endl;
    
    // 第二阶段：计算每个向量的元数据
    uint64_t* d_vec_bit_sizes = nullptr;
    uint8_t* d_vec_e = nullptr;
    uint8_t* d_vec_f = nullptr;
    uint16_t* d_vec_bitw = nullptr;
    int64_t* d_vec_FOR = nullptr;
    uint32_t* d_vec_exc_cnt = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_vec_bit_sizes, total_vectors * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_e, total_vectors * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_f, total_vectors * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_bitw, total_vectors * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_FOR, total_vectors * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_exc_cnt, total_vectors * sizeof(uint32_t)));
    
    kernel_compute_vector_metadata<T><<<grid1, block1>>>(
        d_data, d_starts, d_sizes, d_mode,
        numBlocks, V,
        d_vec_bit_sizes, d_vec_e, d_vec_f,
        d_vec_bitw, d_vec_FOR, d_vec_exc_cnt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 第三阶段：计算位偏移
    std::vector<uint64_t> h_vec_bit_sizes(total_vectors);
    CUDA_CHECK(cudaMemcpy(h_vec_bit_sizes.data(), d_vec_bit_sizes, 
                          total_vectors * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    std::vector<uint64_t> h_block_offsets(numBlocks);
    std::vector<uint64_t> h_vec_offsets(total_vectors);
    
    uint64_t block_offset = 0;
    uint64_t vec_idx = 0;
    
    for(int b = 0; b < numBlocks; b++) {
        h_block_offsets[b] = block_offset;
        
        // 行组头（8位）
        uint64_t current_offset = block_offset + 8;
        
        int numVec = (h_sizes[b] + V - 1) / V;
        for(int v = 0; v < numVec; v++) {
            h_vec_offsets[vec_idx] = current_offset;
            current_offset += h_vec_bit_sizes[vec_idx];
            vec_idx++;
        }
        
        // 对齐到32位边界
        uint64_t block_bits = current_offset - block_offset;
        uint64_t padding = (32 - (block_bits & 31)) & 31;
        block_offset = current_offset + padding;
    }
    
    const uint64_t total_bits = block_offset;
    const uint64_t total_bytes = (total_bits + 7) / 8;
    
    // std::cout << "[DEBUG] 优化后总位数: " << total_bits << std::endl;
    // std::cout << "[DEBUG] 优化后总字节数: " << total_bytes << std::endl;
    
    uint64_t* d_vec_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vec_offsets, total_vectors * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_vec_offsets, h_vec_offsets.data(), 
                         total_vectors * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // 第四阶段：并行写入压缩数据
    uint8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, total_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, total_bytes));
    
    // 写入行组头（使用设备端kernel写入以确保正确性）
    uint64_t* d_block_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_block_offsets, h_block_offsets.data(), 
                         numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // 使用kernel写入行组头
    int header_threads = std::min(numBlocks, 256);
    int header_blocks = (numBlocks + header_threads - 1) / header_threads;
    kernel_write_rowgroup_headers<<<header_blocks, header_threads>>>(
        d_sizes, d_block_offsets, numBlocks, V, d_out
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 并行写入所有向量
    kernel_parallel_emit<T><<<grid1, block1>>>(
        d_data, d_starts, d_sizes, d_mode,
        d_vec_offsets, d_vec_e, d_vec_f,
        d_vec_bitw, d_vec_FOR,
        numBlocks, V, d_out
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_block_offsets));
    
    // 第五阶段：拷回结果
    c.data.resize(total_bytes);
    CUDA_CHECK(cudaMemcpy(c.data.data(), d_out, total_bytes, cudaMemcpyDeviceToHost));
    
    // 填充压缩元数据
    c.offsets = std::move(h_block_offsets);
    c.bit_sizes.resize(numBlocks);
    vec_idx = 0;
    for(int b = 0; b < numBlocks; b++) {
        uint64_t block_bits = 8; // 行组头
        int numVec = (h_sizes[b] + V - 1) / V;
        for(int v = 0; v < numVec; v++) {
            block_bits += h_vec_bit_sizes[vec_idx++];
        }
        c.bit_sizes[b] = block_bits;
    }
    c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
    c.vectorSize = V;
    
    // 清理GPU内存
    CUDA_CHECK(cudaFree(d_vec_offsets));
    CUDA_CHECK(cudaFree(d_vec_exc_cnt));
    CUDA_CHECK(cudaFree(d_vec_FOR));
    CUDA_CHECK(cudaFree(d_vec_bitw));
    CUDA_CHECK(cudaFree(d_vec_f));
    CUDA_CHECK(cudaFree(d_vec_e));
    CUDA_CHECK(cudaFree(d_vec_bit_sizes));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_mode));
    CUDA_CHECK(cudaFree(d_bits));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_starts));
    CUDA_CHECK(cudaFree(d_data));
    
    return c;
}

template<typename T>
static void decompress_impl_fixed(const Compressed& c, T* h_out, size_t n, const Params& p) {
    // std::cout << "[DEBUG] 开始解压缩，数据量: " << n << std::endl;
    
    if (n == 0) return;
    const int numBlocks = (int)c.offsets.size();
    assert((size_t)numBlocks == c.elem_counts.size());

    // std::cout << "[DEBUG] 解压块数: " << numBlocks << std::endl;
    // std::cout << "[DEBUG] 压缩数据大小: " << c.data.size() << " bytes" << std::endl;

    if (c.data.empty()) {
        std::cout << "[ERROR] 压缩数据为空！" << std::endl;
        return;
    }

    // 检查压缩数据
    // std::cout << "[DEBUG] 压缩数据前10字节: ";
    // for (int i = 0; i < 10 && i < c.data.size(); i++) {
    //     printf("%02X ", c.data[i]);
    // }
    // std::cout << std::endl;

    // 初始化输出
    std::fill(h_out, h_out + n, T(-999.0));

    uint8_t* d_bytes = nullptr; 
    CUDA_CHECK(cudaMalloc(&d_bytes, c.data.size()));
    CUDA_CHECK(cudaMemcpy(d_bytes, c.data.data(), c.data.size(), cudaMemcpyHostToDevice));

    uint64_t *d_boff = nullptr, *d_bsiz = nullptr, *d_ost = nullptr;
    CUDA_CHECK(cudaMalloc(&d_boff, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_bsiz, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_boff, c.offsets.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bsiz, c.bit_sizes.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

    std::vector<uint64_t> h_outStarts(numBlocks);
    uint64_t acc = 0; 
    for(int i = 0; i < numBlocks; i++) { 
        h_outStarts[i] = acc; 
        acc += c.elem_counts[i]; 
    }
    
    if (acc != n) {
        std::cout << "[ERROR] elem_counts总和(" << acc << ")不等于输出元素数(" << n << ")" << std::endl;
        return;
    }

    CUDA_CHECK(cudaMalloc(&d_ost, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ost, h_outStarts.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

    T* d_out = nullptr; 
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(T)));

    dim3 gs(numBlocks), bs(1);
    kernel_decompress_debug<T><<<gs, bs>>>(
        d_bytes, d_boff, d_bsiz, d_ost, p.vectorSize, d_out, numBlocks
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(T), cudaMemcpyDeviceToHost));

    // 检查输出结果
    bool all_negative999 = true;
    bool all_zero = true;
    for (size_t i = 0; i < std::min(size_t(10), n); i++) {
        if (h_out[i] != T(-999.0)) all_negative999 = false;
        if (h_out[i] != T(0.0)) all_zero = false;
    }
    
    if (all_negative999) {
        std::cout << "[ERROR] 输出仍为初始值-999，解压失败！" << std::endl;
    } else if (all_zero) {
        std::cout << "[WARNING] 输出全为0！" << std::endl;
    // } else {
    //     std::cout << "[DEBUG] 输出前几个值: ";
    //     for (size_t i = 0; i < std::min(size_t(5), n); i++) {
    //         std::cout << h_out[i] << " ";
    //     }
    //     std::cout << std::endl;
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ost));
    CUDA_CHECK(cudaFree(d_bsiz));
    CUDA_CHECK(cudaFree(d_boff));
    CUDA_CHECK(cudaFree(d_bytes));
}

// 显式实例化API
Compressed compress_double(const double* data, size_t n, const Params& p) { 
    return compress_impl_optimized<double>(data, n, p); 
}
Compressed compress_float(const float* data, size_t n, const Params& p) { 
    return compress_impl_optimized<float>(data, n, p); 
}
void decompress_double(const Compressed& c, double* out, size_t n, const Params& p) { 
    decompress_impl_fixed<double>(c, out, n, p); 
}
void decompress_float(const Compressed& c, float* out, size_t n, const Params& p) { 
    decompress_impl_fixed<float>(c, out, n, p); 
}


// 设备端压缩实现
template<typename T>
static CompressedDevice compress_device_impl(const T* d_data, size_t n, const Params& p, cudaStream_t stream) {
    CompressedDevice c;
    if (n == 0) { c.vectorSize = p.vectorSize; return c; }
    
    const int V = p.vectorSize;
    const int B = p.blockSize > 0 ? p.blockSize : int(n);
    const int numBlocks = int((n + B - 1) / B);
    
    // 计算总向量数和块信息
    uint64_t total_vectors = 0;
    std::vector<uint64_t> h_starts(numBlocks), h_sizes(numBlocks);
    size_t pos = 0;
    for(int i = 0; i < numBlocks; i++) {
        h_starts[i] = pos;
        uint64_t sz = std::min<uint64_t>(B, n - pos);
        h_sizes[i] = sz;
        total_vectors += (sz + V - 1) / V;
        pos += sz;
    }
    
    // 上传块信息到GPU（使用异步传输）
    uint64_t *d_starts = nullptr, *d_sizes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_starts, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sizes, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_starts, h_starts.data(), numBlocks * sizeof(uint64_t), 
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sizes, h_sizes.data(), numBlocks * sizeof(uint64_t), 
                               cudaMemcpyHostToDevice, stream));
    
    // 第一阶段：并行测量
    uint64_t* d_bits = nullptr;
    uint8_t* d_mode = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_mode, numBlocks * sizeof(uint8_t)));
    
    dim3 grid1(numBlocks);
    dim3 block1(THREADS_PER_BLOCK);
    
    kernel_measure_parallel<T><<<grid1, block1, 0, stream>>>(
        d_data, d_starts, d_sizes, numBlocks, V,
        d_bits, d_mode
    );
    
    // 只拷贝必要的元数据回主机
    std::vector<uint64_t> h_bits(numBlocks);
    std::vector<uint8_t> h_mode(numBlocks);
    CUDA_CHECK(cudaMemcpyAsync(h_bits.data(), d_bits, numBlocks * sizeof(uint64_t), 
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_mode.data(), d_mode, numBlocks * sizeof(uint8_t), 
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // 需要元数据来计算偏移
    
    // 第二阶段：计算每个向量的元数据
    uint64_t* d_vec_bit_sizes = nullptr;
    uint8_t* d_vec_e = nullptr;
    uint8_t* d_vec_f = nullptr;
    uint16_t* d_vec_bitw = nullptr;
    int64_t* d_vec_FOR = nullptr;
    uint32_t* d_vec_exc_cnt = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_vec_bit_sizes, total_vectors * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_e, total_vectors * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_f, total_vectors * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_bitw, total_vectors * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_FOR, total_vectors * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_exc_cnt, total_vectors * sizeof(uint32_t)));
    
    kernel_compute_vector_metadata<T><<<grid1, block1, 0, stream>>>(
        d_data, d_starts, d_sizes, d_mode,
        numBlocks, V,
        d_vec_bit_sizes, d_vec_e, d_vec_f,
        d_vec_bitw, d_vec_FOR, d_vec_exc_cnt
    );
    
    // 拷贝向量位大小用于计算偏移
    std::vector<uint64_t> h_vec_bit_sizes(total_vectors);
    CUDA_CHECK(cudaMemcpyAsync(h_vec_bit_sizes.data(), d_vec_bit_sizes, 
                               total_vectors * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // 第三阶段：计算位偏移
    std::vector<uint64_t> h_block_offsets(numBlocks);
    std::vector<uint64_t> h_vec_offsets(total_vectors);
    
    uint64_t block_offset = 0;
    uint64_t vec_idx = 0;
    
    for(int b = 0; b < numBlocks; b++) {
        h_block_offsets[b] = block_offset;
        uint64_t current_offset = block_offset + 8; // 行组头
        
        int numVec = (h_sizes[b] + V - 1) / V;
        for(int v = 0; v < numVec; v++) {
            h_vec_offsets[vec_idx] = current_offset;
            current_offset += h_vec_bit_sizes[vec_idx];
            vec_idx++;
        }
        
        uint64_t block_bits = current_offset - block_offset;
        uint64_t padding = (32 - (block_bits & 31)) & 31;
        block_offset = current_offset + padding;
    }
    
    const uint64_t total_bits = block_offset;
    const uint64_t total_bytes = (total_bits + 7) / 8;
    
    // 上传向量偏移到GPU
    uint64_t* d_vec_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vec_offsets, total_vectors * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_vec_offsets, h_vec_offsets.data(), 
                              total_vectors * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    
    // 第四阶段：分配设备端输出缓冲并写入
    CUDA_CHECK(cudaMalloc(&c.d_data, total_bytes));
    CUDA_CHECK(cudaMemsetAsync(c.d_data, 0, total_bytes, stream));
    c.data_size = total_bytes;
    
    // 写入行组头
    uint64_t* d_block_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_block_offsets, h_block_offsets.data(), 
                               numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    
    int header_threads = std::min(numBlocks, 256);
    int header_blocks = (numBlocks + header_threads - 1) / header_threads;
    kernel_write_rowgroup_headers<<<header_blocks, header_threads, 0, stream>>>(
        d_sizes, d_block_offsets, numBlocks, V, c.d_data
    );
    
    // 并行写入所有向量
    kernel_parallel_emit<T><<<grid1, block1, 0, stream>>>(
        d_data, d_starts, d_sizes, d_mode,
        d_vec_offsets, d_vec_e, d_vec_f,
        d_vec_bitw, d_vec_FOR,
        numBlocks, V, c.d_data
    );
    
    // 填充元数据（保持在主机端）
    c.offsets = std::move(h_block_offsets);
    c.bit_sizes.resize(numBlocks);
    vec_idx = 0;
    for(int b = 0; b < numBlocks; b++) {
        uint64_t block_bits = 8; // 行组头
        int numVec = (h_sizes[b] + V - 1) / V;
        for(int v = 0; v < numVec; v++) {
            block_bits += h_vec_bit_sizes[vec_idx++];
        }
        c.bit_sizes[b] = block_bits;
    }
    c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
    c.vectorSize = V;
    
    // 清理临时GPU内存
    CUDA_CHECK(cudaFree(d_block_offsets));
    CUDA_CHECK(cudaFree(d_vec_offsets));
    CUDA_CHECK(cudaFree(d_vec_exc_cnt));
    CUDA_CHECK(cudaFree(d_vec_FOR));
    CUDA_CHECK(cudaFree(d_vec_bitw));
    CUDA_CHECK(cudaFree(d_vec_f));
    CUDA_CHECK(cudaFree(d_vec_e));
    CUDA_CHECK(cudaFree(d_vec_bit_sizes));
    CUDA_CHECK(cudaFree(d_mode));
    CUDA_CHECK(cudaFree(d_bits));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_starts));
    
    return c;
}

// 设备端解压实现
template<typename T>
static void decompress_device_impl(const CompressedDevice& c, T* d_out, size_t n, const Params& p, cudaStream_t stream) {
    if (n == 0) return;
    const int numBlocks = (int)c.offsets.size();
    
    // 上传元数据到GPU
    uint64_t *d_boff = nullptr, *d_bsiz = nullptr, *d_ost = nullptr;
    CUDA_CHECK(cudaMalloc(&d_boff, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_bsiz, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_boff, c.offsets.data(), numBlocks * sizeof(uint64_t), 
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bsiz, c.bit_sizes.data(), numBlocks * sizeof(uint64_t), 
                               cudaMemcpyHostToDevice, stream));
    
    std::vector<uint64_t> h_outStarts(numBlocks);
    uint64_t acc = 0;
    for(int i = 0; i < numBlocks; i++) {
        h_outStarts[i] = acc;
        acc += c.elem_counts[i];
    }
    
    CUDA_CHECK(cudaMalloc(&d_ost, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_ost, h_outStarts.data(), numBlocks * sizeof(uint64_t), 
                              cudaMemcpyHostToDevice, stream));
    
    // 执行解压kernel
    dim3 gs(numBlocks), bs(1);
    kernel_decompress_debug<T><<<gs, bs, 0, stream>>>(
        c.d_data, d_boff, d_bsiz, d_ost, p.vectorSize, d_out, numBlocks
    );
    
    // 清理
    CUDA_CHECK(cudaFree(d_ost));
    CUDA_CHECK(cudaFree(d_bsiz));
    CUDA_CHECK(cudaFree(d_boff));
}

// API实现
CompressedDevice compress_double_device(const double* d_data, size_t n, const Params& p, cudaStream_t stream) {
    return compress_device_impl<double>(d_data, n, p, stream);
}

CompressedDevice compress_float_device(const float* d_data, size_t n, const Params& p, cudaStream_t stream) {
    return compress_device_impl<float>(d_data, n, p, stream);
}

void decompress_double_device(const CompressedDevice& c, double* d_out, size_t n, const Params& p, cudaStream_t stream) {
    decompress_device_impl<double>(c, d_out, n, p, stream);
}

void decompress_float_device(const CompressedDevice& c, float* d_out, size_t n, const Params& p, cudaStream_t stream) {
    decompress_device_impl<float>(c, d_out, n, p, stream);
}

// 辅助函数
Compressed device_to_host(const CompressedDevice& cd) {
    Compressed c;
    c.data.resize(cd.data_size);
    cudaMemcpy(c.data.data(), cd.d_data, cd.data_size, cudaMemcpyDeviceToHost);
    c.offsets = cd.offsets;
    c.bit_sizes = cd.bit_sizes;
    c.elem_counts = cd.elem_counts;
    c.vectorSize = cd.vectorSize;
    return c;
}

CompressedDevice host_to_device(const Compressed& c) {
    CompressedDevice cd;
    cd.data_size = c.data.size();
    cudaMalloc(&cd.d_data, cd.data_size);
    cudaMemcpy(cd.d_data, c.data.data(), cd.data_size, cudaMemcpyHostToDevice);
    cd.offsets = c.offsets;
    cd.bit_sizes = c.bit_sizes;
    cd.elem_counts = c.elem_counts;
    cd.vectorSize = c.vectorSize;
    return cd;
}

} // namespace alp_gpu