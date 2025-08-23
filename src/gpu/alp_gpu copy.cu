/*
 * ============================================
 *  ALP-GPU 压缩/解压（两级采样版本 - 修复版）
 *  修复内容：
 *    1) 修正kernel线程配置
 *    2) 修正函数声明顺序
 *    3) 确保压缩/解压逻辑正确
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

namespace alp_gpu {

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

namespace SamplingConfig {
    static constexpr int ROWGROUP_SAMPLE_VECTORS = 8;
    static constexpr int VALUES_PER_VECTOR_SAMPLE = 32;
    static constexpr int MAX_KEPT_COMBINATIONS = 5;
    static constexpr int VECTOR_SAMPLE_VALUES = 32;
    static constexpr int EARLY_EXIT_THRESHOLD = 2;
}

struct ALPCombination {
    uint8_t e;
    uint8_t f;
    uint32_t count;
    double score;
    
    __device__ __host__ ALPCombination() : e(0), f(0), count(0), score(1e100) {}
    __device__ __host__ ALPCombination(uint8_t _e, uint8_t _f, uint32_t _c = 0, double _s = 1e100) 
        : e(_e), f(_f), count(_c), score(_s) {}
};

// ===================== 位流 Writer/Reader =====================
struct BitWriter {
    uint8_t* buf;
    uint64_t bitpos;
    __device__ void put1(int b){
        if (!b) { ++bitpos; return; }
        uint64_t byte = bitpos >> 3;
        int off = 7 - int(bitpos & 7ULL);
        buf[byte] = uint8_t(buf[byte] | (uint8_t(1u) << off));
        ++bitpos;
    }
    __device__ void putN(uint64_t v, int bits){
        for(int i=bits-1;i>=0;--i) put1( (v>>i) & 1ULL );
    }
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
};

// ===================== 基础工具函数（先声明）=====================
__device__ __forceinline__ int width_needed_unsigned(unsigned long long range){
    if (range==0ULL) return 1;
    int c=0; while(range){ ++c; range>>=1ULL; } return c;
}

__device__ inline long long fast_round_double(double x){
    const double SWEET = double((1ULL<<51) + (1ULL<<52));
    return (long long)(x + SWEET) - (long long)SWEET;
}

// ===================== ALP核心函数（必须在使用前定义）=====================
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
        if (dec==double(v[i])) { 
            mn=(mn<I?mn:I); 
            mx=(mx>I?mx:I); 
        } else {
            ++exc_cnt;
        }
    }
    unsigned long long range = (mn==LLONG_MAX)? 0ULL : (unsigned long long)(mx - mn);
    bitw = (short)width_needed_unsigned(range);
    FOR  = (mn==LLONG_MAX?0:mn);
}

template<typename T>
__device__ inline bool can_encode_exactly(T value, uint8_t e, uint8_t f) {
    if constexpr (std::is_same_v<T,double>) {
        double enc = value * D_EXP_ARR[e] * D_FRAC_ARR[f];
        long long I = fast_round_double(enc);
        double dec = double(I) * (1.0 / D_FRAC_ARR[f]) * D_FRAC_ARR[e];
        return dec == value;
    } else {
        float enc = value * float(D_EXP_ARR[e]) * float(D_FRAC_ARR[f]);
        int I = __float2int_rn(enc);
        float dec = float(I) * (1.0f/float(D_FRAC_ARR[f])) * float(D_FRAC_ARR[e]);
        return dec == value;
    }
}

template<typename T>
__device__ inline double evaluate_combination(const T* sample, int n, uint8_t e, uint8_t f) {
    int exceptions = 0;
    long long minVal = LLONG_MAX, maxVal = LLONG_MIN;
    
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T,double>) {
            double enc = sample[i] * D_EXP_ARR[e] * D_FRAC_ARR[f];
            long long I = fast_round_double(enc);
            double dec = double(I) * (1.0 / D_FRAC_ARR[f]) * D_FRAC_ARR[e];
            if (dec == sample[i]) {
                minVal = min(minVal, I);
                maxVal = max(maxVal, I);
            } else {
                exceptions++;
            }
        } else {
            float enc = sample[i] * float(D_EXP_ARR[e]) * float(D_FRAC_ARR[f]);
            int I = __float2int_rn(enc);
            float dec = float(I) * (1.0f/float(D_FRAC_ARR[f])) * float(D_FRAC_ARR[e]);
            if (dec == sample[i]) {
                minVal = min(minVal, (long long)I);
                maxVal = max(maxVal, (long long)I);
            } else {
                exceptions++;
            }
        }
    }
    
    if (exceptions >= n * 0.5) {
        return 1e100;
    }
    
    unsigned long long range = (minVal == LLONG_MAX) ? 0 : (maxVal - minVal);
    int bitsNeeded = width_needed_unsigned(range);
    double baseSize = double(n) * bitsNeeded;
    double exceptionSize = exceptions * (sizeof(T) * 8 + 16);
    
    return baseSize + exceptionSize;
}

// 第二级采样（必须在第一级采样之前定义，因为后者可能会用到）
template<typename T>
__device__ void second_level_sampling(
    const T* vec,
    int len,
    const ALPCombination* candidates,
    int num_candidates,
    uint8_t& best_e,
    uint8_t& best_f,
    short& bitw,
    long long& FOR,
    int& exc
) {
    if (num_candidates == 0) {
        // 没有候选，降级到全搜索最小异常
        best_e = 0; best_f = 0;
        int min_exc = len + 1;
        for (uint8_t e = 0; e <= 18; ++e) {
            for (uint8_t f = 0; f <= e; ++f) {
                short _bw; long long _FOR; int _exc;
                alp_vector_analyze<T>(vec, len, e, f, _bw, _FOR, _exc);
                if (_exc < min_exc) {
                    min_exc = _exc;
                    best_e = e; best_f = f; bitw = _bw; FOR = _FOR; exc = _exc;
                }
            }
        }
        return;
    }
    
    if (num_candidates == 1) {
        best_e = candidates[0].e;
        best_f = candidates[0].f;
        alp_vector_analyze<T>(vec, len, best_e, best_f, bitw, FOR, exc);
        return;
    }
    
    // 采样评估
    T sample[SamplingConfig::VECTOR_SAMPLE_VALUES];
    int sampleCount = min(SamplingConfig::VECTOR_SAMPLE_VALUES, len);
    int sampleStep = max(1, len / sampleCount);
    
    for (int i = 0, idx = 0; i < len && idx < sampleCount; i += sampleStep, idx++) {
        sample[idx] = vec[i];
    }
    
    double best_score = 1e100;
    int worse_count = 0;
    
    for (int c = 0; c < num_candidates; ++c) {
        uint8_t e = candidates[c].e;
        uint8_t f = candidates[c].f;
        
        double score = evaluate_combination<T>(sample, sampleCount, e, f);
        
        if (score < best_score) {
            best_score = score;
            best_e = e;
            best_f = f;
            worse_count = 0;
        } else {
            worse_count++;
            if (worse_count >= SamplingConfig::EARLY_EXIT_THRESHOLD) {
                break;
            }
        }
    }
    
    // 用选定的(e,f)做完整分析
    alp_vector_analyze<T>(vec, len, best_e, best_f, bitw, FOR, exc);
}

// 第一级采样
template<typename T>
__global__ void kernel_first_level_sampling(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    int numBlocks,
    int vectorSize,
    ALPCombination* block_candidates,
    int* num_candidates
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= numBlocks) return;
    
    const T* blk = data + blk_starts[blockId];
    int n = (int)blk_sizes[blockId];
    int numVec = (n + vectorSize - 1) / vectorSize;
    
    const int MAX_COMBOS = 19 * 19;
    uint32_t combo_counts[MAX_COMBOS] = {0};
    
    int sampleStep = max(1, numVec / SamplingConfig::ROWGROUP_SAMPLE_VECTORS);
    int sampledVectors = 0;
    
    for (int v = 0; v < numVec && sampledVectors < SamplingConfig::ROWGROUP_SAMPLE_VECTORS; v += sampleStep) {
        int vecStart = v * vectorSize;
        int vecLen = min(vectorSize, n - vecStart);
        
        T sample[SamplingConfig::VALUES_PER_VECTOR_SAMPLE];
        int sampleCount = min(SamplingConfig::VALUES_PER_VECTOR_SAMPLE, vecLen);
        int valueStep = max(1, vecLen / sampleCount);
        
        for (int i = 0, idx = 0; i < vecLen && idx < sampleCount; i += valueStep, idx++) {
            sample[idx] = blk[vecStart + i];
        }
        
        uint8_t best_e = 0, best_f = 0;
        double best_score = 1e100;
        
        for (uint8_t e = 0; e <= 18; ++e) {
            for (uint8_t f = 0; f <= e; ++f) {
                double score = evaluate_combination<T>(sample, sampleCount, e, f);
                if (score < best_score) {
                    best_score = score;
                    best_e = e;
                    best_f = f;
                }
            }
        }
        
        int combo_idx = best_e * 19 + best_f;
        combo_counts[combo_idx]++;
        sampledVectors++;
    }
    
    ALPCombination candidates[SamplingConfig::MAX_KEPT_COMBINATIONS];
    int k = 0;
    
    for (int round = 0; round < SamplingConfig::MAX_KEPT_COMBINATIONS; ++round) {
        uint32_t max_count = 0;
        int best_idx = -1;
        
        for (int i = 0; i < MAX_COMBOS; ++i) {
            if (combo_counts[i] > max_count) {
                max_count = combo_counts[i];
                best_idx = i;
            }
        }
        
        if (best_idx >= 0 && max_count > 0) {
            uint8_t e = best_idx / 19;
            uint8_t f = best_idx % 19;
            candidates[k] = ALPCombination(e, f, max_count, 0);
            combo_counts[best_idx] = 0;
            k++;
        } else {
            break;
        }
    }
    
    int base_idx = blockId * SamplingConfig::MAX_KEPT_COMBINATIONS;
    for (int i = 0; i < k; ++i) {
        block_candidates[base_idx + i] = candidates[i];
    }
    for (int i = k; i < SamplingConfig::MAX_KEPT_COMBINATIONS; ++i) {
        block_candidates[base_idx + i] = ALPCombination();
    }
    num_candidates[blockId] = k;
}

// ALP相关函数
template<typename T>
__device__ inline uint64_t alp_vector_size_bits(int n, uint8_t e, uint8_t f,
                                                short bitw, int exc_cnt){
    int val_bits = std::is_same_v<T,double> ? 64 : 32;
    return 1 + 8+8+16+64+32 + uint64_t(n)*bitw + 16 + uint64_t(exc_cnt)*(val_bits+16);
}

template<typename T>
__device__ inline void alp_vector_write(BitWriter& bw, const T* v, int n,
                                       uint8_t e, uint8_t f, short bitw, long long FOR){
    assert(n <= MAX_VEC);
    bw.put1(1);
    bw.putN((uint64_t)e, 8);
    bw.putN((uint64_t)f, 8);
    bw.putN((uint64_t)bitw, 16);
    bw.putN((uint64_t)FOR, 64);
    bw.putN((uint64_t)n, 32);

    int exc_cnt=0;
    int      exc_pos[MAX_VEC];
    uint64_t exc_val[MAX_VEC];

    for(int i=0;i<n;++i){
        double enc = double(v[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
        long long I = fast_round_double(enc);
        double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
        if (dec==double(v[i])) {
            uint64_t packed = (uint64_t)(I - FOR);
            bw.putN(packed, bitw);
        } else {
            bw.putN(0, bitw);
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
    bw.putN((uint64_t)exc_cnt, 16);
    for(int k=0;k<exc_cnt;++k){
        if constexpr (std::is_same_v<T,double>) bw.putN(exc_val[k], 64);
        else bw.putN(exc_val[k], 32);
        bw.putN((uint64_t)exc_pos[k], 16);
    }
}

// ALPrd相关
template<typename T> struct ALPrdDict {
    uint8_t rightBW;
    uint8_t leftBW;
    uint32_t dict[DICT_SZ];
};

__device__ inline uint32_t mask_lo(int bits){
    return (bits >= 32) ? 0xFFFFFFFFu : ((1u<<bits) - 1u);
}

template<typename T>
__device__ inline void alprd_find_best(const uint64_t* in, int n, ALPrdDict<T>& D){
    double best_score = 1e100;
    int best_rbw = int(sizeof(T)*8) - 1;
    uint32_t best_dict[DICT_SZ] = {0};

    for(int lbw=1; lbw<=CUT_LIM; ++lbw){
        int rbw = int(sizeof(T)*8) - lbw;
        uint32_t lmask = mask_lo(lbw);

        uint32_t uniq_left[MAX_VEC];
        int cnt[MAX_VEC];
        int u = 0;

        for(int i=0;i<n;++i){
            uint32_t left = (uint32_t)((in[i] >> rbw) & lmask);
            int j=0;
            for(; j<u; ++j) {
                if (uniq_left[j]==left) {
                    ++cnt[j];
                    break;
                }
            }
            if (j==u){
                uniq_left[u]=left;
                cnt[u]=1;
                ++u;
            }
        }
        
        uint32_t dict[DICT_SZ]={0};
        int used = (DICT_SZ < u ? DICT_SZ : u);
        for(int k=0;k<used;++k){
            int best=-1, id=-1;
            for(int j=0;j<u;++j){
                bool taken=false;
                for(int t=0;t<k;++t) {
                    if (dict[t]==uniq_left[j]) {
                        taken=true;
                        break;
                    }
                }
                if (taken) continue;
                if (cnt[j]>best){
                    best=cnt[j];
                    id=j;
                }
            }
            dict[k] = uniq_left[id];
        }
        
        int keep=0;
        for(int k=0;k<used;++k){
            for(int j=0;j<u;++j) {
                if (uniq_left[j]==dict[k]) {
                    keep += cnt[j];
                    break;
                }
            }
        }
        int exc = n - keep;

        double bits = 1 + 32 + 8
                      + double(n)*(DICT_BW + rbw)
                      + double(DICT_SZ)*lbw
                      + 16.0*exc + double(lbw)*exc;

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
__device__ inline uint64_t alprd_vector_size_bits(int n, const ALPrdDict<T>& D, int exc_cnt){
    return 1 + 32 + 8 + uint64_t(n)*(DICT_BW + D.rightBW)
           + DICT_SZ*D.leftBW + 16 + uint64_t(exc_cnt)*(D.leftBW+16);
}

template<typename T>
__device__ inline void alprd_vector_write(BitWriter& bw, const uint64_t* in, int n,
                                          const ALPrdDict<T>& D){
    assert(n <= MAX_VEC);
    bw.put1(0);
    bw.putN((uint64_t)n, 32);
    bw.putN((uint64_t)D.rightBW, 8);

    int exc_cnt=0;
    uint16_t exc_pos[MAX_VEC];
    uint32_t exc_left[MAX_VEC];
    uint64_t right_mask = (D.rightBW==64)? ~0ULL : ((1ULL<<D.rightBW)-1ULL);
    uint32_t left_mask  = mask_lo(D.leftBW);

    for(int i=0;i<n;++i){
        uint64_t right = in[i] & right_mask;
        uint32_t left  = (uint32_t)((in[i] >> D.rightBW) & left_mask);
        short idx = DICT_SZ;
        for(int k=0;k<DICT_SZ;++k){
            if (D.dict[k]==left){
                idx=(short)k;
                break;
            }
        }
        if (idx<DICT_SZ){
            bw.putN((uint64_t)idx, DICT_BW);
            bw.putN(right, D.rightBW);
        }else{
            bw.putN(0, DICT_BW);
            bw.putN(right, D.rightBW);
            exc_pos[exc_cnt]  = (uint16_t)i;
            exc_left[exc_cnt] = left;
            ++exc_cnt;
        }
    }
    
    for(int k=0;k<DICT_SZ;++k) bw.putN((uint64_t)D.dict[k], D.leftBW);
    bw.putN((uint64_t)exc_cnt, 16);
    for(int i=0;i<exc_cnt;++i){
        bw.putN((uint64_t)exc_left[i], D.leftBW);
        bw.putN((uint64_t)exc_pos[i], 16);
    }
}

// 模式判定
template<typename T>
__device__ inline bool is_high_precision_value(T v){
    for(uint8_t e=0;e<=18;++e){
        for(uint8_t f=0; f<=e; ++f){
            if (can_encode_exactly<T>(v,e,f)) return false;
        }
    }
    return true;
}

template<typename T>
__device__ inline CompressionMode decide_mode_with_candidates(
    const T* blk, int n, int vectorSize,
    const ALPCombination* candidates, int num_candidates
) {
    int sample_stride = max(1, n/100);
    int total=0, highp=0;
    for(int i=0;i<n; i+=sample_stride){ 
        ++total; 
        if (is_high_precision_value<T>(blk[i])) ++highp; 
    }
    if (total>0 && (double)highp/total > 0.5) return CompressionMode::ALPrd;
    
    if (num_candidates > 0) {
        uint8_t e = candidates[0].e;
        uint8_t f = candidates[0].f;
        
        int sample_count = min(32, n);
        int sample_step = max(1, n / sample_count);
        int exceptions = 0;
        
        for (int i = 0; i < n && i/sample_step < sample_count; i += sample_step) {
            if (!can_encode_exactly<T>(blk[i], e, f)) {
                exceptions++;
            }
        }
        
        double exception_rate = double(exceptions) / double(sample_count);
        if (exception_rate > 0.2) return CompressionMode::ALPrd;
    }
    
    return CompressionMode::ALP;
}

// 修复的压缩kernels - 注意线程数配置
template<typename T>
__global__ void kernel_size_and_mode_with_sampling(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const ALPCombination* block_candidates,
    const int* num_candidates,
    int numBlocks,
    int vectorSize,
    uint64_t* out_bits,
    uint8_t* out_mode
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 修复：正确的线程索引计算
    if (i >= numBlocks) return;
    
    const T* blk = data + blk_starts[i];
    int n = (int)blk_sizes[i];
    
    const ALPCombination* my_candidates = 
        block_candidates + i * SamplingConfig::MAX_KEPT_COMBINATIONS;
    int my_num_candidates = num_candidates[i];
    
    CompressionMode mode = decide_mode_with_candidates<T>(
        blk, n, vectorSize, my_candidates, my_num_candidates
    );
    
    int numVec = (n + vectorSize - 1) / vectorSize;
    uint64_t bits = 8;
    
    if (mode == CompressionMode::ALP) {
        for (int v = 0; v < numVec; ++v) {
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = min(vectorSize, rem);
            
            uint8_t e, f; short bw; long long FOR; int exc;
            second_level_sampling<T>(blk + beg, len, my_candidates, my_num_candidates,
                                     e, f, bw, FOR, exc);
            bits += alp_vector_size_bits<T>(len, e, f, bw, exc);
        }
    } else {
        for (int v = 0; v < numVec; ++v) {
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = min(vectorSize, rem);
            
            uint64_t tmp[MAX_VEC];
            for (int i = 0; i < len; ++i) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else 
                    tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            
            ALPrdDict<T> D;
            alprd_find_best<T>(tmp, len, D);
            
            int exc = 0;
            for (int i = 0; i < len; ++i) {
                uint32_t left = (uint32_t)((tmp[i] >> D.rightBW) & mask_lo(D.leftBW));
                bool inDict = false;
                for (int k = 0; k < DICT_SZ; ++k) {
                    if (D.dict[k] == left) {
                        inDict = true;
                        break;
                    }
                }
                if (!inDict) exc++;
            }
            bits += alprd_vector_size_bits<T>(len, D, exc);
        }
    }
    
    out_bits[i] = bits;
    out_mode[i] = (mode == CompressionMode::ALPrd) ? 1 : 0;
}

template<typename T>
__global__ void kernel_compress_emit_with_sampling(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const uint64_t* bit_offsets,
    const uint8_t* modes,
    const ALPCombination* block_candidates,
    const int* num_candidates,
    const uint64_t* vec_prefix,
    uint8_t* dbg_modes,
    uint8_t* dbg_e,
    uint8_t* dbg_f,
    int enable_diag,
    int numBlocks,
    int vectorSize,
    uint8_t* out_bytes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 修复：正确的线程索引计算
    if (i >= numBlocks) return;
    
    const T* blk = data + blk_starts[i];
    int n = (int)blk_sizes[i];
    BitWriter bw{out_bytes, bit_offsets[i]};
    CompressionMode mode = (modes[i] ? CompressionMode::ALPrd : CompressionMode::ALP);
    
    const ALPCombination* my_candidates = 
        block_candidates + i * SamplingConfig::MAX_KEPT_COMBINATIONS;
    int my_num_candidates = num_candidates[i];
    
    int numVec = (n + vectorSize - 1) / vectorSize;
    bw.putN((uint64_t)numVec, 8);
    
    if (mode == CompressionMode::ALP) {
        for (int v = 0; v < numVec; ++v) {
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = min(vectorSize, rem);
            
            uint8_t e, f; short bw_width; long long FOR; int exc;
            second_level_sampling<T>(blk + beg, len, my_candidates, my_num_candidates,
                                     e, f, bw_width, FOR, exc);
            
            if (enable_diag && dbg_modes) {
                uint64_t gid = vec_prefix[i] + (uint64_t)v;
                dbg_modes[gid] = 0;
                if (dbg_e) dbg_e[gid] = e;
                if (dbg_f) dbg_f[gid] = f;
            }
            
            alp_vector_write<T>(bw, blk + beg, len, e, f, bw_width, FOR);
        }
    } else {
        for (int v = 0; v < numVec; ++v) {
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = min(vectorSize, rem);
            
            uint64_t tmp[MAX_VEC];
            for (int i = 0; i < len; ++i) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else 
                    tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            
            ALPrdDict<T> D;
            alprd_find_best<T>(tmp, len, D);
            
            if (enable_diag && dbg_modes) {
                uint64_t gid = vec_prefix[i] + (uint64_t)v;
                dbg_modes[gid] = 1;
                if (dbg_e) dbg_e[gid] = 0xFF;
                if (dbg_f) dbg_f[gid] = 0xFF;
            }
            
            alprd_vector_write<T>(bw, tmp, len, D);
        }
    }
}

// 解压缩（保持不变）
template<typename T>
__global__ void kernel_decompress(const uint8_t* bytes,
                                  const uint64_t* blk_starts_bits,
                                  const uint64_t* /*blk_bits*/,
                                  const uint64_t* out_starts,
                                  const int vectorSize,
                                  T* out_data, int numBlocks){
    int i = blockIdx.x;
    if (i>=numBlocks) return;

    BitReader br{bytes, blk_starts_bits[i]};
    int numVec = (int)br.getN(8);

    uint64_t out_pos = out_starts[i];
    for(int v=0; v<numVec; ++v){
        int useALP = br.get1();
        if (useALP){
            uint8_t e = (uint8_t)br.getN(8);
            uint8_t f = (uint8_t)br.getN(8);
            short bitw = (short)br.getN(16);
            long long FOR = (long long)br.getN(64);
            int n = (int)br.getN(32);
            assert(n <= MAX_VEC);
            for(int k=0;k<n;++k){
                uint64_t enc = br.getN(bitw);
                long long I = FOR + (long long)enc;
                double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
                out_data[out_pos + k] = (T)dec;
            }
            int exc = (int)br.getN(16);
            for(int t=0;t<exc;++t){
                uint64_t raw = std::is_same_v<T,double> ? br.getN(64) : br.getN(32);
                int pos = (int)br.getN(16);
                if constexpr (std::is_same_v<T,double>){
                    double val = *reinterpret_cast<double*>(&raw);
                    out_data[out_pos + pos] = (T)val;
                } else {
                    uint32_t rv = (uint32_t)raw;
                    float val = *reinterpret_cast<float*>(&rv);
                    out_data[out_pos + pos] = (T)val;
                }
            }
            out_pos += n;
        }else{
            int n = (int)br.getN(32);
            assert(n <= MAX_VEC);
            uint8_t rbw = (uint8_t)br.getN(8);
            uint64_t right[MAX_VEC]; uint16_t leftIdx[MAX_VEC];
            for(int k=0;k<n;++k){
                leftIdx[k] = (uint16_t)br.getN(DICT_BW);
                right[k]   = br.getN(rbw);
            }
            uint8_t lbw = uint8_t(sizeof(T)*8 - rbw);
            uint64_t dict[DICT_SZ];
            for(int k=0;k<DICT_SZ;++k) dict[k] = br.getN(lbw);

            int exc = (int)br.getN(16);
            uint16_t exc_pos[MAX_VEC]; uint64_t exc_left[MAX_VEC];
            for(int t=0;t<exc;++t){
                exc_left[t] = br.getN(lbw);
                exc_pos[t]  = (uint16_t)br.getN(16);
            }
            for(int k=0;k<n;++k){
                uint64_t left = (leftIdx[k]<DICT_SZ)? dict[leftIdx[k]] : 0ULL;
                uint64_t raw  = (left<<rbw) | right[k];
                if constexpr (std::is_same_v<T,double>){
                    double val = *reinterpret_cast<double*>(&raw);
                    out_data[out_pos + k] = (T)val;
                }else{
                    uint32_t r32 = (uint32_t)raw;
                    float val = *reinterpret_cast<float*>(&r32);
                    out_data[out_pos + k] = (T)val;
                }
            }
            for(int t=0;t<exc;++t){
                int p = exc_pos[t];
                uint64_t raw = (exc_left[t]<<rbw) | right[p];
                if constexpr (std::is_same_v<T,double>){
                    double val = *reinterpret_cast<double*>(&raw);
                    out_data[out_pos + p] = (T)val;
                }else{
                    uint32_t r32 = (uint32_t)raw;
                    float val = *reinterpret_cast<float*>(&r32);
                    out_data[out_pos + p] = (T)val;
                }
            }
            out_pos += n;
        }
    }
}

// 主压缩函数
template<typename T>
static Compressed compress_impl(const T* h_data, size_t n, const Params& p){
    Compressed c;
    if (n==0) { c.vectorSize = p.vectorSize; return c; }

    const int V = p.vectorSize;
    const int B = p.blockSize>0 ? p.blockSize : int(n);
    const int numBlocks = int( (n + B - 1)/B );

    std::vector<uint64_t> h_starts(numBlocks), h_sizes(numBlocks);
    size_t pos=0;
    for(int i=0;i<numBlocks;++i){
        h_starts[i]=pos;
        uint64_t sz = std::min<uint64_t>(B, n-pos);
        h_sizes[i]  = sz;
        pos += sz;
    }

    bool diag = (std::getenv("ALP_GPU_DIAG") != nullptr);

    T* d_data=nullptr; 
    cudaMalloc(&d_data, n*sizeof(T));
    cudaMemcpy(d_data, h_data, n*sizeof(T), cudaMemcpyHostToDevice);

    uint64_t *d_starts=nullptr, *d_sizes=nullptr;
    cudaMalloc(&d_starts, numBlocks*sizeof(uint64_t));
    cudaMalloc(&d_sizes,  numBlocks*sizeof(uint64_t));
    cudaMemcpy(d_starts, h_starts.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes,  h_sizes.data(),  numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 第一级采样
    ALPCombination* d_candidates = nullptr;
    int* d_num_candidates = nullptr;
    cudaMalloc(&d_candidates, 
               numBlocks * SamplingConfig::MAX_KEPT_COMBINATIONS * sizeof(ALPCombination));
    cudaMalloc(&d_num_candidates, numBlocks * sizeof(int));
    
    dim3 sample_grid((numBlocks + 31) / 32);
    dim3 sample_block(32);
    kernel_first_level_sampling<T><<<sample_grid, sample_block>>>(
        d_data, d_starts, d_sizes, numBlocks, V,
        d_candidates, d_num_candidates
    );
    cudaDeviceSynchronize();

    // 测量大小和决定模式 - 修复：使用合理的线程配置
    uint64_t* d_bits=nullptr;
    uint8_t* d_mode=nullptr;
    cudaMalloc(&d_bits, numBlocks*sizeof(uint64_t));
    cudaMalloc(&d_mode, numBlocks*sizeof(uint8_t));

    dim3 grid1((numBlocks + 31) / 32);  // 修复：正确的grid配置
    dim3 block1(32);                     // 修复：使用32个线程而非1个
    kernel_size_and_mode_with_sampling<T><<<grid1, block1>>>(
        d_data, d_starts, d_sizes,
        d_candidates, d_num_candidates,
        numBlocks, V, d_bits, d_mode
    );
    cudaDeviceSynchronize();

    std::vector<uint64_t> h_bits(numBlocks);
    std::vector<uint8_t> h_mode(numBlocks);
    cudaMemcpy(h_bits.data(), d_bits, numBlocks*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mode.data(), d_mode, numBlocks*sizeof(uint8_t), cudaMemcpyDeviceToHost);

    std::vector<uint64_t> h_off(numBlocks), padded_bits(numBlocks);
    uint64_t acc = 0;
    for (int i = 0; i < numBlocks; ++i) {
        h_off[i] = acc;
        uint64_t bits = h_bits[i];
        uint64_t pad = (i + 1 < numBlocks) ? ((32 - (bits & 31ULL)) & 31ULL) : 0ULL;
        padded_bits[i] = bits + pad;
        acc += padded_bits[i];
    }
    const uint64_t total_bits  = acc;
    const uint64_t total_bytes = (total_bits + 7) / 8;

    std::vector<uint64_t> h_vec_cnt(numBlocks), h_vec_prefix(numBlocks+1, 0);
    uint64_t total_vecs = 0;
    for (int i=0;i<numBlocks;++i){
        uint64_t cnt = (h_sizes[i] + (uint64_t)V - 1) / (uint64_t)V;
        h_vec_cnt[i] = cnt;
        h_vec_prefix[i+1] = h_vec_prefix[i] + cnt;
        total_vecs += cnt;
    }

    uint64_t* d_vec_prefix = nullptr;
    cudaMalloc(&d_vec_prefix, sizeof(uint64_t)*(numBlocks+1));
    cudaMemcpy(d_vec_prefix, h_vec_prefix.data(), sizeof(uint64_t)*(numBlocks+1), cudaMemcpyHostToDevice);

    uint8_t* d_out=nullptr;
    cudaMalloc(&d_out, total_bytes);
    cudaMemset(d_out, 0, total_bytes);
    
    uint64_t* d_off=nullptr;
    cudaMalloc(&d_off, numBlocks*sizeof(uint64_t));
    cudaMemcpy(d_off, h_off.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint8_t *d_dbg_modes=nullptr, *d_dbg_e=nullptr, *d_dbg_f=nullptr;
    if (diag && total_vecs>0){
        cudaMalloc(&d_dbg_modes, sizeof(uint8_t)*total_vecs);
        cudaMalloc(&d_dbg_e,     sizeof(uint8_t)*total_vecs);
        cudaMalloc(&d_dbg_f,     sizeof(uint8_t)*total_vecs);
        cudaMemset(d_dbg_modes, 0xFF, sizeof(uint8_t)*total_vecs);
        cudaMemset(d_dbg_e,     0xFF, sizeof(uint8_t)*total_vecs);
        cudaMemset(d_dbg_f,     0xFF, sizeof(uint8_t)*total_vecs);
    }

    // 压缩写入 - 修复：使用合理的线程配置
    kernel_compress_emit_with_sampling<T><<<grid1, block1>>>(  // 修复：同样的配置
        d_data, d_starts, d_sizes, d_off, d_mode,
        d_candidates, d_num_candidates,
        d_vec_prefix, d_dbg_modes, d_dbg_e, d_dbg_f,
        diag ? 1 : 0, numBlocks, V, d_out
    );
    cudaDeviceSynchronize();

    c.data.resize(total_bytes);
    cudaMemcpy(c.data.data(), d_out, total_bytes, cudaMemcpyDeviceToHost);
    c.offsets = std::move(h_off);
    c.bit_sizes = std::move(h_bits);
    c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
    c.vectorSize = V;

    if (diag && total_vecs>0){
        std::vector<uint8_t> dbg_modes(total_vecs, 0xFF), dbg_e(total_vecs, 0xFF), dbg_f(total_vecs, 0xFF);
        cudaMemcpy(dbg_modes.data(), d_dbg_modes, sizeof(uint8_t)*total_vecs, cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg_e.data(),     d_dbg_e,     sizeof(uint8_t)*total_vecs, cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg_f.data(),     d_dbg_f,     sizeof(uint8_t)*total_vecs, cudaMemcpyDeviceToHost);

        uint64_t alp_cnt=0, alprd_cnt=0;
        for (uint64_t i=0;i<total_vecs;++i){
            if (dbg_modes[i]==0) ++alp_cnt;
            else if (dbg_modes[i]==1) ++alprd_cnt;
        }
        std::cout << "[GPU-TwoLevel] Vector-mode: ALP="<<alp_cnt
                  << ", ALPrd="<<alprd_cnt << ", totalVec="<< total_vecs << "\n";
        
        std::cout << "[GPU-TwoLevel] First 10 vectors (e,f): ";
        int display_count = (total_vecs < 10) ? (int)total_vecs : 10;
        for (int i = 0; i < display_count; ++i) {
            if (dbg_modes[i] == 0) {
                std::cout << "(" << (int)dbg_e[i] << "," << (int)dbg_f[i] << ") ";
            } else {
                std::cout << "(ALPrd) ";
            }
        }
        std::cout << "\n";
    }

    if (d_dbg_modes) cudaFree(d_dbg_modes);
    if (d_dbg_e)     cudaFree(d_dbg_e);
    if (d_dbg_f)     cudaFree(d_dbg_f);
    if (d_vec_prefix) cudaFree(d_vec_prefix);
    
    cudaFree(d_candidates);
    cudaFree(d_num_candidates);
    cudaFree(d_out);
    cudaFree(d_off);
    cudaFree(d_mode);
    cudaFree(d_bits);
    cudaFree(d_sizes);
    cudaFree(d_starts);
    cudaFree(d_data);
    
    return c;
}

template<typename T>
static void decompress_impl(const Compressed& c, T* h_out, size_t n, const Params& p){
    if (n==0) return;
    const int numBlocks = (int)c.offsets.size();
    assert((size_t)numBlocks == c.elem_counts.size());

    uint8_t* d_bytes=nullptr;
    cudaMalloc(&d_bytes, c.data.size());
    cudaMemcpy(d_bytes, c.data.data(), c.data.size(), cudaMemcpyHostToDevice);

    uint64_t *d_boff=nullptr, *d_bsiz=nullptr, *d_ost=nullptr;
    cudaMalloc(&d_boff, numBlocks*sizeof(uint64_t));
    cudaMalloc(&d_bsiz, numBlocks*sizeof(uint64_t));
    cudaMemcpy(d_boff, c.offsets.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsiz, c.bit_sizes.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::vector<uint64_t> h_outStarts(numBlocks);
    uint64_t acc=0;
    for(int i=0;i<numBlocks;++i){
        h_outStarts[i]=acc;
        acc+=c.elem_counts[i];
    }
    assert(acc == n && "elem_counts 总和必须等于输出元素数");

    cudaMalloc(&d_ost, numBlocks*sizeof(uint64_t));
    cudaMemcpy(d_ost, h_outStarts.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);

    T* d_out=nullptr;
    cudaMalloc(&d_out, n*sizeof(T));

    dim3 gs(numBlocks), bs(1);
    kernel_decompress<T><<<gs,bs>>>(d_bytes, d_boff, d_bsiz, d_ost, p.vectorSize, d_out, numBlocks);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, n*sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_ost);
    cudaFree(d_bsiz);
    cudaFree(d_boff);
    cudaFree(d_bytes);
}

Compressed compress_double(const double* data, size_t n, const Params& p){ return compress_impl<double>(data,n,p); }
Compressed compress_float (const float*  data, size_t n, const Params& p){ return compress_impl<float >(data,n,p); }
void decompress_double(const Compressed& c, double* out, size_t n, const Params& p){ decompress_impl<double>(c,out,n,p); }
void decompress_float (const Compressed& c, float*  out, size_t n, const Params& p){ decompress_impl<float >(c,out,n,p); }

} // namespace alp_gpu