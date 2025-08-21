/*该版本代码的选择器与CPU略有不同，是尽可能少产生异常值的标准*/

#include "alp_gpu.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <climits> // LLONG_MAX, LLONG_MIN

using std::uint8_t; using std::uint32_t; using std::uint64_t;

namespace alp_gpu {

// ===================== 常量（与 CPU 版一致） =====================
// e,f 取值范围 0..18（19个）
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
// ALPrd：3bit 字典（大小 8），CUTTING_LIMIT = 16（与 CPU 保持）
static constexpr int   DICT_BW  = 3;
static constexpr int   DICT_SZ  = 1 << DICT_BW;
static constexpr int   CUT_LIM  = 16;
static constexpr int   MAX_VEC  = 4096; // 保护上限（<= 4096）

// ===================== 设备端位流 Writer/Reader =====================
struct BitWriter {
    uint8_t* buf;      // 全局输出缓冲（按 bit 偏移写）
    uint64_t bitpos;   // 写入起始 bit 偏移（每块各自独立）
    // __device__ void put1(int b){
    //     if (!b) { ++bitpos; return; } // 初始缓冲区已清零；只在写 1 时 OR 即可
    //     uint64_t byte = bitpos >> 3;
    //     int      off  = 7 - int(bitpos & 7ULL);
    //     unsigned int* p32 = reinterpret_cast<unsigned int*>(&buf[byte & ~3ULL]);
    //     unsigned int  m   = 1u << ((byte & 3ULL)*8 + off);
    //     atomicOr(p32, m);
    //     ++bitpos;
    // }
    // __device__ void putN(uint64_t v, int bits){
    //     for(int i=bits-1;i>=0;--i) put1( (v>>i) & 1ULL );
    // }
    __device__ void put1(int b){
        if (!b) { ++bitpos; return; }          // 初始缓冲区已清零
        uint64_t byte = bitpos >> 3;
        int      off  = 7 - int(bitpos & 7ULL);
        buf[byte] = uint8_t(buf[byte] | (uint8_t(1u) << off));  // 普通字节写
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

// ===================== 公共工具（GPU） =====================
__device__ __forceinline__ int width_needed_unsigned(unsigned long long range){
    if (range==0ULL) return 1;
    int c=0; while(range){ ++c; range>>=1ULL; } return c;
}

__device__ inline long long fast_round_double(double x){
    // 与 CPU 版相同的“甜 spot”整型回转
    const double SWEET = double((1ULL<<51) + (1ULL<<52));
    return (long long)(x + SWEET) - (long long)SWEET;
}

// ============= 采样 & 模式判定（与 CPU 逻辑对齐） =============
template<typename T>
__device__ inline bool alp_exact_equal(T v, uint8_t e, uint8_t f){
    if constexpr (std::is_same_v<T,double>) {
        double enc = v * D_EXP_ARR[e] * D_FRAC_ARR[f];
        long long I = fast_round_double(enc);
        // 解码：I * 10^{f} * 10^{-e} = I * (1/FRAC[f]) * FRAC[e]
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
__device__ inline bool is_high_precision_value(T v){
    for(uint8_t e=0;e<=18;++e){
        for(uint8_t f=0; f<=e; ++f){
            if (alp_exact_equal<T>(v,e,f)) return false;
        }
    }
    return true;
}

// 估计 ALP 异常率（对采样子集）
template<typename T>
__device__ inline double estimate_alp_exception_rate(const T* a, int n){
    int best_exc = n+1;
    for(uint8_t e=0;e<=18;++e){
        for(uint8_t f=0;f<=e;++f){
            int exc=0; long long mn=LLONG_MAX, mx=LLONG_MIN;
            for(int i=0;i<n;++i){
                double enc = double(a[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
                long long I = fast_round_double(enc);
                double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
                if (dec==double(a[i])) { mn = (mn<I?mn:I); mx = (mx>I?mx:I); }
                else ++exc;
            }
            if (exc < best_exc) best_exc = exc;
        }
    }
    return double(best_exc)/double(n);
}

template<typename T>
__device__ inline CompressionMode decide_mode_block(const T* blk, int n, int sample_stride){
    int total=0, highp=0;
    for(int i=0;i<n; i+=sample_stride){ ++total; if (is_high_precision_value<T>(blk[i])) ++highp; }
    if (total>0 && (double)highp/total > 0.5) return CompressionMode::ALPrd;

    const int SAMPLE_MAX = (32 < total ? 32 : total);
    if (SAMPLE_MAX<=0) return CompressionMode::ALP;
    T tmp[32];
    int k=0;
    for(int i=0;i<n && k<SAMPLE_MAX; i+=sample_stride) tmp[k++]=blk[i];
    double er = estimate_alp_exception_rate<T>(tmp, k);
    if (er>0.2) return CompressionMode::ALPrd;
    return CompressionMode::ALP;
}

// ============= ALP 单向量：统计 & 写入 =============
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
__device__ inline uint64_t alp_vector_size_bits(int n, uint8_t e, uint8_t f,
                                                short bitw, int exc_cnt){
    int val_bits = std::is_same_v<T,double> ? 64 : 32;
    return 1 + 8+8+16+64+32 + uint64_t(n)*bitw + 16 + uint64_t(exc_cnt)*(val_bits+16);
}

template<typename T>
__device__ inline void alp_vector_write(BitWriter& bw, const T* v, int n,
                                       uint8_t e, uint8_t f, short bitw, long long FOR){
    assert(n <= MAX_VEC);
    bw.put1(1); // useALP = 1
    bw.putN((uint64_t)e, 8); bw.putN((uint64_t)f, 8);
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
            bw.putN(0, bitw); // 占位
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

// ============= ALPrd：字典与写入（精确 top-8，本地小表，无 shared） =============
template<typename T> struct ALPrdDict {
    uint8_t rightBW;        // 右半位宽
    uint8_t leftBW;         // 左半位宽
    uint32_t dict[DICT_SZ]; // left parts 字典（<= 2^leftBW-1）
};

__device__ inline uint32_t mask_lo(int bits){
    return (bits >= 32) ? 0xFFFFFFFFu : ((1u<<bits) - 1u);
}

template<typename T>
__device__ inline void alprd_find_best(const uint64_t* in, int n, ALPrdDict<T>& D){
    // 穷举左宽 1..CUT_LIM，精确统计频次 → 取 top-8
    double best_score = 1e100; int best_rbw = int(sizeof(T)*8) - 1;
    uint32_t best_dict[DICT_SZ] = {0};

    for(int lbw=1; lbw<=CUT_LIM; ++lbw){
        int rbw = int(sizeof(T)*8) - lbw;
        uint32_t lmask = mask_lo(lbw);

        // 频次统计（最多 n=vectorSize 个不同 left），用小表（最多 n 项）
        uint32_t uniq_left[MAX_VEC]; int cnt[MAX_VEC];
        int u = 0;

        for(int i=0;i<n;++i){
            uint32_t left = (uint32_t)((in[i] >> rbw) & lmask);
            int j=0; for(; j<u; ++j) if (uniq_left[j]==left) { ++cnt[j]; break; }
            if (j==u){ uniq_left[u]=left; cnt[u]=1; ++u; }
        }
        // 选 top-8
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
        // 异常数量
        int keep=0;
        for(int k=0;k<used;++k){
            for(int j=0;j<u;++j) if (uniq_left[j]==dict[k]) { keep += cnt[j]; break; }
        }
        int exc = n - keep;

        // 估计位数：n*(DICT_BW+rbw) + dict(8*lbw) + 异常(16+lbw)*exc + 头(1+32+8)
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
    bw.put1(0); // useALP=0
    bw.putN((uint64_t)n, 32);
    bw.putN((uint64_t)D.rightBW, 8);

    // 逐值输出：leftIdx(3)+right(rbw)，同时记录异常
    int exc_cnt=0; uint16_t exc_pos[MAX_VEC]; uint32_t exc_left[MAX_VEC];
    uint64_t right_mask = (D.rightBW==64)? ~0ULL : ((1ULL<<D.rightBW)-1ULL);
    uint32_t left_mask  = mask_lo(D.leftBW);

    for(int i=0;i<n;++i){
        uint64_t right = in[i] & right_mask;
        uint32_t left  = (uint32_t)((in[i] >> D.rightBW) & left_mask);
        short idx = DICT_SZ;
        for(int k=0;k<DICT_SZ;++k){ if (D.dict[k]==left){ idx=(short)k; break; } }
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
    // 字典
    for(int k=0;k<DICT_SZ;++k) bw.putN((uint64_t)D.dict[k], D.leftBW);
    // 异常
    bw.putN((uint64_t)exc_cnt, 16);
    for(int i=0;i<exc_cnt;++i){
        bw.putN((uint64_t)exc_left[i], D.leftBW);
        bw.putN((uint64_t)exc_pos[i], 16);
    }
}

// ============= 每块：统计位数（含行组头 + 全向量） =============
template<typename T>
__device__ uint64_t measure_block_bits(const T* blk, int n, int vectorSize, CompressionMode& mode){
    int stride = (n/1024>0? n/1024 : 1);
    mode = decide_mode_block<T>(blk, n, stride);

    int numVec = (n + vectorSize - 1)/vectorSize;
    uint64_t bits = 8; // rowGroup.size()

    if (mode==CompressionMode::ALP){
        for(int v=0; v<numVec; ++v){
            int beg = v*vectorSize;
            int rem = n-beg;
            int len = (vectorSize < rem ? vectorSize : rem);
            uint8_t best_e=0, best_f=0; int best_exc=len+1; short bitw=0; long long FOR=0; int exc=0;
            for(uint8_t e=0;e<=18;++e){
                for(uint8_t f=0;f<=e;++f){
                    short _bitw; long long _FOR; int _exc;
                    alp_vector_analyze<T>(blk+beg, len, e, f, _bitw, _FOR, _exc);
                    if (_exc<best_exc){ best_exc=_exc; best_e=e; best_f=f; bitw=_bitw; FOR=_FOR; exc=_exc; }
                }
            }
            bits += alp_vector_size_bits<T>(len, best_e, best_f, bitw, exc);
        }
    }else{
        for(int v=0; v<numVec; ++v){
            int beg=v*vectorSize;
            int rem=n-beg;
            int len=(vectorSize<rem?vectorSize:rem);
            uint64_t tmp[MAX_VEC];
            assert(len <= MAX_VEC);
            for(int i=0;i<len;++i){
                if constexpr (std::is_same_v<T,double>) tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            ALPrdDict<T> D; alprd_find_best<T>(tmp, len, D);

            // 统计异常（用小映射）
            int exc=0;
            for(int i=0;i<len;++i){
                uint32_t left = (uint32_t)((tmp[i] >> D.rightBW) & mask_lo(D.leftBW));
                bool inDict=false; for(int k=0;k<DICT_SZ;++k) if (D.dict[k]==left){ inDict=true; break; }
                if (!inDict) ++exc;
            }
            bits += alprd_vector_size_bits<T>(len, D, exc);
        }
    }
    return bits;
}

// ============= 每块：实际写入（按 offset） =============
template<typename T>
__device__ void emit_block_bits(const T* blk, int n, int vectorSize,
                                CompressionMode mode, BitWriter& bw){
    int numVec = (n + vectorSize - 1)/vectorSize;
    bw.putN((uint64_t)numVec, 8); // 行组头

    if (mode==CompressionMode::ALP){
        for(int v=0; v<numVec; ++v){
            int beg=v*vectorSize;
            int rem=n-beg;
            int len=(vectorSize<rem?vectorSize:rem);
            uint8_t best_e=0, best_f=0; int best_exc=len+1; short bitw=0; long long FOR=0; int exc=0;
            for(uint8_t e=0;e<=18;++e){
                for(uint8_t f=0;f<=e;++f){
                    short _bitw; long long _FOR; int _exc;
                    alp_vector_analyze<T>(blk+beg, len, e, f, _bitw, _FOR, _exc);
                    if (_exc<best_exc){ best_exc=_exc; best_e=e; best_f=f; bitw=_bitw; FOR=_FOR; exc=_exc; }
                }
            }
            alp_vector_write<T>(bw, blk+beg, len, best_e, best_f, bitw, FOR);
        }
    } else {
        for(int v=0; v<numVec; ++v){
            int beg=v*vectorSize;
            int rem=n-beg;
            int len=(vectorSize<rem?vectorSize:rem);
            uint64_t tmp[MAX_VEC];
            assert(len <= MAX_VEC);
            for(int i=0;i<len;++i){
                if constexpr (std::is_same_v<T,double>) tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            ALPrdDict<T> D; alprd_find_best<T>(tmp, len, D);
            alprd_vector_write<T>(bw, tmp, len, D);
        }
    }
}

// ===================== Kernels：size & compress & decompress =====================
template<typename T>
__global__ void kernel_size_and_mode(const T* data, const uint64_t* blk_starts,
                                     const uint64_t* blk_sizes, int numBlocks,
                                     int vectorSize,
                                     uint64_t* out_bits, uint8_t* out_mode){
    int i = blockIdx.x; // 一线程一块
    if (i>=numBlocks) return;
    const T* blk = data + blk_starts[i];
    int n = (int)blk_sizes[i];
    CompressionMode mode;
    uint64_t bits = measure_block_bits<T>(blk, n, vectorSize, mode);
    out_bits[i] = bits;
    out_mode[i] = (mode==CompressionMode::ALPrd)?1:0;
}

template<typename T>
__global__ void kernel_compress_emit(const T* data,
                                     const uint64_t* blk_starts,
                                     const uint64_t* blk_sizes,
                                     const uint64_t* bit_offsets,
                                     const uint8_t*  modes,
                                     int numBlocks, int vectorSize,
                                     uint8_t* out_bytes){
    int i = blockIdx.x; // 一线程一块
    if (i>=numBlocks) return;
    const T* blk = data + blk_starts[i];
    int n = (int)blk_sizes[i];
    BitWriter bw{out_bytes, bit_offsets[i]};
    CompressionMode mode = (modes[i]?CompressionMode::ALPrd:CompressionMode::ALP);
    emit_block_bits<T>(blk, n, vectorSize, mode, bw);
}

template<typename T>
__global__ void kernel_decompress(const uint8_t* bytes,
                                  const uint64_t* blk_starts_bits,
                                  const uint64_t* /*blk_bits*/,
                                  const uint64_t* out_starts,
                                  const int vectorSize,
                                  T* out_data, int numBlocks){
    int i = blockIdx.x; // 一线程一块
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

// ===================== Host 侧封装 =====================
template<typename T>
static Compressed compress_impl(const T* h_data, size_t n, const Params& p){
    Compressed c;
    if (n==0) { c.vectorSize = p.vectorSize; return c; }

    // 切块：一个线程处理一个数据块
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

    // 上传输入与分块表
    T* d_data=nullptr; cudaMalloc(&d_data, n*sizeof(T));
    cudaMemcpy(d_data, h_data, n*sizeof(T), cudaMemcpyHostToDevice);

    uint64_t *d_starts=nullptr, *d_sizes=nullptr;
    cudaMalloc(&d_starts, numBlocks*sizeof(uint64_t));
    cudaMalloc(&d_sizes,  numBlocks*sizeof(uint64_t));
    cudaMemcpy(d_starts, h_starts.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes,  h_sizes.data(),  numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 第一阶段：测位数 & 模式（每块 1 线程）
    uint64_t* d_bits=nullptr;  uint8_t* d_mode=nullptr;
    cudaMalloc(&d_bits, numBlocks*sizeof(uint64_t));
    cudaMalloc(&d_mode, numBlocks*sizeof(uint8_t));

    dim3 gs(numBlocks), bs(1);
    kernel_size_and_mode<T><<<gs,bs>>>(d_data, d_starts, d_sizes, numBlocks, V, d_bits, d_mode);
    cudaDeviceSynchronize();

    std::vector<uint64_t> h_bits(numBlocks); std::vector<uint8_t> h_mode(numBlocks);
    cudaMemcpy(h_bits.data(), d_bits, numBlocks*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mode.data(), d_mode, numBlocks*sizeof(uint8_t),   cudaMemcpyDeviceToHost);

    // exclusive-scan 计算 bit 偏移
    // std::vector<uint64_t> h_off(numBlocks);
    // uint64_t acc=0; for(int i=0;i<numBlocks;++i){ h_off[i]=acc; acc+=h_bits[i]; }
    // const uint64_t total_bits  = acc;


    // --- change ---
    std::vector<uint64_t> h_off(numBlocks), padded_bits(numBlocks);
    uint64_t acc = 0;
    for (int i = 0; i < numBlocks; ++i) {
        h_off[i] = acc;
        uint64_t bits = h_bits[i];
        // 对除了最后一块之外，都向上补齐到 32bit 边界
        uint64_t pad = (i + 1 < numBlocks) ? ((32 - (bits & 31ULL)) & 31ULL) : 0ULL;
        padded_bits[i] = bits + pad;
        acc += padded_bits[i];
    }
    const uint64_t total_bits  = acc;
    const uint64_t total_bytes = (total_bits + 7) / 8;



    // 分配输出 buffer
    uint8_t* d_out=nullptr; cudaMalloc(&d_out, total_bytes);
    cudaMemset(d_out, 0, total_bytes);
    uint64_t* d_off=nullptr; cudaMalloc(&d_off, numBlocks*sizeof(uint64_t));
    cudaMemcpy(d_off, h_off.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 第二阶段：真正写入
    kernel_compress_emit<T><<<gs,bs>>>(d_data, d_starts, d_sizes, d_off, d_mode, numBlocks, V, d_out);
    cudaDeviceSynchronize();

    // 拷回结果 & per-block 元信息
    c.data.resize(total_bytes);
    cudaMemcpy(c.data.data(), d_out, total_bytes, cudaMemcpyDeviceToHost);
    c.offsets = std::move(h_off);
    c.bit_sizes = std::move(h_bits);
    c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
    c.vectorSize = V;

    // 清理
    cudaFree(d_out); cudaFree(d_off);
    cudaFree(d_mode); cudaFree(d_bits);
    cudaFree(d_sizes); cudaFree(d_starts);
    cudaFree(d_data);

    return c;
}

template<typename T>
static void decompress_impl(const Compressed& c, T* h_out, size_t n, const Params& p){
    if (n==0) return;
    const int numBlocks = (int)c.offsets.size();
    assert((size_t)numBlocks == c.elem_counts.size());

    // 上传压缩位流与 per-block 信息
    uint8_t* d_bytes=nullptr; cudaMalloc(&d_bytes, c.data.size());
    cudaMemcpy(d_bytes, c.data.data(), c.data.size(), cudaMemcpyHostToDevice);

    uint64_t *d_boff=nullptr, *d_bsiz=nullptr, *d_ost=nullptr;
    cudaMalloc(&d_boff, numBlocks*sizeof(uint64_t));
    cudaMalloc(&d_bsiz, numBlocks*sizeof(uint64_t));
    cudaMemcpy(d_boff, c.offsets.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsiz, c.bit_sizes.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 由 elem_counts 前缀和得到每块输出起点
    std::vector<uint64_t> h_outStarts(numBlocks);
    uint64_t acc=0; for(int i=0;i<numBlocks;++i){ h_outStarts[i]=acc; acc+=c.elem_counts[i]; }
    assert(acc == n && "elem_counts 总和必须等于输出元素数");

    cudaMalloc(&d_ost, numBlocks*sizeof(uint64_t));
    cudaMemcpy(d_ost, h_outStarts.data(), numBlocks*sizeof(uint64_t), cudaMemcpyHostToDevice);

    T* d_out=nullptr; cudaMalloc(&d_out, n*sizeof(T));

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

// 显式实例化 API
Compressed compress_double(const double* data, size_t n, const Params& p){ return compress_impl<double>(data,n,p); }
Compressed compress_float (const float*  data, size_t n, const Params& p){ return compress_impl<float >(data,n,p); }
void decompress_double(const Compressed& c, double* out, size_t n, const Params& p){ decompress_impl<double>(c,out,n,p); }
void decompress_float (const Compressed& c, float*  out, size_t n, const Params& p){ decompress_impl<float >(c,out,n,p); }

} // namespace alp_gpu