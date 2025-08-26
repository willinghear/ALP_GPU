// /*
//  * ============================================
//  *  ALP-GPU 压缩/解压（修复版）
//  *  主要修复：
//  *    1) 内存越界问题：增加安全边距和边界检查
//  *    2) BitWriter/Reader同步问题：强制串行化
//  *    3) CUDA错误检查：添加完整错误处理
//  *    4) 大小估算问题：更保守的估算算法
//  * ============================================
//  */

// #include "alp_gpu.hpp"
// #include <cuda_runtime.h>
// #include <cassert>
// #include <cmath>
// #include <algorithm>
// #include <numeric>
// #include <stdexcept>
// #include <iostream>
// #include <climits>
// #include <cstdlib>
// #include <vector>

// using std::uint8_t; using std::uint32_t; using std::uint64_t;

// // CUDA错误检查宏
// #define CUDA_CHECK(call) \
//     do { \
//         cudaError_t error = call; \
//         if (error != cudaSuccess) { \
//             std::cerr << "[CUDA Error] " << __FILE__ << ":" << __LINE__ \
//                       << " " << cudaGetErrorString(error) << std::endl; \
//         } \
//     } while(0)

// namespace alp_gpu {

// // ===================== 两级采样配置 =====================
// namespace sampling_config {
//     static constexpr int ROWGROUP_SIZE = 100000;
//     static constexpr int ROWGROUP_VECTOR_SAMPLES = 8;
//     static constexpr int SAMPLES_PER_VECTOR = 32;
//     static constexpr int MAX_K_COMBINATIONS = 5;
//     static constexpr int EARLY_EXIT_THRESHOLD = 2;
// }

// template<typename T> struct SamplingConstants {
//     static constexpr size_t RD_SIZE_THRESHOLD_LIMIT = 
//         sizeof(T) == 8 ? (48 * sampling_config::SAMPLES_PER_VECTOR) 
//                        : (22 * sampling_config::SAMPLES_PER_VECTOR);
// };

// // ===================== 常量 =====================
// __device__ __constant__ double D_EXP_ARR[19] = {
//   1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
//   100000000.0, 1000000000.0, 10000000000.0, 100000000000.0, 1000000000000.0,
//   10000000000000.0, 100000000000000.0, 1000000000000000.0,
//   10000000000000000.0, 100000000000000000.0, 1000000000000000000.0
// };
// __device__ __constant__ double D_FRAC_ARR[20] = {
//   1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
//   0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.000000000001,
//   0.0000000000001, 0.00000000000001, 0.000000000000001, 0.0000000000000001,
//   0.00000000000000001, 0.000000000000000001
// };

// static constexpr int   DICT_BW  = 3;
// static constexpr int   DICT_SZ  = 1 << DICT_BW;
// static constexpr int   CUT_LIM  = 16;
// static constexpr int   MAX_VEC  = 4096;

// // ===================== 安全的设备端位流 Writer/Reader =====================
// struct SafeBitWriter {
//     uint8_t* buf;
//     uint64_t bitpos;
//     uint64_t max_bits;  // 最大可写位数
    
//     __device__ SafeBitWriter(uint8_t* buffer, uint64_t start_bit, uint64_t buffer_size_bits)
//         : buf(buffer), bitpos(start_bit), max_bits(start_bit + buffer_size_bits) {}
    
//     __device__ bool put1(int b){
//         if (bitpos >= max_bits) {
//             printf("[DEVICE-ERROR] BitWriter overflow at bit %llu (max %llu)\n", bitpos, max_bits);
//             return false;
//         }
//         if (!b) { ++bitpos; return true; }
        
//         uint64_t byte = bitpos >> 3;
//         int off = 7 - int(bitpos & 7ULL);
//         buf[byte] |= (uint8_t(1u) << off);
//         ++bitpos;
//         return true;
//     }
    
//     __device__ bool putN(uint64_t v, int bits){
//         if (bits <= 0 || bits > 64) return false;
//         if (bitpos + bits > max_bits) {
//             printf("[DEVICE-ERROR] BitWriter would overflow: pos=%llu + bits=%d > max=%llu\n", 
//                    bitpos, bits, max_bits);
//             return false;
//         }
//         for(int i=bits-1;i>=0;--i) {
//             if (!put1( (v>>i) & 1ULL )) return false;
//         }
//         return true;
//     }
    
//     __device__ uint64_t get_pos() const { return bitpos; }
//     __device__ uint64_t remaining_bits() const { return max_bits - bitpos; }
// };

// struct BitReader {
//     const uint8_t* buf;
//     uint64_t bitpos;
//     __device__ int get1(){
//         uint64_t byte = bitpos >> 3;
//         int off = 7 - int(bitpos & 7ULL);
//         int b = (buf[byte] >> off) & 1;
//         ++bitpos; return b;
//     }
//     __device__ uint64_t getN(int bits){
//         uint64_t v=0;
//         for(int i=0;i<bits;++i){ v = (v<<1) | get1(); }
//         return v;
//     }
//     __device__ uint64_t get_pos() const { return bitpos; }
// };

// // ===================== 工具函数 =====================
// __device__ __forceinline__ int width_needed_unsigned(unsigned long long range){
//     if (range==0ULL) return 1;
//     int c=0; while(range){ ++c; range>>=1ULL; } return c;
// }

// __device__ inline long long fast_round_double(double x){
//     const double SWEET = double((1ULL<<51) + (1ULL<<52));
//     return (long long)(x + SWEET) - (long long)SWEET;
// }

// __device__ inline uint32_t mask_lo(int bits){
//     return (bits >= 32) ? 0xFFFFFFFFu : ((1u<<bits) - 1u);
// }

// // ===================== ALP 判断与分析 =====================
// template<typename T>
// __device__ inline bool alp_exact_equal(T v, uint8_t e, uint8_t f){
//     if constexpr (std::is_same_v<T,double>) {
//         double enc = v * D_EXP_ARR[e] * D_FRAC_ARR[f];
//         long long I = fast_round_double(enc);
//         double dec = double(I) * (1.0 / D_FRAC_ARR[f]) * D_FRAC_ARR[e];
//         return dec==v;
//     } else {
//         float enc = v * float(D_EXP_ARR[e]) * float(D_FRAC_ARR[f]);
//         int   I   = __float2int_rn(enc);
//         float dec = float(I) * (1.0f/float(D_FRAC_ARR[f])) * float(D_FRAC_ARR[e]);
//         return dec==v;
//     }
// }

// template<typename T>
// __device__ inline void alp_vector_analyze(const T* v, int n, uint8_t e, uint8_t f,
//                                           short& bitw, long long& FOR,
//                                           int& exc_cnt){
//     long long mn=LLONG_MAX, mx=LLONG_MIN;
//     exc_cnt=0;
//     for(int i=0;i<n;++i){
//         double enc = double(v[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
//         long long I = fast_round_double(enc);
//         double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
//         if (dec==double(v[i])) { mn=(mn<I?mn:I); mx=(mx>I?mx:I); }
//         else ++exc_cnt;
//     }
//     unsigned long long range = (mn==LLONG_MAX)? 0ULL : (unsigned long long)(mx - mn);
//     bitw = (short)width_needed_unsigned(range);
//     FOR  = (mn==LLONG_MAX?0:mn);
// }

// template<typename T>
// __device__ inline void alp_vector_choose_best_bits(
//     const T* v, int n,
//     uint8_t& best_e, uint8_t& best_f,
//     short& bitw, long long& FOR, int& exc)
// {
//     const int val_bits = std::is_same_v<T,double> ? 64 : 32;
//     double best_score = 1e300;
//     best_e=0; best_f=0; bitw=0; FOR=0; exc=0;

//     for(uint8_t e=0;e<=18;++e){
//         for(uint8_t f=0;f<=e;++f){
//             short _bw; long long _FOR; int _exc;
//             alp_vector_analyze<T>(v, n, e, f, _bw, _FOR, _exc);
//             double score = double(n)*_bw + double(_exc)*(val_bits + 16);
//             if (score < best_score){
//                 best_score = score;
//                 best_e = e; best_f = f; bitw = _bw; FOR = _FOR; exc = _exc;
//             }
//         }
//     }
// }

// // 修复：更保守的大小估算，增加安全边距
// template<typename T>
// __device__ inline uint64_t alp_vector_size_bits_safe(int n, uint8_t e, uint8_t f,
//                                                      short bitw, int exc_cnt){
//     const int val_bits = std::is_same_v<T,double> ? 64 : 32;
    
//     // ALP向量精确位数：145 + n*bitw + exc_cnt*(val_bits+16)
//     return 145ULL + uint64_t(n) * uint64_t(bitw) + uint64_t(exc_cnt) * (val_bits + 16);
// }

// template<typename T>
// __device__ inline bool alp_vector_write_safe(SafeBitWriter& bw, const T* v, int n,
//                                             uint8_t e, uint8_t f, short bitw, long long FOR){
//     assert(n <= MAX_VEC);

//     if (!bw.put1(1)) return false; // useALP = 1
//     if (!bw.putN((uint64_t)e, 8)) return false;
//     if (!bw.putN((uint64_t)f, 8)) return false;
//     if (!bw.putN((uint64_t)bitw, 16)) return false;
//     if (!bw.putN((uint64_t)FOR, 64)) return false;
//     if (!bw.putN((uint64_t)n, 32)) return false;

//     int exc_cnt=0;
//     int      exc_pos[MAX_VEC];
//     uint64_t exc_val[MAX_VEC];

//     for(int i=0;i<n;++i){
//         double enc = double(v[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
//         long long I = fast_round_double(enc);
//         double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
//         if (dec==double(v[i])) {
//             uint64_t packed = (uint64_t)(I - FOR);
//             if (!bw.putN(packed, bitw)) return false;
//         } else {
//             if (!bw.putN(0, bitw)) return false;
//             if constexpr (std::is_same_v<T,double>) {
//                 uint64_t raw = *reinterpret_cast<const uint64_t*>(&v[i]);
//                 exc_val[exc_cnt] = raw;
//             } else {
//                 uint32_t raw = *reinterpret_cast<const uint32_t*>(&v[i]);
//                 exc_val[exc_cnt] = raw;
//             }
//             exc_pos[exc_cnt] = i;
//             ++exc_cnt;
//         }
//     }
//     if (!bw.putN((uint64_t)exc_cnt, 16)) return false;
//     for(int k=0;k<exc_cnt;++k){
//         if constexpr (std::is_same_v<T,double>) {
//             if (!bw.putN(exc_val[k], 64)) return false;
//         } else {
//             if (!bw.putN(exc_val[k], 32)) return false;
//         }
//         if (!bw.putN((uint64_t)exc_pos[k], 16)) return false;
//     }
//     return true;
// }

// // ===================== ALPrd 结构与函数 =====================
// template<typename T> struct ALPrdDict {
//     uint8_t rightBW;
//     uint8_t leftBW;
//     uint32_t dict[DICT_SZ];
// };

// template<typename T>
// __device__ inline void alprd_find_best(const uint64_t* in, int n, ALPrdDict<T>& D){
//     double best_score = 1e100; int best_rbw = int(sizeof(T)*8) - 1;
//     uint32_t best_dict[DICT_SZ] = {0};

//     for(int lbw=1; lbw<=CUT_LIM; ++lbw){
//         int rbw = int(sizeof(T)*8) - lbw;
//         uint32_t lmask = mask_lo(lbw);

//         uint32_t uniq_left[MAX_VEC]; int cnt[MAX_VEC];
//         int u = 0;

//         for(int i=0;i<n;++i){
//             uint32_t left = (uint32_t)((in[i] >> rbw) & lmask);
//             int j=0; for(; j<u; ++j) if (uniq_left[j]==left) { ++cnt[j]; break; }
//             if (j==u){ uniq_left[u]=left; cnt[u]=1; ++u; }
//         }
        
//         uint32_t dict[DICT_SZ]={0};
//         int used = (DICT_SZ < u ? DICT_SZ : u);
//         for(int k=0;k<used;++k){
//             int best=-1, id=-1;
//             for(int j=0;j<u;++j){
//                 bool taken=false;
//                 for(int t=0;t<k;++t) if (dict[t]==uniq_left[j]) { taken=true; break; }
//                 if (taken) continue;
//                 if (cnt[j]>best){ best=cnt[j]; id=j; }
//             }
//             dict[k] = uniq_left[id];
//         }
        
//         int keep=0;
//         for(int k=0;k<used;++k){
//             for(int j=0;j<u;++j) if (uniq_left[j]==dict[k]) { keep += cnt[j]; break; }
//         }
//         int exc = n - keep;

//         double bits = 1 + 32 + 8 + double(n)*(DICT_BW + rbw) + double(DICT_SZ)*lbw + 16.0*exc + double(lbw)*exc;

//         if (bits < best_score){
//             best_score = bits;
//             best_rbw   = rbw;
//             for(int k=0;k<DICT_SZ;++k) best_dict[k]=dict[k];
//         }
//     }
//     D.rightBW = (uint8_t)best_rbw;
//     D.leftBW  = (uint8_t)(int(sizeof(T)*8) - best_rbw);
//     for(int k=0;k<DICT_SZ;++k) D.dict[k]=best_dict[k];
// }

// template<typename T>
// __device__ inline uint64_t alprd_vector_size_bits_safe(int n, const ALPrdDict<T>& D, int exc_cnt){
//     // ALPrd向量精确格式：1+32+8 + n*(DICT_BW+rightBW) + DICT_SZ*leftBW + 16 + exc_cnt*(leftBW+16) = 57 + ...
//     uint64_t base = 57ULL + uint64_t(n)*(DICT_BW + D.rightBW) + DICT_SZ*D.leftBW + uint64_t(exc_cnt)*(D.leftBW+16);

//     return base ;//+ margin;
// }

// template<typename T>
// __device__ inline bool alprd_vector_write_safe(SafeBitWriter& bw, const uint64_t* in, int n,
//                                               const ALPrdDict<T>& D){
//     assert(n <= MAX_VEC);
    
//     // // 预估需要位数
//     // uint64_t estimated_bits = 1 + 32 + 8 + uint64_t(n)*(DICT_BW + D.rightBW) + DICT_SZ*D.leftBW ;//+ 200;
//     // if (bw.remaining_bits() < estimated_bits) {
//     //     printf("[DEVICE-ERROR] Insufficient space for ALPrd vector\n");
//     //     return false;
//     // }
    
//     if (!bw.put1(0)) return false; // useALP=0
//     if (!bw.putN((uint64_t)n, 32)) return false;
//     if (!bw.putN((uint64_t)D.rightBW, 8)) return false;

//     int exc_cnt=0; uint16_t exc_pos[MAX_VEC]; uint32_t exc_left[MAX_VEC];
//     uint64_t right_mask = (D.rightBW==64)? ~0ULL : ((1ULL<<D.rightBW)-1ULL);
//     uint32_t left_mask  = mask_lo(D.leftBW);

//     for(int i=0;i<n;++i){
//         uint64_t right = in[i] & right_mask;
//         uint32_t left  = (uint32_t)((in[i] >> D.rightBW) & left_mask);
//         short idx = DICT_SZ;
//         for(int k=0;k<DICT_SZ;++k){ if (D.dict[k]==left){ idx=(short)k; break; } }
//         if (idx<DICT_SZ){
//             if (!bw.putN((uint64_t)idx, DICT_BW)) return false;
//             if (!bw.putN(right, D.rightBW)) return false;
//         }else{
//             if (!bw.putN(0, DICT_BW)) return false;
//             if (!bw.putN(right, D.rightBW)) return false;
//             exc_pos[exc_cnt]  = (uint16_t)i;
//             exc_left[exc_cnt] = left;
//             ++exc_cnt;
//         }
//     }
    
//     for(int k=0;k<DICT_SZ;++k) {
//         if (!bw.putN((uint64_t)D.dict[k], D.leftBW)) return false;
//     }
    
//     if (!bw.putN((uint64_t)exc_cnt, 16)) return false;
//     for(int i=0;i<exc_cnt;++i){
//         if (!bw.putN((uint64_t)exc_left[i], D.leftBW)) return false;
//         if (!bw.putN((uint64_t)exc_pos[i], 16)) return false;
//     }
//     return true;
// }

// // ===================== 采样函数 =====================
// template<typename T>
// __device__ void rowgroup_sample_and_find_k_combinations(
//     const T* rowgroup_data, 
//     int rowgroup_size,
//     int vectorSize,
//     EFCombination* best_k_combinations,
//     int& k_actual,
//     CompressionMode& mode
// ) {
//     int total_vectors = (rowgroup_size + vectorSize - 1) / vectorSize;
//     int sample_stride = max(1, total_vectors / sampling_config::ROWGROUP_VECTOR_SAMPLES);
    
//     struct LocalStats {
//         int count;
//         double total_score;
//     } stats[19][19];
    
//     for(int e=0; e<=18; e++) {
//         for(int f=0; f<=e; f++) {
//             stats[e][f].count = 0;
//             stats[e][f].total_score = 0;
//         }
//     }
    
//     double best_overall_compression_size = 1e30;
//     int samples_taken = 0;
    
//     for(int v = 0; v < total_vectors && samples_taken < sampling_config::ROWGROUP_VECTOR_SAMPLES; 
//         v += sample_stride) {
        
//         int vec_start = v * vectorSize;
//         int vec_size = min(vectorSize, rowgroup_size - vec_start);
//         if(vec_size <= 0) break;
        
//         T samples[32];
//         int sample_count = min(sampling_config::SAMPLES_PER_VECTOR, vec_size);
//         int sample_step = max(1, vec_size / sample_count);
        
//         for(int i = 0; i < sample_count; i++) {
//             samples[i] = rowgroup_data[vec_start + i * sample_step];
//         }
        
//         uint8_t best_e = 0, best_f = 0;
//         short bitw; long long FOR; int exc;
//         alp_vector_choose_best_bits<T>(samples, sample_count, best_e, best_f, bitw, FOR, exc);
        
//         int val_bits = std::is_same_v<T,double> ? 64 : 32;
//         double compression_size = sample_count * bitw + exc * (val_bits + 16);
        
//         stats[best_e][best_f].count++;
//         stats[best_e][best_f].total_score += compression_size;
        
//         if(compression_size < best_overall_compression_size) {
//             best_overall_compression_size = compression_size;
//         }
        
//         samples_taken++;
//     }
    
//     if(best_overall_compression_size >= SamplingConstants<T>::RD_SIZE_THRESHOLD_LIMIT) {
//         mode = CompressionMode::ALPrd;
//         k_actual = 0;
//         return;
//     }
    
//     mode = CompressionMode::ALP;
    
//     EFCombination all_combinations[361];  
//     int num_combinations = 0;
    
//     for(int e = 0; e <= 18; e++) {
//         for(int f = 0; f <= e; f++) {
//             if(stats[e][f].count > 0) {
//                 all_combinations[num_combinations].e = e;
//                 all_combinations[num_combinations].f = f;
//                 all_combinations[num_combinations].count = stats[e][f].count;
//                 all_combinations[num_combinations].score = 
//                     stats[e][f].total_score / stats[e][f].count;
//                 num_combinations++;
//             }
//         }
//     }
    
//     for(int i = 0; i < num_combinations - 1; i++) {
//         for(int j = i + 1; j < num_combinations; j++) {
//             bool swap = false;
//             if(all_combinations[j].count > all_combinations[i].count) {
//                 swap = true;
//             } else if(all_combinations[j].count == all_combinations[i].count) {
//                 if(all_combinations[j].score < all_combinations[i].score) {
//                     swap = true;
//                 }
//             }
            
//             if(swap) {
//                 EFCombination tmp = all_combinations[i];
//                 all_combinations[i] = all_combinations[j];
//                 all_combinations[j] = tmp;
//             }
//         }
//     }
    
//     k_actual = min(sampling_config::MAX_K_COMBINATIONS, num_combinations);
//     for(int i = 0; i < k_actual; i++) {
//         best_k_combinations[i] = all_combinations[i];
//     }
// }

// template<typename T>
// __device__ void vector_choose_from_k_combinations(
//     const T* vec_data,
//     int vec_size,
//     const EFCombination* k_combinations,
//     int k,
//     uint8_t& best_e,
//     uint8_t& best_f,
//     short& bitw,
//     long long& FOR,
//     int& exc
// ) {
//     if(k == 1) {
//         best_e = k_combinations[0].e;
//         best_f = k_combinations[0].f;
//         alp_vector_analyze<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
//         return;
//     }
    
//     T samples[32];
//     int sample_count = min(sampling_config::SAMPLES_PER_VECTOR, vec_size);
//     int sample_step = max(1, vec_size / sample_count);
    
//     for(int i = 0; i < sample_count; i++) {
//         samples[i] = vec_data[i * sample_step];
//     }
    
//     double best_score = 1e30;
//     int worse_count = 0;
    
//     for(int kid = 0; kid < k; kid++) {
//         uint8_t e = k_combinations[kid].e;
//         uint8_t f = k_combinations[kid].f;
        
//         short test_bitw;
//         long long test_FOR;
//         int test_exc;
//         alp_vector_analyze<T>(samples, sample_count, e, f, test_bitw, test_FOR, test_exc);
        
//         int val_bits = std::is_same_v<T,double> ? 64 : 32;
//         double score = sample_count * test_bitw + test_exc * (val_bits + 16);
        
//         if(score < best_score) {
//             best_score = score;
//             best_e = e;
//             best_f = f;
//             worse_count = 0;
//         } else {
//             worse_count++;
//             if(worse_count >= sampling_config::EARLY_EXIT_THRESHOLD) {
//                 break;
//             }
//         }
//     }
    
//     alp_vector_analyze<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
// }

// // ===================== 修复的Kernels =====================
// static constexpr int THREADS_PER_BLOCK = 128;
// static constexpr int MAX_VECS_PER_BLOCK = 256;

// // 修复的测量kernel：使用保守的大小估算
// template<typename T>
// __global__ void kernel_measure_with_sampling_safe(
//     const T* data, 
//     const uint64_t* blk_starts,
//     const uint64_t* blk_sizes, 
//     int numBlocks,
//     int vectorSize,
//     uint64_t* out_bits,     
//     uint8_t* out_mode      
// ) {
//     int blockId = blockIdx.x;
//     if (blockId >= numBlocks) return;
    
//     const T* blk = data + blk_starts[blockId];
//     int n = (int)blk_sizes[blockId];
//     int numVec = (n + vectorSize - 1) / vectorSize;
    
//     EFCombination k_combinations[5];
//     int k_actual = 0;
//     CompressionMode mode;
    
//     if(threadIdx.x == 0) {
//         rowgroup_sample_and_find_k_combinations<T>(
//             blk, n, vectorSize,
//             k_combinations, k_actual, mode
//         );
//     }
//     __syncthreads();
    
//     __shared__ CompressionMode sh_mode;
//     __shared__ EFCombination sh_k_combinations[5];
//     __shared__ int sh_k_actual;
    
//     if(threadIdx.x == 0) {
//         sh_mode = mode;
//         sh_k_actual = k_actual;
//         for(int i = 0; i < k_actual; i++) {
//             sh_k_combinations[i] = k_combinations[i];
//         }
//     }
//     __syncthreads();
    
//     __shared__ uint64_t sh_total_bits;
//     if(threadIdx.x == 0) {
//         sh_total_bits = 8;  // 行组头
//     }
//     __syncthreads();
    
//     if(sh_mode == CompressionMode::ALP) {
//         for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
//             int beg = v * vectorSize;
//             int rem = n - beg;
//             int len = (vectorSize < rem ? vectorSize : rem);
            
//             uint8_t e, f; short bw; long long FOR; int exc;
//             vector_choose_from_k_combinations<T>(
//                 blk + beg, len,
//                 sh_k_combinations, sh_k_actual,
//                 e, f, bw, FOR, exc
//             );
            
//             // 使用更保守的大小估算
//             uint64_t bits = alp_vector_size_bits_safe<T>(len, e, f, bw, exc);
//             atomicAdd((unsigned long long*)&sh_total_bits, bits);
//         }
//     } else {
//         for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
//             int beg = v * vectorSize;
//             int rem = n - beg;
//             int len = (vectorSize < rem ? vectorSize : rem);
            
//             uint64_t tmp[MAX_VEC];
//             for(int i = 0; i < len; i++) {
//                 if constexpr (std::is_same_v<T,double>) 
//                     tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
//                 else 
//                     tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
//             }
            
//             ALPrdDict<T> D;
//             alprd_find_best<T>(tmp, len, D);
            
//             int exc = 0;
//             for(int i = 0; i < len; i++) {
//                 uint32_t left = (uint32_t)((tmp[i] >> D.rightBW) & mask_lo(D.leftBW));
//                 bool inDict = false;
//                 for(int k = 0; k < DICT_SZ; k++) {
//                     if(D.dict[k] == left) {
//                         inDict = true;
//                         break;
//                     }
//                 }
//                 if(!inDict) exc++;
//             }
            
//             uint64_t bits = alprd_vector_size_bits_safe<T>(len, D, exc);
//             atomicAdd((unsigned long long*)&sh_total_bits, bits);
//         }
//     }
    
//     __syncthreads();
    
//     if(threadIdx.x == 0) {
//         out_bits[blockId] = sh_total_bits;
//         out_mode[blockId] = (sh_mode == CompressionMode::ALPrd) ? 1 : 0;
//     }
// }

// // 修复的emit kernel：完全串行化并使用安全BitWriter
// template<typename T>
// __global__ void kernel_emit_with_sampling_safe(
//     const T* data,
//     const uint64_t* blk_starts,
//     const uint64_t* blk_sizes,
//     const uint64_t* bit_offsets,
//     const uint64_t* bit_budgets,
//     const uint8_t* modes,
//     const uint64_t* vec_prefix,
//     uint8_t* dbg_modes, 
//     uint8_t* dbg_e, 
//     uint8_t* dbg_f,
//     int enable_diag,
//     int numBlocks, 
//     int vectorSize,
//     uint8_t* out_bytes
// ) {
//     int blockId = blockIdx.x;
//     if (blockId >= numBlocks) return;
    
//     // 强制串行化，避免竞争条件
//     if (threadIdx.x != 0) return;
    
//     const T* blk = data + blk_starts[blockId];
//     int n = (int)blk_sizes[blockId];
//     int numVec = (n + vectorSize - 1) / vectorSize;
//     CompressionMode mode = (modes[blockId] ? CompressionMode::ALPrd : CompressionMode::ALP);
    
//     // 使用安全的BitWriter
//     SafeBitWriter bw(out_bytes, bit_offsets[blockId], bit_budgets[blockId]);
    
//     if (blockId == 0) {
//         printf("[DEVICE-EMIT] Block0: n=%d, numVec=%d, bit_offset=%llu, budget=%llu\n", 
//                n, numVec, bit_offsets[blockId], bit_budgets[blockId]);
//     }
    
//     if (numVec <= 0) return;
    
//     // 写入行组头
//     if (!bw.putN((uint64_t)numVec, 8)) {
//         printf("[DEVICE-ERROR] Failed to write numVec header\n");
//         return;
//     }
    
//     // 重新采样获取k个组合
//     EFCombination k_combinations[5] = {{0,0,0,0}};
//     int k_actual = 1;
//     CompressionMode mode_check;
//     rowgroup_sample_and_find_k_combinations<T>(
//         blk, n, vectorSize,
//         k_combinations, k_actual, mode_check
//     );
    
//     if (mode == CompressionMode::ALP) {
//         for(int v = 0; v < numVec; v++) {
//             int beg = v * vectorSize;
//             int rem = n - beg;
//             int len = (vectorSize < rem ? vectorSize : rem);
            
//             uint64_t before_pos = bw.get_pos();
            
//             // 检查剩余空间
//             if (bw.remaining_bits() < 1000) {
//                 printf("[DEVICE-ERROR] Insufficient space for vector %d\n", v);
//                 break;
//             }
            
//             uint8_t e, f; short bitw; long long FOR; int exc;
//             vector_choose_from_k_combinations<T>(
//                 blk + beg, len,
//                 k_combinations, k_actual,
//                 e, f, bitw, FOR, exc
//             );
            
//             // 记录调试信息
//             if (enable_diag && dbg_modes) {
//                 uint64_t gid = vec_prefix[blockId] + (uint64_t)v;
//                 dbg_modes[gid] = 0;  // ALP
//                 if (dbg_e) dbg_e[gid] = e;
//                 if (dbg_f) dbg_f[gid] = f;
//             }
            
//             // 安全写入
//             if (!alp_vector_write_safe<T>(bw, blk + beg, len, e, f, bitw, FOR)) {
//                 printf("[DEVICE-ERROR] Failed to write ALP vector %d\n", v);
//                 break;
//             }
            
//             uint64_t after_pos = bw.get_pos();
//             if (blockId == 0 && v == 0) {
//                 printf("[DEVICE-EMIT] Vec0: wrote %llu bits (pos %llu->%llu)\n",
//                        after_pos - before_pos, before_pos, after_pos);
//             }
//         }
//     } else {
//         // ALPrd模式
//         for(int v = 0; v < numVec; v++) {
//             if (bw.remaining_bits() < 1000) break;
            
//             int beg = v * vectorSize;
//             int rem = n - beg;
//             int len = (vectorSize < rem ? vectorSize : rem);
            
//             uint64_t tmp[MAX_VEC];
//             assert(len <= MAX_VEC);
//             for(int i = 0; i < len; i++) {
//                 if constexpr (std::is_same_v<T,double>) 
//                     tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
//                 else 
//                     tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
//             }
            
//             ALPrdDict<T> D;
//             alprd_find_best<T>(tmp, len, D);
            
//             // 记录调试信息
//             if (enable_diag && dbg_modes) {
//                 uint64_t gid = vec_prefix[blockId] + (uint64_t)v;
//                 dbg_modes[gid] = 1;  // ALPrd
//                 if (dbg_e) dbg_e[gid] = 0xFF;
//                 if (dbg_f) dbg_f[gid] = 0xFF;
//             }
            
//             if (!alprd_vector_write_safe<T>(bw, tmp, len, D)) {
//                 printf("[DEVICE-ERROR] Failed to write ALPrd vector %d\n", v);
//                 break;
//             }
//         }
//     }
    
//     if (blockId == 0) {
//         printf("[DEVICE-EMIT] Block0 final position: %llu / %llu\n", 
//                bw.get_pos(), bw.max_bits);
//     }
// }

// // 解压kernel保持不变，但添加更多调试
// template<typename T>
// __global__ void kernel_decompress_debug(const uint8_t* bytes,
//                                         const uint64_t* blk_starts_bits,
//                                         const uint64_t* blk_bits,
//                                         const uint64_t* out_starts,
//                                         const int vectorSize,
//                                         T* out_data, 
//                                         int numBlocks) {
//     int blockId = blockIdx.x;
//     if (blockId >= numBlocks) return;

//     uint64_t bit_offset = blk_starts_bits[blockId];
//     BitReader br{bytes, bit_offset};
    
//     int numVec = (int)br.getN(8);
    
//     if (blockId == 0) {
//         printf("[DEVICE-DECOMP] Block0: bit_offset=%llu, numVec=%d\n", bit_offset, numVec);
//     }

//     if (numVec <= 0 || numVec > 10000) {
//         if (blockId == 0) {
//             printf("[DEVICE-ERROR] Invalid numVec: %d\n", numVec);
//         }
//         return;
//     }

//     uint64_t out_pos = out_starts[blockId];
    
//     for(int v = 0; v < numVec; v++) {
//         int useALP = br.get1();
        
//         if (blockId == 0 && v == 0) {
//             printf("[DEVICE-DECOMP] Vec0: useALP=%d\n", useALP);
//         }

//         if (useALP) {
//             uint8_t e = (uint8_t)br.getN(8);
//             uint8_t f = (uint8_t)br.getN(8);
//             short bitw = (short)br.getN(16);
//             long long FOR = (long long)br.getN(64);
//             int n = (int)br.getN(32);
            
//             if (blockId == 0 && v == 0) {
//                 printf("[DEVICE-DECOMP] Vec0 ALP: e=%d f=%d bitw=%d n=%d\n", e, f, bitw, n);
//             }

//             if (n <= 0 || n > MAX_VEC) return;

//             for(int k = 0; k < n; k++) {
//                 uint64_t enc = br.getN(bitw);
//                 long long I = FOR + (long long)enc;
//                 double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
//                 out_data[out_pos + k] = (T)dec;
//             }

//             int exc = (int)br.getN(16);
//             for(int t = 0; t < exc; t++) {
//                 uint64_t raw = std::is_same_v<T,double> ? br.getN(64) : br.getN(32);
//                 int pos = (int)br.getN(16);
                
//                 if (pos < n) {
//                     if constexpr (std::is_same_v<T,double>) {
//                         double val = *reinterpret_cast<double*>(&raw);
//                         out_data[out_pos + pos] = (T)val;
//                     } else {
//                         uint32_t rv = (uint32_t)raw;
//                         float val = *reinterpret_cast<float*>(&rv);
//                         out_data[out_pos + pos] = (T)val;
//                     }
//                 }
//             }
//             out_pos += n;
//         } else {
//             int n = (int)br.getN(32);
//             if (n <= 0 || n > MAX_VEC) return;

//             uint8_t rbw = (uint8_t)br.getN(8);
//             uint64_t right[MAX_VEC]; 
//             uint16_t leftIdx[MAX_VEC];
            
//             for(int k = 0; k < n; k++) {
//                 leftIdx[k] = (uint16_t)br.getN(DICT_BW);
//                 right[k] = br.getN(rbw);
//             }

//             uint8_t lbw = uint8_t(sizeof(T)*8 - rbw);
//             uint64_t dict[DICT_SZ];
//             for(int k = 0; k < DICT_SZ; k++) {
//                 dict[k] = br.getN(lbw);
//             }

//             int exc = (int)br.getN(16);
//             uint16_t exc_pos[MAX_VEC]; 
//             uint64_t exc_left[MAX_VEC];
//             for(int t = 0; t < exc; t++) {
//                 exc_left[t] = br.getN(lbw);
//                 exc_pos[t] = (uint16_t)br.getN(16);
//             }

//             for(int k = 0; k < n; k++) {
//                 uint64_t left = (leftIdx[k] < DICT_SZ) ? dict[leftIdx[k]] : 0ULL;
//                 uint64_t raw = (left << rbw) | right[k];
                
//                 if constexpr (std::is_same_v<T,double>) {
//                     double val = *reinterpret_cast<double*>(&raw);
//                     out_data[out_pos + k] = (T)val;
//                 } else {
//                     uint32_t r32 = (uint32_t)raw;
//                     float val = *reinterpret_cast<float*>(&r32);
//                     out_data[out_pos + k] = (T)val;
//                 }
//             }

//             for(int t = 0; t < exc; t++) {
//                 int p = exc_pos[t];
//                 uint64_t raw = (exc_left[t] << rbw) | right[p];
                
//                 if (p < n) {
//                     if constexpr (std::is_same_v<T,double>) {
//                         double val = *reinterpret_cast<double*>(&raw);
//                         out_data[out_pos + p] = (T)val;
//                     } else {
//                         uint32_t r32 = (uint32_t)raw;
//                         float val = *reinterpret_cast<float*>(&r32);
//                         out_data[out_pos + p] = (T)val;
//                     }
//                 }
//             }
//             out_pos += n;
//         }
//     }
// }

// // ===================== 修复的主要API实现 =====================
// template<typename T>
// static Compressed compress_impl_fixed(const T* h_data, size_t n, const Params& p) {
//     std::cout << "[DEBUG] 开始压缩，数据量: " << n << std::endl;
    
//     Compressed c;
//     if (n == 0) { c.vectorSize = p.vectorSize; return c; }

//     // 检查输入数据
//     std::cout << "[DEBUG] 输入数据前几个值: ";
//     for (size_t i = 0; i < std::min(size_t(5), n); i++) {
//         std::cout << h_data[i] << " ";
//     }
//     std::cout << std::endl;

//     const int V = p.vectorSize;
//     const int B = p.blockSize > 0 ? p.blockSize : int(n);
//     const int numBlocks = int((n + B - 1) / B);

//     std::cout << "[DEBUG] 块数: " << numBlocks << ", 向量大小: " << V << std::endl;

//     std::vector<uint64_t> h_starts(numBlocks), h_sizes(numBlocks);
//     size_t pos = 0;
//     for(int i = 0; i < numBlocks; i++) {
//         h_starts[i] = pos;
//         uint64_t sz = std::min<uint64_t>(B, n - pos);
//         h_sizes[i] = sz;
//         pos += sz;
//     }

//     bool diag = (std::getenv("ALP_GPU_DIAG") != nullptr);

//     // 上传数据
//     T* d_data = nullptr; 
//     CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
//     CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));

//     uint64_t *d_starts = nullptr, *d_sizes = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_starts, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_sizes, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_starts, h_starts.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

//     // 第一阶段：使用安全的测量kernel
//     uint64_t* d_bits = nullptr;  
//     uint8_t* d_mode = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_bits, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_mode, numBlocks * sizeof(uint8_t)));

//     dim3 grid1(numBlocks);
//     dim3 block1(THREADS_PER_BLOCK);
    
//     kernel_measure_with_sampling_safe<T><<<grid1, block1>>>(
//         d_data, d_starts, d_sizes, numBlocks, V, 
//         d_bits, d_mode
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());

//     std::vector<uint64_t> h_bits(numBlocks); 
//     std::vector<uint8_t> h_mode(numBlocks);
//     CUDA_CHECK(cudaMemcpy(h_bits.data(), d_bits, numBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(h_mode.data(), d_mode, numBlocks * sizeof(uint8_t), cudaMemcpyDeviceToHost));

//     // 调试输出
//     std::cout << "[DEBUG] 各块压缩位数: ";
//     for (int i = 0; i < std::min(5, numBlocks); i++) {
//         std::cout << h_bits[i] << "(" << (h_mode[i] ? "ALPrd" : "ALP") << ") ";
//     }
//     std::cout << std::endl;

//     // 计算偏移，增加额外安全缓冲
//     std::vector<uint64_t> h_off(numBlocks), h_budgets(numBlocks);
//     uint64_t acc = 0;
//     for (int i = 0; i < numBlocks; ++i) {
//         h_off[i] = acc;
//         uint64_t bits = h_bits[i];
        
//         // 关键修复：增加30%安全边距
//         uint64_t safe_bits = bits;// + bits * 1 / 10;  // +30%
//         uint64_t pad = (i + 1 < numBlocks) ? ((32 - (safe_bits & 31ULL)) & 31ULL) : 0ULL;
//         uint64_t padded_bits = safe_bits + pad;
        
//         h_budgets[i] = padded_bits;  // 记录每个块的位数预算
//         acc += padded_bits;
//     }
    
//     const uint64_t total_bits = acc;
//     const uint64_t total_bytes = (total_bits + 7) / 8;

//     std::cout << "[DEBUG] 原始总位数: " << std::accumulate(h_bits.begin(), h_bits.end(), 0ULL) << std::endl;
//     std::cout << "[DEBUG] 安全总位数: " << total_bits << std::endl;
//     std::cout << "[DEBUG] 总字节数: " << total_bytes << std::endl;

//     // 计算向量前缀
//     std::vector<uint64_t> h_vec_cnt(numBlocks), h_vec_prefix(numBlocks+1, 0);
//     uint64_t total_vecs = 0;
//     for (int i = 0; i < numBlocks; i++) {
//         uint64_t cnt = (h_sizes[i] + (uint64_t)V - 1) / (uint64_t)V;
//         h_vec_cnt[i] = cnt;
//         h_vec_prefix[i+1] = h_vec_prefix[i] + cnt;
//         total_vecs += cnt;
//     }

//     uint64_t* d_vec_prefix = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_vec_prefix, sizeof(uint64_t)*(numBlocks+1)));
//     CUDA_CHECK(cudaMemcpy(d_vec_prefix, h_vec_prefix.data(), sizeof(uint64_t)*(numBlocks+1), cudaMemcpyHostToDevice));

//     // 分配输出缓冲
//     uint8_t* d_out = nullptr; 
//     CUDA_CHECK(cudaMalloc(&d_out, total_bytes));
//     CUDA_CHECK(cudaMemset(d_out, 0, total_bytes));
    
//     uint64_t *d_off = nullptr, *d_budgets = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_off, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_budgets, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_off, h_off.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_budgets, h_budgets.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

//     // 调试缓冲
//     uint8_t *d_dbg_modes = nullptr, *d_dbg_e = nullptr, *d_dbg_f = nullptr;
//     if (diag && total_vecs > 0) {
//         CUDA_CHECK(cudaMalloc(&d_dbg_modes, sizeof(uint8_t) * total_vecs));
//         CUDA_CHECK(cudaMalloc(&d_dbg_e, sizeof(uint8_t) * total_vecs));
//         CUDA_CHECK(cudaMalloc(&d_dbg_f, sizeof(uint8_t) * total_vecs));
//         CUDA_CHECK(cudaMemset(d_dbg_modes, 0xFF, sizeof(uint8_t) * total_vecs));
//         CUDA_CHECK(cudaMemset(d_dbg_e, 0xFF, sizeof(uint8_t) * total_vecs));
//         CUDA_CHECK(cudaMemset(d_dbg_f, 0xFF, sizeof(uint8_t) * total_vecs));
//     }

//     // 第二阶段：使用安全的emit kernel
//     kernel_emit_with_sampling_safe<T><<<grid1, block1>>>(
//         d_data, d_starts, d_sizes, d_off, d_budgets, d_mode,
//         d_vec_prefix, d_dbg_modes, d_dbg_e, d_dbg_f,
//         diag ? 1 : 0, numBlocks, V, d_out
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // 拷回结果
//     c.data.resize(total_bytes);
//     CUDA_CHECK(cudaMemcpy(c.data.data(), d_out, total_bytes, cudaMemcpyDeviceToHost));
//     c.offsets = std::move(h_off);
//     c.bit_sizes = std::move(h_bits);  // 注意：这里用原始bits，不是budgets
//     c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
//     c.vectorSize = V;

//     // 调试：检查压缩数据
//     if (!c.data.empty()) {
//         std::cout << "[DEBUG] 压缩数据前10字节: ";
//         for (int i = 0; i < 10 && i < c.data.size(); i++) {
//             printf("%02X ", c.data[i]);
//         }
//         std::cout << std::endl;
//     }

//     // 诊断打印
//     if (diag && total_vecs > 0) {
//         std::vector<uint8_t> dbg_modes(total_vecs, 0xFF);
//         CUDA_CHECK(cudaMemcpy(dbg_modes.data(), d_dbg_modes, sizeof(uint8_t) * total_vecs, cudaMemcpyDeviceToHost));

//         uint64_t alp_cnt = 0, alprd_cnt = 0;
//         for (uint64_t i = 0; i < total_vecs; i++) {
//             if (dbg_modes[i] == 0) ++alp_cnt;
//             else if (dbg_modes[i] == 1) ++alprd_cnt;
//         }
//         std::cout << "[GPU-Diag] Vector-mode distribution: ALP=" << alp_cnt
//                   << ", ALPrd=" << alprd_cnt << ", totalVec=" << total_vecs << std::endl;
//     }

//     // 清理
//     if (d_dbg_modes) CUDA_CHECK(cudaFree(d_dbg_modes));
//     if (d_dbg_e) CUDA_CHECK(cudaFree(d_dbg_e));
//     if (d_dbg_f) CUDA_CHECK(cudaFree(d_dbg_f));
//     if (d_vec_prefix) CUDA_CHECK(cudaFree(d_vec_prefix));

//     CUDA_CHECK(cudaFree(d_budgets));
//     CUDA_CHECK(cudaFree(d_out)); 
//     CUDA_CHECK(cudaFree(d_off));
//     CUDA_CHECK(cudaFree(d_mode)); 
//     CUDA_CHECK(cudaFree(d_bits));
//     CUDA_CHECK(cudaFree(d_sizes)); 
//     CUDA_CHECK(cudaFree(d_starts));
//     CUDA_CHECK(cudaFree(d_data));
    
//     return c;
// }

// template<typename T>
// static void decompress_impl_fixed(const Compressed& c, T* h_out, size_t n, const Params& p) {
//     std::cout << "[DEBUG] 开始解压缩，数据量: " << n << std::endl;
    
//     if (n == 0) return;
//     const int numBlocks = (int)c.offsets.size();
//     assert((size_t)numBlocks == c.elem_counts.size());

//     std::cout << "[DEBUG] 解压块数: " << numBlocks << std::endl;
//     std::cout << "[DEBUG] 压缩数据大小: " << c.data.size() << " bytes" << std::endl;

//     if (c.data.empty()) {
//         std::cout << "[ERROR] 压缩数据为空！" << std::endl;
//         return;
//     }

//     // 检查压缩数据
//     std::cout << "[DEBUG] 压缩数据前10字节: ";
//     for (int i = 0; i < 10 && i < c.data.size(); i++) {
//         printf("%02X ", c.data[i]);
//     }
//     std::cout << std::endl;

//     // 初始化输出为特殊值以便检测
//     std::fill(h_out, h_out + n, T(-999.0));

//     uint8_t* d_bytes = nullptr; 
//     CUDA_CHECK(cudaMalloc(&d_bytes, c.data.size()));
//     CUDA_CHECK(cudaMemcpy(d_bytes, c.data.data(), c.data.size(), cudaMemcpyHostToDevice));

//     uint64_t *d_boff = nullptr, *d_bsiz = nullptr, *d_ost = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_boff, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_bsiz, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_boff, c.offsets.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_bsiz, c.bit_sizes.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

//     std::vector<uint64_t> h_outStarts(numBlocks);
//     uint64_t acc = 0; 
//     for(int i = 0; i < numBlocks; i++) { 
//         h_outStarts[i] = acc; 
//         acc += c.elem_counts[i]; 
//     }
    
//     if (acc != n) {
//         std::cout << "[ERROR] elem_counts总和(" << acc << ")不等于输出元素数(" << n << ")" << std::endl;
//         return;
//     }

//     CUDA_CHECK(cudaMalloc(&d_ost, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_ost, h_outStarts.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

//     T* d_out = nullptr; 
//     CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(T)));
//     CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(T)));

//     dim3 gs(numBlocks), bs(1);
//     kernel_decompress_debug<T><<<gs, bs>>>(
//         d_bytes, d_boff, d_bsiz, d_ost, p.vectorSize, d_out, numBlocks
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(T), cudaMemcpyDeviceToHost));

//     // 检查输出结果
//     bool all_negative999 = true;
//     bool all_zero = true;
//     for (size_t i = 0; i < std::min(size_t(10), n); i++) {
//         if (h_out[i] != T(-999.0)) all_negative999 = false;
//         if (h_out[i] != T(0.0)) all_zero = false;
//     }
    
//     if (all_negative999) {
//         std::cout << "[ERROR] 输出仍为初始值-999，解压失败！" << std::endl;
//     } else if (all_zero) {
//         std::cout << "[WARNING] 输出全为0！" << std::endl;
//     } else {
//         std::cout << "[DEBUG] 输出前几个值: ";
//         for (size_t i = 0; i < std::min(size_t(5), n); i++) {
//             std::cout << h_out[i] << " ";
//         }
//         std::cout << std::endl;
//     }

//     CUDA_CHECK(cudaFree(d_out));
//     CUDA_CHECK(cudaFree(d_ost));
//     CUDA_CHECK(cudaFree(d_bsiz));
//     CUDA_CHECK(cudaFree(d_boff));
//     CUDA_CHECK(cudaFree(d_bytes));
// }

// // 显式实例化API，使用修复的实现
// Compressed compress_double(const double* data, size_t n, const Params& p) { 
//     return compress_impl_fixed<double>(data, n, p); 
// }
// Compressed compress_float(const float* data, size_t n, const Params& p) { 
//     return compress_impl_fixed<float>(data, n, p); 
// }
// void decompress_double(const Compressed& c, double* out, size_t n, const Params& p) { 
//     decompress_impl_fixed<double>(c, out, n, p); 
// }
// void decompress_float(const Compressed& c, float* out, size_t n, const Params& p) { 
//     decompress_impl_fixed<float>(c, out, n, p); 
// }

// } // namespace alp_gpu
/*
 * ============================================
 *  ALP-GPU 压缩/解压（修复版）
 *  主要修复：
 *    1) 内存越界问题：增加安全边距和边界检查
 *    2) BitWriter/Reader同步问题：强制串行化
 *    3) CUDA错误检查：添加完整错误处理
 *    4) 大小估算问题：更保守的估算算法
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
        buf[byte] |= (uint8_t(1u) << off);
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

// 精确的ALP向量大小计算（基本无安全边距）
template<typename T>
__device__ inline uint64_t alp_vector_size_bits_safe(int n, uint8_t e, uint8_t f,
                                                     short bitw, int exc_cnt){
    int val_bits = std::is_same_v<T,double> ? 64 : 32;
    // ALP向量精确格式：1+8+8+16+64+32 + n*bitw + 16 + exc_cnt*(val_bits+16) = 145 + n*bitw + exc_cnt*(val_bits+16)
    uint64_t base = 145ULL + uint64_t(n)*bitw + uint64_t(exc_cnt)*(val_bits+16);
    
    // 极小安全边距，主要用于位对齐和实现细节差异
    uint64_t margin_candidate = base / 2000ULL;  // 0.05%
    uint64_t margin = (margin_candidate > 128ULL) ? margin_candidate : 128ULL;
    return base + margin;
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
    // ALPrd向量精确格式：1+32+8 + n*(DICT_BW+rightBW) + DICT_SZ*leftBW + 16 + exc_cnt*(leftBW+16) = 57 + ...
    uint64_t base = 57ULL + uint64_t(n)*(DICT_BW + D.rightBW) + DICT_SZ*D.leftBW + uint64_t(exc_cnt)*(D.leftBW+16);
    
    // 极小安全边距，主要用于位对齐
    uint64_t margin_candidate = base / 2000ULL; // 0.05%
    uint64_t margin = (margin_candidate > 128ULL) ? margin_candidate : 128ULL;
    return base + margin;
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

// ===================== 修复的Kernels =====================
static constexpr int THREADS_PER_BLOCK = 128;
static constexpr int MAX_VECS_PER_BLOCK = 256;

// 修复的测量kernel：使用保守的大小估算
template<typename T>
__global__ void kernel_measure_with_sampling_safe(
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
    
    EFCombination k_combinations[5];
    int k_actual = 0;
    CompressionMode mode;
    
    if(threadIdx.x == 0) {
        rowgroup_sample_and_find_k_combinations<T>(
            blk, n, vectorSize,
            k_combinations, k_actual, mode
        );
    }
    __syncthreads();
    
    __shared__ CompressionMode sh_mode;
    __shared__ EFCombination sh_k_combinations[5];
    __shared__ int sh_k_actual;
    
    if(threadIdx.x == 0) {
        sh_mode = mode;
        sh_k_actual = k_actual;
        for(int i = 0; i < k_actual; i++) {
            sh_k_combinations[i] = k_combinations[i];
        }
    }
    __syncthreads();
    
    __shared__ uint64_t sh_total_bits;
    if(threadIdx.x == 0) {
        sh_total_bits = 8;  // 行组头
    }
    __syncthreads();
    
    if(sh_mode == CompressionMode::ALP) {
        for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = (vectorSize < rem ? vectorSize : rem);
            
            uint8_t e, f; short bw; long long FOR; int exc;
            vector_choose_from_k_combinations<T>(
                blk + beg, len,
                sh_k_combinations, sh_k_actual,
                e, f, bw, FOR, exc
            );
            
            // 使用更保守的大小估算
            uint64_t bits = alp_vector_size_bits_safe<T>(len, e, f, bw, exc);
            atomicAdd((unsigned long long*)&sh_total_bits, bits);
        }
    } else {
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
            atomicAdd((unsigned long long*)&sh_total_bits, bits);
        }
    }
    
    __syncthreads();
    
    if(threadIdx.x == 0) {
        out_bits[blockId] = sh_total_bits;
        out_mode[blockId] = (sh_mode == CompressionMode::ALPrd) ? 1 : 0;
    }
}

// 并行版emit kernel：使用预计算偏移避免竞争
template<typename T>
__global__ void kernel_emit_with_sampling_safe(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const uint64_t* bit_offsets,
    const uint64_t* bit_budgets,
    const uint8_t* modes,
    const uint64_t* vec_prefix,
    uint8_t* dbg_modes, 
    uint8_t* dbg_e, 
    uint8_t* dbg_f,
    int enable_diag,
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
    
    if (blockId == 0 && threadIdx.x == 0) {
        printf("[DEVICE-EMIT] Block0: n=%d, numVec=%d, bit_offset=%llu, budget=%llu\n", 
               n, numVec, bit_offsets[blockId], bit_budgets[blockId]);
    }
    
    if (numVec <= 0) return;

    // 重新采样获取k个组合（线程0执行）
    EFCombination k_combinations[5] = {{0,0,0,0}};
    int k_actual = 1;
    CompressionMode mode_check;
    if(threadIdx.x == 0) {
        rowgroup_sample_and_find_k_combinations<T>(
            blk, n, vectorSize,
            k_combinations, k_actual, mode_check
        );
    }
    __syncthreads();
    
    // 广播到共享内存
    __shared__ EFCombination sh_k_combinations[5];
    __shared__ int sh_k_actual;
    
    if(threadIdx.x == 0) {
        sh_k_actual = k_actual;
        for(int i = 0; i < k_actual; i++) {
            sh_k_combinations[i] = k_combinations[i];
        }
    }
    __syncthreads();

    // 限制：如果向量太多，使用单线程串行处理
    if (numVec > MAX_VECS_PER_BLOCK) {
        if (threadIdx.x == 0) {
            SafeBitWriter bw(out_bytes, bit_offsets[blockId], bit_budgets[blockId]);
            if (!bw.putN((uint64_t)numVec, 8)) {
                printf("[DEVICE-ERROR] Failed to write numVec header\n");
                return;
            }
            
            if (mode == CompressionMode::ALP) {
                for(int v = 0; v < numVec; v++) {
                    int beg = v * vectorSize;
                    int rem = n - beg;
                    int len = (vectorSize < rem ? vectorSize : rem);
                    
                    if (bw.remaining_bits() < 1000) break;
                    
                    uint8_t e, f; short bitw; long long FOR; int exc;
                    vector_choose_from_k_combinations<T>(
                        blk + beg, len,
                        sh_k_combinations, sh_k_actual,
                        e, f, bitw, FOR, exc
                    );
                    
                    if (enable_diag && dbg_modes) {
                        uint64_t gid = vec_prefix[blockId] + (uint64_t)v;
                        dbg_modes[gid] = 0;  // ALP
                        if (dbg_e) dbg_e[gid] = e;
                        if (dbg_f) dbg_f[gid] = f;
                    }
                    
                    if (!alp_vector_write_safe<T>(bw, blk + beg, len, e, f, bitw, FOR)) {
                        printf("[DEVICE-ERROR] Failed to write ALP vector %d\n", v);
                        break;
                    }
                }
            } else {
                for(int v = 0; v < numVec; v++) {
                    if (bw.remaining_bits() < 1000) break;
                    
                    int beg = v * vectorSize;
                    int rem = n - beg;
                    int len = (vectorSize < rem ? vectorSize : rem);
                    
                    uint64_t tmp[MAX_VEC];
                    assert(len <= MAX_VEC);
                    for(int i = 0; i < len; i++) {
                        if constexpr (std::is_same_v<T,double>) 
                            tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                        else 
                            tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
                    }
                    
                    ALPrdDict<T> D;
                    alprd_find_best<T>(tmp, len, D);
                    
                    if (enable_diag && dbg_modes) {
                        uint64_t gid = vec_prefix[blockId] + (uint64_t)v;
                        dbg_modes[gid] = 1;  // ALPrd
                        if (dbg_e) dbg_e[gid] = 0xFF;
                        if (dbg_f) dbg_f[gid] = 0xFF;
                    }
                    
                    if (!alprd_vector_write_safe<T>(bw, tmp, len, D)) {
                        printf("[DEVICE-ERROR] Failed to write ALPrd vector %d\n", v);
                        break;
                    }
                }
            }
        }
        return;
    }

    // 并行处理（向量数较少时）
    __shared__ uint64_t sh_vec_bits[MAX_VECS_PER_BLOCK];
    __shared__ uint64_t sh_vec_offsets[MAX_VECS_PER_BLOCK + 1];
    __shared__ uint8_t sh_vec_e[MAX_VECS_PER_BLOCK];
    __shared__ uint8_t sh_vec_f[MAX_VECS_PER_BLOCK];
    __shared__ short sh_vec_bitw[MAX_VECS_PER_BLOCK];
    __shared__ long long sh_vec_FOR[MAX_VECS_PER_BLOCK];
    __shared__ ALPrdDict<T> sh_vec_dict[MAX_VECS_PER_BLOCK];  // ALPrd字典
    
    // 第一步：并行计算每个向量的参数和精确位数
    for (int v = threadIdx.x; v < numVec; v += blockDim.x) {
        int beg = v * vectorSize;
        int rem = n - beg;
        int len = (vectorSize < rem ? vectorSize : rem);
        
        if (mode == CompressionMode::ALP) {
            uint8_t e, f;
            short bitw;
            long long FOR;
            int exc;
            vector_choose_from_k_combinations<T>(
                blk + beg, len,
                sh_k_combinations, sh_k_actual,
                e, f, bitw, FOR, exc
            );
            
            // 计算精确位数（无安全边距）
            int val_bits = std::is_same_v<T,double> ? 64 : 32;
            uint64_t exact_bits = 1 + 8+8+16+64+32 + uint64_t(len)*bitw + 16 + uint64_t(exc)*(val_bits+16);
            
            sh_vec_bits[v] = exact_bits;
            sh_vec_e[v] = e;
            sh_vec_f[v] = f;
            sh_vec_bitw[v] = bitw;
            sh_vec_FOR[v] = FOR;
        } else {
            uint64_t tmp[MAX_VEC];
            assert(len <= MAX_VEC);
            for (int i = 0; i < len; ++i) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else 
                    tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            
            ALPrdDict<T> D;
            alprd_find_best<T>(tmp, len, D);
            
            // 统计异常数
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
            
            // 计算精确位数
            uint64_t exact_bits = 1 + 32 + 8 + uint64_t(len)*(DICT_BW + D.rightBW) 
                                + DICT_SZ*D.leftBW + 16 + uint64_t(exc)*(D.leftBW+16);
            
            sh_vec_bits[v] = exact_bits;
            sh_vec_dict[v] = D;  // 保存字典
        }
    }
    
    __syncthreads();
    
    // 第二步：计算前缀和（线程0执行串行扫描）
    if (threadIdx.x == 0) {
        sh_vec_offsets[0] = 8;  // 行组头占8位
        for (int v = 0; v < numVec; ++v) {
            sh_vec_offsets[v + 1] = sh_vec_offsets[v] + sh_vec_bits[v];
        }
        
        // 检查是否超出预算
        uint64_t total_needed = sh_vec_offsets[numVec];
        if (total_needed > bit_budgets[blockId]) {
            printf("[DEVICE-ERROR] Block%d exceeds budget: need %llu, have %llu\n", 
                   blockId, total_needed, bit_budgets[blockId]);
        }
    }
    __syncthreads();
    
    // 第三步：写行组头（线程0）
    if (threadIdx.x == 0) {
        SafeBitWriter bw(out_bytes, bit_offsets[blockId], bit_budgets[blockId]);
        if (!bw.putN((uint64_t)numVec, 8)) {
            printf("[DEVICE-ERROR] Failed to write numVec header\n");
        }
    }
    __syncthreads();
    
    // 第四步：并行写入每个向量的压缩数据
    for (int v = threadIdx.x; v < numVec; v += blockDim.x) {
        int beg = v * vectorSize;
        int rem = n - beg;
        int len = (vectorSize < rem ? vectorSize : rem);
        
        // 每个线程独立的SafeBitWriter，基于预计算的精确偏移
        uint64_t vec_bit_offset = bit_offsets[blockId] + sh_vec_offsets[v];
        uint64_t vec_bit_budget = sh_vec_bits[v] + 64; // 小缓冲防止边界错误
        SafeBitWriter vec_bw(out_bytes, vec_bit_offset, vec_bit_budget);
        
        if (mode == CompressionMode::ALP) {
            // 使用保存的(e,f)参数
            uint8_t e = sh_vec_e[v];
            uint8_t f = sh_vec_f[v];
            short bitw = sh_vec_bitw[v];
            long long FOR = sh_vec_FOR[v];
            
            // 记录调试信息
            if (enable_diag && dbg_modes) {
                uint64_t gid = vec_prefix[blockId] + (uint64_t)v;
                dbg_modes[gid] = 0;  // ALP
                if (dbg_e) dbg_e[gid] = e;
                if (dbg_f) dbg_f[gid] = f;
            }
            
            // 写入向量
            if (!alp_vector_write_safe<T>(vec_bw, blk + beg, len, e, f, bitw, FOR)) {
                printf("[DEVICE-ERROR] Failed to write ALP vector %d\n", v);
            }
            
        } else { // ALPrd
            uint64_t tmp[MAX_VEC];
            assert(len <= MAX_VEC);
            for (int i = 0; i < len; ++i) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else 
                    tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
            }
            
            // 记录调试信息
            if (enable_diag && dbg_modes) {
                uint64_t gid = vec_prefix[blockId] + (uint64_t)v;
                dbg_modes[gid] = 1;  // ALPrd
                if (dbg_e) dbg_e[gid] = 0xFF;
                if (dbg_f) dbg_f[gid] = 0xFF;
            }
            
            // 使用保存的字典
            if (!alprd_vector_write_safe<T>(vec_bw, tmp, len, sh_vec_dict[v])) {
                printf("[DEVICE-ERROR] Failed to write ALPrd vector %d\n", v);
            }
        }
        
        // 调试：检查第一个向量
        if (blockId == 0 && v == 0) {
            printf("[DEVICE-EMIT] Vec0: wrote %llu bits at offset %llu\n",
                   sh_vec_bits[v], vec_bit_offset);
        }
    }
}

// 解压kernel保持不变，但添加更多调试
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
    
    if (blockId == 0) {
        printf("[DEVICE-DECOMP] Block0: bit_offset=%llu, numVec=%d\n", bit_offset, numVec);
    }

    if (numVec <= 0 || numVec > 10000) {
        if (blockId == 0) {
            printf("[DEVICE-ERROR] Invalid numVec: %d\n", numVec);
        }
        return;
    }

    uint64_t out_pos = out_starts[blockId];
    
    for(int v = 0; v < numVec; v++) {
        int useALP = br.get1();
        
        if (blockId == 0 && v == 0) {
            printf("[DEVICE-DECOMP] Vec0: useALP=%d\n", useALP);
        }

        if (useALP) {
            uint8_t e = (uint8_t)br.getN(8);
            uint8_t f = (uint8_t)br.getN(8);
            short bitw = (short)br.getN(16);
            long long FOR = (long long)br.getN(64);
            int n = (int)br.getN(32);
            
            if (blockId == 0 && v == 0) {
                printf("[DEVICE-DECOMP] Vec0 ALP: e=%d f=%d bitw=%d n=%d\n", e, f, bitw, n);
            }

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

// ===================== 修复的主要API实现 =====================
template<typename T>
static Compressed compress_impl_fixed(const T* h_data, size_t n, const Params& p) {
    std::cout << "[DEBUG] 开始压缩，数据量: " << n << std::endl;
    
    Compressed c;
    if (n == 0) { c.vectorSize = p.vectorSize; return c; }

    // 检查输入数据
    std::cout << "[DEBUG] 输入数据前几个值: ";
    for (size_t i = 0; i < std::min(size_t(5), n); i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    const int V = p.vectorSize;
    const int B = p.blockSize > 0 ? p.blockSize : int(n);
    const int numBlocks = int((n + B - 1) / B);

    std::cout << "[DEBUG] 块数: " << numBlocks << ", 向量大小: " << V << std::endl;

    std::vector<uint64_t> h_starts(numBlocks), h_sizes(numBlocks);
    size_t pos = 0;
    for(int i = 0; i < numBlocks; i++) {
        h_starts[i] = pos;
        uint64_t sz = std::min<uint64_t>(B, n - pos);
        h_sizes[i] = sz;
        pos += sz;
    }

    bool diag = (std::getenv("ALP_GPU_DIAG") != nullptr);

    // 上传数据
    T* d_data = nullptr; 
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));

    uint64_t *d_starts = nullptr, *d_sizes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_starts, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sizes, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_starts, h_starts.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // 第一阶段：使用安全的测量kernel
    uint64_t* d_bits = nullptr;  
    uint8_t* d_mode = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_mode, numBlocks * sizeof(uint8_t)));

    dim3 grid1(numBlocks);
    dim3 block1(THREADS_PER_BLOCK);
    
    kernel_measure_with_sampling_safe<T><<<grid1, block1>>>(
        d_data, d_starts, d_sizes, numBlocks, V, 
        d_bits, d_mode
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> h_bits(numBlocks); 
    std::vector<uint8_t> h_mode(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_bits.data(), d_bits, numBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mode.data(), d_mode, numBlocks * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // 调试输出
    std::cout << "[DEBUG] 各块压缩位数: ";
    for (int i = 0; i < std::min(5, numBlocks); i++) {
        std::cout << h_bits[i] << "(" << (h_mode[i] ? "ALPrd" : "ALP") << ") ";
    }
    std::cout << std::endl;

    // 计算偏移，使用精确大小无需额外安全缓冲
    std::vector<uint64_t> h_off(numBlocks), h_budgets(numBlocks);
    uint64_t acc = 0;
    for (int i = 0; i < numBlocks; ++i) {
        h_off[i] = acc;
        uint64_t bits = h_bits[i];  // 已经包含极小的安全边距
        
        // 只进行32位边界对齐，无额外缓冲
        uint64_t pad = (i + 1 < numBlocks) ? ((32 - (bits & 31ULL)) & 31ULL) : 0ULL;
        uint64_t padded_bits = bits + pad;
        
        h_budgets[i] = padded_bits;  // 记录每个块的位数预算
        acc += padded_bits;
    }
    
    const uint64_t total_bits = acc;
    const uint64_t total_bytes = (total_bits + 7) / 8;

    std::cout << "[DEBUG] 原始总位数: " << std::accumulate(h_bits.begin(), h_bits.end(), 0ULL) << std::endl;
    std::cout << "[DEBUG] 安全总位数: " << total_bits << std::endl;
    std::cout << "[DEBUG] 总字节数: " << total_bytes << std::endl;

    // 计算向量前缀
    std::vector<uint64_t> h_vec_cnt(numBlocks), h_vec_prefix(numBlocks+1, 0);
    uint64_t total_vecs = 0;
    for (int i = 0; i < numBlocks; i++) {
        uint64_t cnt = (h_sizes[i] + (uint64_t)V - 1) / (uint64_t)V;
        h_vec_cnt[i] = cnt;
        h_vec_prefix[i+1] = h_vec_prefix[i] + cnt;
        total_vecs += cnt;
    }

    uint64_t* d_vec_prefix = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vec_prefix, sizeof(uint64_t)*(numBlocks+1)));
    CUDA_CHECK(cudaMemcpy(d_vec_prefix, h_vec_prefix.data(), sizeof(uint64_t)*(numBlocks+1), cudaMemcpyHostToDevice));

    // 分配输出缓冲
    uint8_t* d_out = nullptr; 
    CUDA_CHECK(cudaMalloc(&d_out, total_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, total_bytes));
    
    uint64_t *d_off = nullptr, *d_budgets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_off, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_budgets, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_off, h_off.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_budgets, h_budgets.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // 调试缓冲
    uint8_t *d_dbg_modes = nullptr, *d_dbg_e = nullptr, *d_dbg_f = nullptr;
    if (diag && total_vecs > 0) {
        CUDA_CHECK(cudaMalloc(&d_dbg_modes, sizeof(uint8_t) * total_vecs));
        CUDA_CHECK(cudaMalloc(&d_dbg_e, sizeof(uint8_t) * total_vecs));
        CUDA_CHECK(cudaMalloc(&d_dbg_f, sizeof(uint8_t) * total_vecs));
        CUDA_CHECK(cudaMemset(d_dbg_modes, 0xFF, sizeof(uint8_t) * total_vecs));
        CUDA_CHECK(cudaMemset(d_dbg_e, 0xFF, sizeof(uint8_t) * total_vecs));
        CUDA_CHECK(cudaMemset(d_dbg_f, 0xFF, sizeof(uint8_t) * total_vecs));
    }

    // 第二阶段：使用安全的emit kernel
    kernel_emit_with_sampling_safe<T><<<grid1, block1>>>(
        d_data, d_starts, d_sizes, d_off, d_budgets, d_mode,
        d_vec_prefix, d_dbg_modes, d_dbg_e, d_dbg_f,
        diag ? 1 : 0, numBlocks, V, d_out
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷回结果
    c.data.resize(total_bytes);
    CUDA_CHECK(cudaMemcpy(c.data.data(), d_out, total_bytes, cudaMemcpyDeviceToHost));
    c.offsets = std::move(h_off);
    c.bit_sizes = std::move(h_bits);  // 注意：这里用原始bits，不是budgets
    c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
    c.vectorSize = V;

    // 调试：检查压缩数据
    if (!c.data.empty()) {
        std::cout << "[DEBUG] 压缩数据前10字节: ";
        for (int i = 0; i < 10 && i < c.data.size(); i++) {
            printf("%02X ", c.data[i]);
        }
        std::cout << std::endl;
    }

    // 诊断打印
    if (diag && total_vecs > 0) {
        std::vector<uint8_t> dbg_modes(total_vecs, 0xFF);
        CUDA_CHECK(cudaMemcpy(dbg_modes.data(), d_dbg_modes, sizeof(uint8_t) * total_vecs, cudaMemcpyDeviceToHost));

        uint64_t alp_cnt = 0, alprd_cnt = 0;
        for (uint64_t i = 0; i < total_vecs; i++) {
            if (dbg_modes[i] == 0) ++alp_cnt;
            else if (dbg_modes[i] == 1) ++alprd_cnt;
        }
        std::cout << "[GPU-Diag] Vector-mode distribution: ALP=" << alp_cnt
                  << ", ALPrd=" << alprd_cnt << ", totalVec=" << total_vecs << std::endl;
    }

    // 清理
    if (d_dbg_modes) CUDA_CHECK(cudaFree(d_dbg_modes));
    if (d_dbg_e) CUDA_CHECK(cudaFree(d_dbg_e));
    if (d_dbg_f) CUDA_CHECK(cudaFree(d_dbg_f));
    if (d_vec_prefix) CUDA_CHECK(cudaFree(d_vec_prefix));

    CUDA_CHECK(cudaFree(d_budgets));
    CUDA_CHECK(cudaFree(d_out)); 
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_mode)); 
    CUDA_CHECK(cudaFree(d_bits));
    CUDA_CHECK(cudaFree(d_sizes)); 
    CUDA_CHECK(cudaFree(d_starts));
    CUDA_CHECK(cudaFree(d_data));
    
    return c;
}

template<typename T>
static void decompress_impl_fixed(const Compressed& c, T* h_out, size_t n, const Params& p) {
    std::cout << "[DEBUG] 开始解压缩，数据量: " << n << std::endl;
    
    if (n == 0) return;
    const int numBlocks = (int)c.offsets.size();
    assert((size_t)numBlocks == c.elem_counts.size());

    std::cout << "[DEBUG] 解压块数: " << numBlocks << std::endl;
    std::cout << "[DEBUG] 压缩数据大小: " << c.data.size() << " bytes" << std::endl;

    if (c.data.empty()) {
        std::cout << "[ERROR] 压缩数据为空！" << std::endl;
        return;
    }

    // 检查压缩数据
    std::cout << "[DEBUG] 压缩数据前10字节: ";
    for (int i = 0; i < 10 && i < c.data.size(); i++) {
        printf("%02X ", c.data[i]);
    }
    std::cout << std::endl;

    // 初始化输出为特殊值以便检测
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
    } else {
        std::cout << "[DEBUG] 输出前几个值: ";
        for (size_t i = 0; i < std::min(size_t(5), n); i++) {
            std::cout << h_out[i] << " ";
        }
        std::cout << std::endl;
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ost));
    CUDA_CHECK(cudaFree(d_bsiz));
    CUDA_CHECK(cudaFree(d_boff));
    CUDA_CHECK(cudaFree(d_bytes));
}

// 显式实例化API，使用修复的实现
Compressed compress_double(const double* data, size_t n, const Params& p) { 
    return compress_impl_fixed<double>(data, n, p); 
}
Compressed compress_float(const float* data, size_t n, const Params& p) { 
    return compress_impl_fixed<float>(data, n, p); 
}
void decompress_double(const Compressed& c, double* out, size_t n, const Params& p) { 
    decompress_impl_fixed<double>(c, out, n, p); 
}
void decompress_float(const Compressed& c, float* out, size_t n, const Params& p) { 
    decompress_impl_fixed<float>(c, out, n, p); 
}

} // namespace alp_gpu