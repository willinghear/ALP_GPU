#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>
#include <cuda_runtime.h> 
// ================================
// ALP-GPU 压缩/解压公共接口（头文件）
// 说明：
//   - 本头文件仅暴露必要的类型与 API，避免把实现细节泄露到 ODR（链接）边界；
//   - 与 CPU 版本保持一致的概念：向量(vector)、行组(row group)、两种模式(ALP/ALPrd)；
//   - CUDA 实现将原始数据划分为多个“数据块”，每个线程处理一个块，实现 GPU 并行加速；
//   - 压缩元信息（每块的位流偏移、比特数、元素个数）随压缩字节流一起返回。
// ================================

namespace alp_gpu {

// ============ 添加采样相关配置和结构 ============
namespace config {
    static constexpr int VECTOR_SIZE = 1024;
    static constexpr int ROWGROUP_SIZE = 100 * VECTOR_SIZE;
    static constexpr int ROWGROUP_VECTOR_SAMPLES = 8;
    static constexpr int SAMPLES_PER_VECTOR = 32;
    static constexpr int MAX_K_COMBINATIONS = 5;
    static constexpr int SAMPLING_EARLY_EXIT_THRESHOLD = 2;
}

// 阈值定义
template<typename T> struct Constants {
    static constexpr int EXCEPTION_SIZE = sizeof(T) == 8 ? 64 : 32;
    static constexpr int EXCEPTION_POSITION_SIZE = 16;
    static constexpr int MAX_EXPONENT = 18;
    static constexpr size_t RD_SIZE_THRESHOLD_LIMIT = 
        sizeof(T) == 8 ? (48 * config::SAMPLES_PER_VECTOR) : (22 * config::SAMPLES_PER_VECTOR);
};

// (e,f)组合及其统计信息
struct EFCombination {
    uint8_t e, f;
    int count;       // 出现次数
    double score;    // 压缩评分
};




// 压缩模式（与 CPU 版一致）
enum class CompressionMode : std::uint8_t {
    ALP  = 0,   // 适合“十进制可精确表示”的数据（全局量化成功率高）
    ALPrd = 1   // 适合“真正的 IEEE 浮点分布”的数据（切割+字典）
};

// 运行参数
struct Params {
    int  vectorSize      = 1024;     // 每个向量的长度（与 CPU 常量一致）
    int  blockSize       = 102400;   // 每个数据块的元素数（一个线程处理一个块）
    int  threadsPerBlock = 256;      // 每个 CUDA block 启动的线程数
    bool use_alprd_cutting = true;   // 允许在 ALPrd 中做 bit 切割与字典
    bool prefer_alprd      = false;  // 数据不明确时优先选 ALPrd
    bool debug             = true;  // 打印调试信息（会稍微减速）
};

// 压缩结果（位流+每块元信息），用于解压
struct Compressed {
    std::vector<std::uint8_t>  data;        // 压缩后的连续字节流
    std::vector<std::uint64_t> offsets;     // 每块位流起始 bit 偏移
    std::vector<std::uint64_t> bit_sizes;   // 每块占用的 bit 数
    std::vector<std::uint32_t> elem_counts; // 每块包含元素个数
    int                        vectorSize = 1024;

    inline bool   empty() const { return data.empty(); }
    inline size_t bytes() const { return data.size(); }
    inline size_t blocks() const { return offsets.size(); }
    inline size_t total_elems() const {
        size_t s = 0; for (auto v : elem_counts) s += v; return s;
    }
};

// 压缩 / 解压 API（与 C 版保持语义一致；CUDA 内部采用“一个线程处理一个数据块”的并行策略）
Compressed compress_double(const double* data, size_t n, const Params& p);
Compressed compress_float (const float*  data, size_t n, const Params& p);

void decompress_double(const Compressed& c, double* out, size_t n, const Params& p);
void decompress_float (const Compressed& c, float*  out, size_t n, const Params& p);


// 设备端压缩结果（压缩数据保存在GPU上）
struct CompressedDevice {
    uint8_t* d_data;                // 设备端压缩数据指针
    size_t data_size;               // 压缩数据大小（字节）
    std::vector<uint64_t> offsets;  // 每块位流起始bit偏移（保持在主机端）
    std::vector<uint64_t> bit_sizes;// 每块占用的bit数
    std::vector<uint32_t> elem_counts; // 每块包含元素个数
    int vectorSize;
    
    // 析构函数自动释放设备内存
    ~CompressedDevice() {
        if(d_data) cudaFree(d_data);
    }
    
    // 移动构造函数
    CompressedDevice(CompressedDevice&& other) noexcept 
        : d_data(other.d_data), data_size(other.data_size),
          offsets(std::move(other.offsets)), bit_sizes(std::move(other.bit_sizes)),
          elem_counts(std::move(other.elem_counts)), vectorSize(other.vectorSize) {
        other.d_data = nullptr;
    }
    
    // 默认构造函数
    CompressedDevice() : d_data(nullptr), data_size(0), vectorSize(1024) {}
};

// 新的设备端API
CompressedDevice compress_double_device(const double* d_data, size_t n, const Params& p, cudaStream_t stream = 0);
CompressedDevice compress_float_device(const float* d_data, size_t n, const Params& p, cudaStream_t stream = 0);

void decompress_double_device(const CompressedDevice& c, double* d_out, size_t n, const Params& p, cudaStream_t stream = 0);
void decompress_float_device(const CompressedDevice& c, float* d_out, size_t n, const Params& p, cudaStream_t stream = 0);

// 辅助函数：将设备端压缩结果拷贝到主机
Compressed device_to_host(const CompressedDevice& cd);
CompressedDevice host_to_device(const Compressed& c);

} // namespace alp_gpu
