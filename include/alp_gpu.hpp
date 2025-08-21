#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>

namespace alp_gpu {

// 压缩模式（与 CPU 版一致）
enum class CompressionMode : uint8_t { ALP = 0, ALPrd = 1 };

// 运行参数
struct Params {
    int  vectorSize = 1000;   // 每个向量的长度（与 CPU 常量一致）
    int  blockSize  = 100000; // 每个数据块的元素数（一个线程处理一个块）
    bool debug     = true;   // ← 开启后收集每向量的模式与(e,f)
};

// 压缩输出（合并后的位流 + per-block 偏移/长度/元素数，便于解压）
struct Compressed {
    std::vector<uint8_t>  data;        // 合并后的压缩位流（字节对齐）
    std::vector<uint64_t> offsets;     // 每块 bit 起始位置（bit）
    std::vector<uint64_t> bit_sizes;   // 每块占用 bit 数（bit）
    std::vector<uint64_t> elem_counts; // 每块原始元素数（个）——用于解压还原输出定位
    int  vectorSize = 1000;

    // ↓↓↓ 以下仅在 Params.debug=true 时填充（否则为空）
    std::vector<uint8_t>  dbg_modes;   // 每向量：0=ALP, 1=ALPrd
    std::vector<uint8_t>  dbg_e;       // 仅 ALP 有效；ALPrd 填 0xFF
    std::vector<uint8_t>  dbg_f;       // 仅 ALP 有效；ALPrd 填 0xFF
};

// 核心 API（double / float 都支持）
Compressed compress_double(const double* data, size_t n, const Params& p);
Compressed compress_float (const float*  data, size_t n, const Params& p);

void decompress_double(const Compressed& c, double* out, size_t n, const Params& p);
void decompress_float (const Compressed& c, float*  out, size_t n, const Params& p);

} // namespace alp_gpu
