// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"
#include "attention_universal.h"
#include "reduce.h"
#include "src/turbomind/kernels/core/thread_map.h"
#include "utils.h"

namespace turbomind {

template<class Kernel>
bool invokeDecoding(const typename Kernel::ParamType& params)
{
    static const size_t kSmemSize = sizeof(typename Kernel::SharedStorage);

    if constexpr (1) {
        [[maybe_unused]] static const int _ = [&] {
            // std::cout << __PRETTY_FUNCTION__ << std::endl;
            // std::cout << "GmemMap:\n";
            // Print(typename Kernel::Impl::ThreadMapKV{});
            // std::cout << "\nDynamic smem size: " << kSmemSize << "\n";
            return 0;
        }();
    }

    const int max_cp_k_len    = cdiv(params.max_k_len, (int)params.cp_size);
    const int tile_count      = cdiv(std::min(max_cp_k_len, params.window_size), Kernel::CTA_S);
    const int max_split_count = std::min(params.max_split_k, tile_count);

    using CtaMap = typename Kernel::CtaMap;

    dim3 block(Kernel::kWarpCount * WARP_SIZE);

    auto kernel_func = &attention_kernel<Kernel>;

    // NOTE:
    // - sm75: MaxSharedMemoryPerBlock=48KB, Optin=64KB (you measured this already)
    // - If (dynamic + static) shared memory > optin limit, kernel launch may fail with "invalid argument"
    thread_local const int2 caps = [&] {
        // Query kernel static shared memory usage (for debugging)
        cudaFuncAttributes attr{};
        auto aerr = cudaFuncGetAttributes(&attr, kernel_func);
        if (aerr != cudaSuccess) {
            std::cerr << "[Decoding] cudaFuncGetAttributes failed: "
                      << cudaGetErrorName(aerr) << " " << cudaGetErrorString(aerr) << "\n";
        }

        // Query device shared memory limits (for debugging)
        int device_id{};
        cudaGetDevice(&device_id);

        int smem_max = 0, smem_optin = 0, sm_count = 0;
        cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);

        std::cerr << "[Decoding] dev=" << device_id
                  << " kSmemSize(dynamic)=" << kSmemSize
                  << " staticShared=" << attr.sharedSizeBytes
                  << " totalShared=" << (kSmemSize + (size_t)attr.sharedSizeBytes)
                  << " devMax=" << smem_max
                  << " devOptin=" << smem_optin
                  << " block.x=" << block.x
                  << "\n";

        // Clear any previous error
        (void)cudaGetLastError();

        // Opt-in dynamic shared memory size for this kernel
        auto err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmemSize);
        if (err != cudaSuccess) {
            std::cerr << "[Decoding] cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: "
                      << cudaGetErrorName(err) << " " << cudaGetErrorString(err)
                      << " kSmemSize=" << kSmemSize
                      << " staticShared=" << attr.sharedSizeBytes
                      << " totalShared=" << (kSmemSize + (size_t)attr.sharedSizeBytes)
                      << " devOptin=" << smem_optin
                      << "\n";
            std::abort();
        }

        int max_active_ctas{};
        auto occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_ctas, kernel_func, block.x, (int)kSmemSize);
        if (occ_err != cudaSuccess) {
            std::cerr << "[Decoding] cudaOccupancyMaxActiveBlocksPerMultiprocessor failed: "
                      << cudaGetErrorName(occ_err) << " " << cudaGetErrorString(occ_err)
                      << " block.x=" << block.x << " smem=" << kSmemSize << "\n";
            std::abort();
        }

        return int2{sm_count, max_active_ctas};
    }();

    const int q_group_size   = params.num_heads / params.num_kv_heads;
    const int q_head_per_cta = std::min(q_group_size, Kernel::CTA_H);

    // cta needed to process one query group
    const int cta_per_q_group = (q_group_size + q_head_per_cta - 1) / q_head_per_cta;

    dim3 grid = CtaMap::get_grid_shape(params.num_kv_heads, params.batch_size, 1, cta_per_q_group);

    const int grid_size = grid.x * grid.y * grid.z;
    const int split_cnt = GetSplitCount(max_split_count, grid_size, caps.y, caps.x, 4);

    grid = CtaMap::get_grid_shape(params.num_kv_heads, params.batch_size, split_cnt, cta_per_q_group);

    auto cache_iter_factory = CreateCacheIterFactory<typename Kernel::CacheIteratorFactory>::apply(params);

    // Clear previous error
    (void)cudaGetLastError();

    kernel_func<<<grid, block, kSmemSize, params.stream>>>(
        params, cache_iter_factory, CtaMap{}, q_group_size, q_head_per_cta, cta_per_q_group);

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        // Print as much as possible to pinpoint why launch fails
        cudaFuncAttributes attr{};
        (void)cudaFuncGetAttributes(&attr, kernel_func);
        int device_id{};
        cudaGetDevice(&device_id);
        int smem_optin = 0;
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);

        std::cerr << "[Decoding] launch failed: "
                  << cudaGetErrorName(err) << " " << cudaGetErrorString(err)
                  << " dev=" << device_id
                  << " grid=(" << grid.x << "," << grid.y << "," << grid.z << ")"
                  << " block=(" << block.x << "," << block.y << "," << block.z << ")"
                  << " kSmemSize(dynamic)=" << kSmemSize
                  << " staticShared=" << attr.sharedSizeBytes
                  << " totalShared=" << (kSmemSize + (size_t)attr.sharedSizeBytes)
                  << " devOptin=" << smem_optin
                  << " num_kv_heads=" << params.num_kv_heads
                  << " num_heads=" << params.num_heads
                  << " batch=" << params.batch_size
                  << " token_num=" << params.token_num
                  << " cp_size=" << params.cp_size
                  << " cp_rank=" << params.cp_rank
                  << "\n";

        std::abort();
    }

    if (params.cp_fn) {
        params.cp_fn(params.cp_fn_ctx);
    }

    if (split_cnt > 1 || params.cp_size > 1) {
        attention::invokeReduceV3<Kernel::kHeadDim>(params.out,
                                                    params.partial_ML,
                                                    params.partial_O,
                                                    split_cnt > 1 ? params.split_cnt : nullptr,
                                                    params.max_split_k,
                                                    split_cnt,
                                                    params.cp_size,
                                                    params.cp_rank,
                                                    params.token_num,
                                                    params.num_heads,
                                                    params.inv_sqrt_dh,
                                                    params.stream);
    }

    return true;
}

}  // namespace turbomind