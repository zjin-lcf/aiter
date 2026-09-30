// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_bf16gemm_configs.hpp"
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <hip/hip_runtime.h>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;
    p2 _p0;
    void* ptr_C;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    void* ptr_B;
    p2 _p3;
    float alpha;
    p3 _p4;
    float beta;
    p3 _p5;
    unsigned int stride_D0;
    p3 _p6;
    unsigned int stride_D1;
    p3 _p7;
    unsigned int stride_C0;
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;
    p3 _p10;
    unsigned int stride_A1;
    p3 _p11;
    unsigned int stride_B0;
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    unsigned int splitk;
    p3 _p17;
    unsigned int is_out_b16;
    p3 _p18;
    void* ptr_Bias;
    p2 _p19;
    unsigned int add_bias;
    p3 _p20;
    void* ptr_semaphore;
    p2 _p21;
};

std::tuple<std::string, int>
get_heuristic_kernel(int M,
                     int N,
                     int K,
                     CFG* cfgs,
                     std::string arch_id,
                     bool bpreshuffle,
                     int add_bias,
                     int splitk           = -1,
                     const char* kernelName = nullptr)
{
    AITER_CHECK(K % 64 == 0, __func__, " Kdim must be divisible by 64 !"); // load min size is 128b
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu = dev_prop.multiProcessorCount;

    uint32_t empty_cu      = num_cu;
    uint32_t pure_tg_num   = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0;
    int oob                = M;

    std::string selectedKernelName = "";
    int selectedsplitK             = 1;

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(kernelName && el.first != (arch_id + kernelName))
            continue;
        // check specified kernel name
        if(kernelName)
        {
            AITER_CHECK(N % cfg.tileN == 0 && cfg.bPreshuffle == (bpreshuffle ? 1 : 0) &&
                            (add_bias == 0 || cfg.bias == 1),
                        __func__,
                        " The specified kernel name ",
                        el.first,
                        " cannot support the input shape (N=",
                        N,
                        ", tileN=",
                        cfg.tileN,
                        ") or bias/preshuffle setting (preshuffle=",
                        bpreshuffle,
                        ", bias=",
                        add_bias,
                        ").");
            selectedKernelName = el.first;
            if(splitk >= 0)
            {
                selectedsplitK = splitk;
                selectedsplitK = std::min({selectedsplitK, 16, static_cast<int>(K / cfg.subK)});
                AITER_CHECK((selectedsplitK > 1 && cfg.splitK == 1) ||
                                (selectedsplitK <= 1 && cfg.splitK == 0),
                            __func__,
                            " The specified splitK ",
                            selectedsplitK,
                            " cannot be supported by the specified kernel or Kdim",
                            el.first,
                            ".");
                break;
            }
        }
        // auto select splitk or kernel
        if(N % cfg.tileN == 0 && cfg.bPreshuffle == (bpreshuffle ? 1 : 0) &&
           (add_bias == 0 || cfg.bias == 1))
        {
            // 1. select splitk
            int split_K = 1;
            pure_tg_num = ((M + cfg.tileM - 1) / cfg.tileM) * (N / cfg.tileN);
            if(cfg.splitK == 1 && K / cfg.subK >= 2) // kernel and Kdim support splitk
            {
                AITER_CHECK(cfg.subK > 0,
                    __func__,
                    " cfg.subK must be greater than 0 to avoid division by zero.");
                int max_splitk = std::min(std::min(static_cast<int>(num_cu / pure_tg_num), 16),
                                          static_cast<int>(K / cfg.subK));
                split_K =
                    std::max(2, max_splitk); // if kernel support splitk, set splitk to 2 at least.
            }
            // 2. better or not
            uint32_t tg_num      = pure_tg_num * split_K;
            uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
            float local_compute2mem_effi =
                static_cast<float>(cfg.tileM * cfg.tileN) / (cfg.tileM + cfg.tileN);
            bool is_earlier_round        = (local_round < round);
            bool is_same_round           = (local_round == round);
            bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
            bool has_same_empty_cu       = (empty_cu == (local_round * num_cu - tg_num));
            bool has_better_efficiency   = (local_compute2mem_effi > compute2mem_effi);
            bool less_oob = (M % cfg.tileM == 0) ? (oob > 0) : (cfg.tileM - M % cfg.tileM < oob);
            bool has_same_oob = (cfg.tileM - (M % cfg.tileM)) == oob;

            if(is_earlier_round || (is_same_round && (has_sufficient_empty_cu || less_oob)) ||
               (is_same_round && has_same_empty_cu && has_same_oob && has_better_efficiency))
            {
                round              = local_round;
                empty_cu           = local_round * num_cu - tg_num;
                compute2mem_effi   = local_compute2mem_effi;
                oob                = (M % cfg.tileM == 0) ? 0 : cfg.tileM - (M % cfg.tileM);
                selectedKernelName = el.first;
                selectedsplitK     = split_K;
            }
        }
    }
    AITER_CHECK(
        selectedKernelName != "", __func__, " not find kernel for bf16gemm~ " + selectedKernelName);
    return std::make_tuple(selectedKernelName, selectedsplitK);
}

AiterAsmKernel* get_or_load_kernel(const std::string& selectedKernelName,
                                   CFG* config_map,
                                   unsigned int& SUBM,
                                   unsigned int& SUBN)
{
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    auto it_kl = config_map->find(selectedKernelName);
    AITER_CHECK(it_kl != config_map->end(), __func__, " not find kernel~ " + selectedKernelName);

    const auto& cfg     = it_kl->second;
    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    SUBM                = cfg.tileM;
    SUBN                = cfg.tileN;

    return &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
}

namespace
{
// One packed staging tile per device, stream, element size and K. Never freed:
// a captured graph holds the pointer. A larger tile allocates another buffer.
struct APadBuf
{
    int device;
    uintptr_t stream;
    size_t elem;
    int k;
    int rows;
    int live; // [live, rows) is zero. [0, live) may hold a previous A.
    bool captured;
    void* ptr;
};

std::mutex apad_mu;
std::vector<APadBuf> apad_bufs;

// Caller holds apad_mu. `fresh` means the new tile is already being zeroed.
APadBuf& apad_buffer(int device,
                     hipStream_t stream,
                     size_t elem,
                     int k,
                     int rows,
                     size_t row_bytes,
                     bool& fresh)
{
    fresh                    = false;
    const uintptr_t stream_k = reinterpret_cast<uintptr_t>(stream);
    APadBuf* best            = nullptr;
    for(auto& buf : apad_bufs)
    {
        if(buf.device == device && buf.stream == stream_k && buf.elem == elem && buf.k == k &&
           buf.rows >= rows && (best == nullptr || buf.rows < best->rows))
            best = &buf;
    }
    if(best != nullptr)
        return *best;

    void* ptr = nullptr;
    HIP_CALL(hipMalloc(&ptr, row_bytes * static_cast<size_t>(rows)));
    HIP_CALL(hipMemsetAsync(ptr, 0, row_bytes * static_cast<size_t>(rows), stream));
    apad_bufs.push_back(APadBuf{device, stream_k, elem, k, rows, 0, false, ptr});
    fresh = true;
    return apad_bufs.back();
}
} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gemm_a16w16_asm,
    (aiter_tensor_t* A,
                     aiter_tensor_t* B,
                     aiter_tensor_t* out,
                     aiter_tensor_t* semaphore,
                     aiter_tensor_t* bias,
                     int          splitK,
                     const char*  kernelName,
                     int          bpreshuffle,
                     hipStream_t  stream),
    (A, B, out, semaphore, bias, splitK, kernelName, bpreshuffle, stream))
{
    AITER_CHECK(A->dtype() == AITER_DTYPE_bf16 || A->dtype() == AITER_DTYPE_fp16,
                "GEMM A16W16 asm: A must be Bf16 or Fp16, got ", AiterDtype_to_str(A->dtype()));
    AITER_CHECK(B->dtype() == AITER_DTYPE_bf16 || B->dtype() == AITER_DTYPE_fp16,
                "GEMM A16W16 asm: B must be Bf16 or Fp16, got ", AiterDtype_to_str(B->dtype()));
    AITER_CHECK(out->dtype() == AITER_DTYPE_fp32 || out->dtype() == AITER_DTYPE_bf16,
                "GEMM A16W16 asm: out must be Float32 or Bf16, got ", AiterDtype_to_str(out->dtype()));

    const HipDeviceGuard device_guard(A->device_id);

    std::string arch_id = get_gpu_arch();
    int Mdim            = A->size(0);
    int Ndim            = B->size(0);
    int Kdim            = A->size(1);

    unsigned int SUBM = 32;
    unsigned int SUBN = 64;

    KernelArgs args = {};
    args.ptr_D      = out->ptr;
    args.ptr_C      = nullptr;
    args.ptr_A      = A->ptr;
    args.ptr_B      = B->ptr;
    args.ptr_Bias   = bias ? bias->ptr : nullptr;
    args.alpha      = 1.0f;
    args.beta       = 0.0f;
    args.stride_A0  = A->stride(0) * A->element_size();
    args.stride_B0  = B->stride(0) * B->element_size();
    args.stride_C0 = args.stride_D0 = Ndim * out->element_size();
    args.M                          = Mdim;
    args.N                          = Ndim;
    args.K                          = Kdim;
    args.is_out_b16                 = (out->dtype() == AITER_DTYPE_bf16) ? 1 : 0;
    args.add_bias                   = bias ? 1 : 0;

    CFG* config_map = &cfg_bf16gemm_fp32bf16;

    auto [name, split] = get_heuristic_kernel(
        Mdim, Ndim, Kdim, config_map, arch_id, bpreshuffle, args.add_bias, splitK, kernelName);
    args.splitk              = split;
    AiterAsmKernel* impl_ptr = get_or_load_kernel(name, config_map, SUBM, SUBN);

    // The asm kernels load a full SUBM-row tile of A. Stores are predicated on M,
    // so the extra rows do not change the result, but the loads are not predicated.
    // The A buffer starts at stride_A0 * M bytes and the K prologue shrinks that;
    // a record count of 0 turns the bound off and the extra rows fault
    // (ROCm/aiter#5781). Stage a packed tile with a zero tail instead.
    //
    // The tile is reused. While M does not shrink, only the live rows are copied.
    // A captured launch always clears its own tail, so replay does not depend on
    // zeros left behind by an earlier shape.
    std::unique_lock<std::mutex> pad_lock(apad_mu, std::defer_lock);
    if(Mdim < static_cast<int>(SUBM))
    {
        const size_t row_bytes = static_cast<size_t>(Kdim) * A->element_size();
        pad_lock.lock();
        bool fresh  = false;
        APadBuf& pad = apad_buffer(A->device_id,
                                   stream,
                                   A->element_size(),
                                   Kdim,
                                   static_cast<int>(SUBM),
                                   row_bytes,
                                   fresh);
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        hipError_t capture_err       = hipStreamIsCapturing(stream, &capture);
        if(capture_err == hipSuccess && capture == hipStreamCaptureStatusActive)
            pad.captured = true;
        else if(capture_err != hipSuccess)
            (void)hipGetLastError();
        if(!fresh && (pad.captured || pad.live > Mdim))
        {
            const int clear_end = pad.captured ? static_cast<int>(SUBM) : pad.live;
            if(clear_end > Mdim)
            {
                HIP_CALL(hipMemsetAsync(static_cast<char*>(pad.ptr) +
                                            row_bytes * static_cast<size_t>(Mdim),
                                        0,
                                        row_bytes * static_cast<size_t>(clear_end - Mdim),
                                        stream));
            }
        }
        if(Mdim > 0)
        {
            if(static_cast<size_t>(args.stride_A0) == row_bytes)
            {
                HIP_CALL(hipMemcpyAsync(pad.ptr,
                                        A->ptr,
                                        row_bytes * static_cast<size_t>(Mdim),
                                        hipMemcpyDeviceToDevice,
                                        stream));
            }
            else
            {
                HIP_CALL(hipMemcpy2DAsync(pad.ptr,
                                          row_bytes,
                                          A->ptr,
                                          static_cast<size_t>(args.stride_A0),
                                          row_bytes,
                                          static_cast<size_t>(Mdim),
                                          hipMemcpyDeviceToDevice,
                                          stream));
            }
        }
        pad.live       = Mdim;
        args.ptr_A     = pad.ptr;
        args.stride_A0 = static_cast<unsigned int>(row_bytes);
    }

    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = (Mdim + SUBM - 1) / SUBM;
    int gdz = split;

    if(split > 1)
    {
        AITER_CHECK(gdx * gdy <= 1024, __func__, " gdx * gdy (", gdx * gdy, ") must be <= 16*64");
    }
    if(split > 1 && semaphore->numel() > 0)
    {
        args.ptr_semaphore = semaphore->ptr;
    }
    else
    {
        args.ptr_semaphore = nullptr;
    }

    size_t arg_size = sizeof(args);
    impl_ptr->launch_kernel({&args, &arg_size, gdx, gdy, gdz, 256, 1, 1, stream});
}
