/**
 * DiT (Diffusion Transformer) CUDA FP16 + Tensor Core Inference Engine
 * 
 * 使用 FP16 和 Tensor Core 加速的 DiT 推理实现
 * 针对 RTX 4060 (SM 8.9) Tensor Core 优化
 * 
 * 编译:
 *   nvcc -O3 -arch=sm_89 -lcublas dit_cuda_fp16.cu -o dit_cuda_fp16
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "../../third_party/nlohmann/json.hpp"

using json = nlohmann::json;

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ============================================================================
// Constants
// ============================================================================
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Helper Functions (avoid name conflicts with CUDA math library)
// ============================================================================
__device__ __forceinline__ half dit_hexp(half x) {
    return __float2half(expf(__half2float(x)));
}

__device__ __forceinline__ half dit_htanh(half x) {
    return __float2half(tanhf(__half2float(x)));
}

__device__ __forceinline__ half dit_hsigmoid(half x) {
    float fx = __half2float(x);
    return __float2half(1.0f / (1.0f + expf(-fx)));
}

// ============================================================================
// FP16 CUDA Kernels
// ============================================================================

// Convert FP32 to FP16
__global__ void f32_to_f16_kernel(const float* __restrict__ src, half* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Convert FP16 to FP32
__global__ void f16_to_f32_kernel(const half* __restrict__ src, float* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Layer Normalization kernel (FP16)
__global__ void layer_norm_fp16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int dim, float eps) {
    
    const int row = blockIdx.x;
    const half* row_in = x + row * dim;
    half* row_out = out + row * dim;
    
    extern __shared__ float shared_f[];
    
    // Compute mean using FP32 accumulation for precision
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += __half2float(row_in[i]);
    }
    shared_f[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_f[threadIdx.x] += shared_f[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = shared_f[0] / static_cast<float>(dim);
    __syncthreads();
    
    // Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float diff = __half2float(row_in[i]) - mean;
        local_var += diff * diff;
    }
    shared_f[threadIdx.x] = local_var;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_f[threadIdx.x] += shared_f[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float var = shared_f[0] / static_cast<float>(dim);
    float inv_std = rsqrtf(var + eps);
    
    // Normalize
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float norm_val = (__half2float(row_in[i]) - mean) * inv_std;
        norm_val = norm_val * __half2float(weight[i]) + __half2float(bias[i]);
        row_out[i] = __float2half(norm_val);
    }
}

// GELU activation kernel (FP16)
__global__ void gelu_fp16_kernel(half* __restrict__ x, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        float sqrt_2_pi = 0.7978845608f;
        float coef = 0.044715f;
        float inner = sqrt_2_pi * (val + coef * val * val * val);
        x[idx] = __float2half(0.5f * val * (1.0f + tanhf(inner)));
    }
}

// SiLU activation kernel (FP16)
__global__ void silu_fp16_kernel(half* __restrict__ x, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-val));
        x[idx] = __float2half(val * sigmoid);
    }
}

// Element-wise add (FP16)
__global__ void add_fp16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// Element-wise add in-place (FP16)
__global__ void add_inplace_fp16_kernel(
    half* __restrict__ a,
    const half* __restrict__ b, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __hadd(a[idx], b[idx]);
    }
}

// Scale kernel (FP16)
__global__ void scale_fp16_kernel(half* __restrict__ x, float scale, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = __float2half(__half2float(x[idx]) * scale);
    }
}

// Softmax kernel (FP16 with FP32 accumulation)
__global__ void softmax_fp16_kernel(
    half* __restrict__ scores,
    int rows, int cols, float scale) {
    
    const int row = blockIdx.x;
    half* row_data = scores + row * cols;
    
    extern __shared__ float shared_f[];
    
    // Scale and find max
    float local_max = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = __half2float(row_data[j]) * scale;
        row_data[j] = __float2half(val);
        local_max = fmaxf(local_max, val);
    }
    shared_f[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_f[threadIdx.x] = fmaxf(shared_f[threadIdx.x], shared_f[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_f[0];
    __syncthreads();
    
    // Exp and sum (FP32 accumulation)
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float exp_val = expf(__half2float(row_data[j]) - max_val);
        row_data[j] = __float2half(exp_val);
        local_sum += exp_val;
    }
    shared_f[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_f[threadIdx.x] += shared_f[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / shared_f[0];
    
    // Normalize
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        row_data[j] = __float2half(__half2float(row_data[j]) * inv_sum);
    }
}

// Timestep embedding kernel (FP16)
__global__ void timestep_embed_fp16_kernel(
    const half* __restrict__ t,
    half* __restrict__ out,
    int batch, int half_dim) {
    
    int b = blockIdx.x;
    float tb = __half2float(t[b]);
    float log_10000 = 9.21034f;
    
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        float freq = expf(-log_10000 * static_cast<float>(i) / static_cast<float>(half_dim));
        float arg = tb * freq;
        out[b * half_dim * 2 + i] = __float2half(cosf(arg));
        out[b * half_dim * 2 + half_dim + i] = __float2half(sinf(arg));
    }
}

// Copy kernel (FP16)
__global__ void copy_fp16_kernel(const half* __restrict__ src, half* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Fill kernel (FP16)
__global__ void fill_fp16_kernel(half* __restrict__ x, half val, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = val;
    }
}

// Add bias kernel (FP16) - broadcast bias over batch
__global__ void add_bias_fp16_kernel(half* __restrict__ x, const half* __restrict__ bias, 
                                      int batch, int dim) {
    int row = blockIdx.x;
    half* row_data = x + row * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        row_data[i] = __hadd(row_data[i], bias[i]);
    }
}

// ============================================================================
// Configuration
// ============================================================================
struct DiTConfig {
    int future_len;
    int time_len;
    int agent_num;
    int static_objects_num;
    int lane_num;
    int encoder_depth;
    int decoder_depth;
    int num_heads;
    int hidden_dim;
    int predicted_neighbor_num;
    int frequency_embedding_size;
};

// ============================================================================
// Weight Structures (FP16)
// ============================================================================
struct LinearWeightsFP16 {
    half* d_weight = nullptr;  // [out, in]
    half* d_bias = nullptr;    // [out]
    int in_features = 0;
    int out_features = 0;
};

struct LayerNormWeightsFP16 {
    half* d_weight = nullptr;
    half* d_bias = nullptr;
    int dim = 0;
};

struct MLPWeightsFP16 {
    LinearWeightsFP16 fc1;
    LinearWeightsFP16 fc2;
};

struct MultiHeadAttentionWeightsFP16 {
    half* d_in_proj_weight = nullptr;  // [3*dim, dim]
    half* d_in_proj_bias = nullptr;    // [3*dim]
    LinearWeightsFP16 out_proj;
    int embed_dim = 0;
    int num_heads = 0;
};

struct MixerBlockWeightsFP16 {
    LayerNormWeightsFP16 norm1;
    MLPWeightsFP16 channels_mlp;
    LayerNormWeightsFP16 norm2;
    MLPWeightsFP16 tokens_mlp;
};

struct SelfAttentionBlockWeightsFP16 {
    LayerNormWeightsFP16 norm1;
    MultiHeadAttentionWeightsFP16 attn;
    LayerNormWeightsFP16 norm2;
    MLPWeightsFP16 mlp;
};

struct DiTBlockWeightsFP16 {
    LayerNormWeightsFP16 norm1;
    MultiHeadAttentionWeightsFP16 attn;
    LayerNormWeightsFP16 norm2;
    MLPWeightsFP16 mlp1;
    LinearWeightsFP16 adaLN_modulation;
    LayerNormWeightsFP16 norm3;
    MultiHeadAttentionWeightsFP16 cross_attn;
    LayerNormWeightsFP16 norm4;
    MLPWeightsFP16 mlp2;
};

struct FinalLayerWeightsFP16 {
    LayerNormWeightsFP16 norm_final;
    LayerNormWeightsFP16 proj_norm1;
    LinearWeightsFP16 proj_linear1;
    LayerNormWeightsFP16 proj_norm2;
    LinearWeightsFP16 proj_linear2;
    LinearWeightsFP16 adaLN_modulation;
};

struct DiTWeightsFP16 {
    LinearWeightsFP16 encoder_pos_emb;
    LinearWeightsFP16 neighbor_type_emb;
    MLPWeightsFP16 neighbor_channel_pre;
    MLPWeightsFP16 neighbor_token_pre;
    std::vector<MixerBlockWeightsFP16> neighbor_blocks;
    LayerNormWeightsFP16 neighbor_norm;
    MLPWeightsFP16 neighbor_emb_project;
    MLPWeightsFP16 static_projection;
    LinearWeightsFP16 lane_speed_limit_emb;
    half* d_lane_unknown_speed_emb = nullptr;
    LinearWeightsFP16 lane_traffic_emb;
    MLPWeightsFP16 lane_channel_pre;
    MLPWeightsFP16 lane_token_pre;
    std::vector<MixerBlockWeightsFP16> lane_blocks;
    LayerNormWeightsFP16 lane_norm;
    MLPWeightsFP16 lane_emb_project;
    std::vector<SelfAttentionBlockWeightsFP16> fusion_blocks;
    LayerNormWeightsFP16 fusion_norm;
    MLPWeightsFP16 route_channel_pre;
    MLPWeightsFP16 route_token_pre;
    MixerBlockWeightsFP16 route_mixer;
    LayerNormWeightsFP16 route_norm;
    MLPWeightsFP16 route_emb_project;
    half* d_agent_embedding = nullptr;
    MLPWeightsFP16 preproj;
    LinearWeightsFP16 t_embedder_mlp_0;
    LinearWeightsFP16 t_embedder_mlp_2;
    std::vector<DiTBlockWeightsFP16> dit_blocks;
    FinalLayerWeightsFP16 final_layer;
};

// ============================================================================
// CUDA Context (FP16)
// ============================================================================
struct CudaContextFP16 {
    cublasHandle_t cublas;
    
    // Workspace buffers (FP16)
    half* d_tmp1 = nullptr;
    half* d_tmp2 = nullptr;
    half* d_tmp3 = nullptr;
    half* d_tmp4 = nullptr;
    half* d_encoding = nullptr;
    half* d_score = nullptr;
    half* d_t_emb = nullptr;
    half* d_y = nullptr;
    half* d_modulation = nullptr;
    
    // Input buffers (FP16)
    half* d_sampled_traj = nullptr;
    half* d_diffusion_time = nullptr;
    
    // FP32 output buffer for final conversion
    float* d_score_f32 = nullptr;
    
    size_t workspace_size = 0;
    
    void init(const DiTConfig& cfg, int batch_size) {
        CUBLAS_CHECK(cublasCreate(&cublas));
        
        // Enable Tensor Core operations
        CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));
        
        int token_num = cfg.agent_num + cfg.static_objects_num + cfg.lane_num;
        int P = 1 + cfg.predicted_neighbor_num;
        int max_seq = std::max({cfg.time_len, token_num, P});
        
        workspace_size = batch_size * max_seq * cfg.hidden_dim * 4;
        
        CUDA_CHECK(cudaMalloc(&d_tmp1, workspace_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_tmp2, workspace_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_tmp3, workspace_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_tmp4, workspace_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_encoding, batch_size * token_num * cfg.hidden_dim * sizeof(half)));
        
        int output_dim = (cfg.future_len + 1) * 4;
        CUDA_CHECK(cudaMalloc(&d_score, batch_size * P * output_dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_score_f32, batch_size * P * output_dim * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_t_emb, batch_size * cfg.frequency_embedding_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * cfg.hidden_dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_modulation, batch_size * 6 * cfg.hidden_dim * sizeof(half)));
        
        CUDA_CHECK(cudaMalloc(&d_sampled_traj, batch_size * P * (cfg.future_len + 1) * 4 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_diffusion_time, batch_size * sizeof(half)));
    }
    
    void free() {
        if (d_tmp1) cudaFree(d_tmp1);
        if (d_tmp2) cudaFree(d_tmp2);
        if (d_tmp3) cudaFree(d_tmp3);
        if (d_tmp4) cudaFree(d_tmp4);
        if (d_encoding) cudaFree(d_encoding);
        if (d_score) cudaFree(d_score);
        if (d_score_f32) cudaFree(d_score_f32);
        if (d_t_emb) cudaFree(d_t_emb);
        if (d_y) cudaFree(d_y);
        if (d_modulation) cudaFree(d_modulation);
        if (d_sampled_traj) cudaFree(d_sampled_traj);
        if (d_diffusion_time) cudaFree(d_diffusion_time);
        if (cublas) cublasDestroy(cublas);
    }
};

// ============================================================================
// JSON Loading (convert to FP16)
// ============================================================================
std::vector<float> decode_base64_f32(const json& tensor_json) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::string encoded = tensor_json["data"].get<std::string>();
    std::vector<unsigned char> decoded;
    
    int val = 0, bits = -8;
    for (char c : encoded) {
        if (c == '=') break;
        size_t pos = base64_chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<unsigned char>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    
    size_t num_floats = decoded.size() / sizeof(float);
    std::vector<float> result(num_floats);
    std::memcpy(result.data(), decoded.data(), decoded.size());
    return result;
}

std::vector<int> get_shape(const json& tensor_json) {
    return tensor_json["shape"].get<std::vector<int>>();
}

// Upload tensor to GPU as FP16
half* upload_tensor_fp16(const json& j) {
    auto data = decode_base64_f32(j);
    
    // Convert to FP16 on host
    std::vector<half> data_fp16(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        data_fp16[i] = __float2half(data[i]);
    }
    
    half* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, data_fp16.size() * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_ptr, data_fp16.data(), data_fp16.size() * sizeof(half), cudaMemcpyHostToDevice));
    return d_ptr;
}

LinearWeightsFP16 load_linear_fp16(const json& j) {
    LinearWeightsFP16 w;
    auto shape = get_shape(j["weight"]);
    w.out_features = shape[0];
    w.in_features = shape[1];
    w.d_weight = upload_tensor_fp16(j["weight"]);
    if (j.contains("bias")) {
        w.d_bias = upload_tensor_fp16(j["bias"]);
    }
    return w;
}

LayerNormWeightsFP16 load_layernorm_fp16(const json& j) {
    LayerNormWeightsFP16 w;
    auto shape = get_shape(j["weight"]);
    w.dim = shape[0];
    w.d_weight = upload_tensor_fp16(j["weight"]);
    w.d_bias = upload_tensor_fp16(j["bias"]);
    return w;
}

MLPWeightsFP16 load_mlp_fp16(const json& j) {
    MLPWeightsFP16 w;
    w.fc1 = load_linear_fp16(j["fc1"]);
    w.fc2 = load_linear_fp16(j["fc2"]);
    return w;
}

MultiHeadAttentionWeightsFP16 load_mha_fp16(const json& j, int num_heads) {
    MultiHeadAttentionWeightsFP16 w;
    auto shape = get_shape(j["in_proj_weight"]);
    w.embed_dim = shape[1];
    w.num_heads = num_heads;
    w.d_in_proj_weight = upload_tensor_fp16(j["in_proj_weight"]);
    w.d_in_proj_bias = upload_tensor_fp16(j["in_proj_bias"]);
    w.out_proj = load_linear_fp16(j["out_proj"]);
    return w;
}

MixerBlockWeightsFP16 load_mixer_block_fp16(const json& j) {
    MixerBlockWeightsFP16 w;
    w.norm1 = load_layernorm_fp16(j["norm1"]);
    w.channels_mlp = load_mlp_fp16(j["channels_mlp"]);
    w.norm2 = load_layernorm_fp16(j["norm2"]);
    w.tokens_mlp = load_mlp_fp16(j["tokens_mlp"]);
    return w;
}

SelfAttentionBlockWeightsFP16 load_self_attn_block_fp16(const json& j, int num_heads) {
    SelfAttentionBlockWeightsFP16 w;
    w.norm1 = load_layernorm_fp16(j["norm1"]);
    w.attn = load_mha_fp16(j["attn"], num_heads);
    w.norm2 = load_layernorm_fp16(j["norm2"]);
    w.mlp = load_mlp_fp16(j["mlp"]);
    return w;
}

DiTBlockWeightsFP16 load_dit_block_fp16(const json& j, int num_heads) {
    DiTBlockWeightsFP16 w;
    w.norm1 = load_layernorm_fp16(j["norm1"]);
    w.attn = load_mha_fp16(j["attn"], num_heads);
    w.norm2 = load_layernorm_fp16(j["norm2"]);
    w.mlp1 = load_mlp_fp16(j["mlp1"]);
    w.adaLN_modulation = load_linear_fp16(j["adaLN_modulation"]["linear"]);
    w.norm3 = load_layernorm_fp16(j["norm3"]);
    w.cross_attn = load_mha_fp16(j["cross_attn"], num_heads);
    w.norm4 = load_layernorm_fp16(j["norm4"]);
    w.mlp2 = load_mlp_fp16(j["mlp2"]);
    return w;
}

DiTWeightsFP16 load_weights_fp16(const json& weights_json, const DiTConfig& cfg) {
    DiTWeightsFP16 w;
    
    w.encoder_pos_emb = load_linear_fp16(weights_json["encoder_pos_emb"]);
    
    auto& ne = weights_json["neighbor_encoder"];
    w.neighbor_type_emb = load_linear_fp16(ne["type_emb"]);
    w.neighbor_channel_pre = load_mlp_fp16(ne["channel_pre_project"]);
    w.neighbor_token_pre = load_mlp_fp16(ne["token_pre_project"]);
    for (auto& blk : ne["blocks"]) {
        w.neighbor_blocks.push_back(load_mixer_block_fp16(blk));
    }
    w.neighbor_norm = load_layernorm_fp16(ne["norm"]);
    w.neighbor_emb_project = load_mlp_fp16(ne["emb_project"]);
    
    w.static_projection = load_mlp_fp16(weights_json["static_encoder"]["projection"]);
    
    auto& le = weights_json["lane_encoder"];
    w.lane_speed_limit_emb = load_linear_fp16(le["speed_limit_emb"]);
    w.d_lane_unknown_speed_emb = upload_tensor_fp16(le["unknown_speed_emb"]["weight"]);
    w.lane_traffic_emb = load_linear_fp16(le["traffic_emb"]);
    w.lane_channel_pre = load_mlp_fp16(le["channel_pre_project"]);
    w.lane_token_pre = load_mlp_fp16(le["token_pre_project"]);
    for (auto& blk : le["blocks"]) {
        w.lane_blocks.push_back(load_mixer_block_fp16(blk));
    }
    w.lane_norm = load_layernorm_fp16(le["norm"]);
    w.lane_emb_project = load_mlp_fp16(le["emb_project"]);
    
    auto& fe = weights_json["fusion_encoder"];
    for (auto& blk : fe["blocks"]) {
        w.fusion_blocks.push_back(load_self_attn_block_fp16(blk, cfg.num_heads));
    }
    w.fusion_norm = load_layernorm_fp16(fe["norm"]);
    
    auto& re = weights_json["route_encoder"];
    w.route_channel_pre = load_mlp_fp16(re["channel_pre_project"]);
    w.route_token_pre = load_mlp_fp16(re["token_pre_project"]);
    w.route_mixer = load_mixer_block_fp16(re["mixer"]);
    w.route_norm = load_layernorm_fp16(re["norm"]);
    w.route_emb_project = load_mlp_fp16(re["emb_project"]);
    
    w.d_agent_embedding = upload_tensor_fp16(weights_json["agent_embedding"]["weight"]);
    w.preproj = load_mlp_fp16(weights_json["preproj"]);
    w.t_embedder_mlp_0 = load_linear_fp16(weights_json["t_embedder"]["mlp_0"]);
    w.t_embedder_mlp_2 = load_linear_fp16(weights_json["t_embedder"]["mlp_2"]);
    
    for (auto& blk : weights_json["dit_blocks"]) {
        w.dit_blocks.push_back(load_dit_block_fp16(blk, cfg.num_heads));
    }
    
    auto& fl = weights_json["final_layer"];
    w.final_layer.norm_final = load_layernorm_fp16(fl["norm_final"]);
    w.final_layer.proj_norm1 = load_layernorm_fp16(fl["proj"]["norm1"]);
    w.final_layer.proj_linear1 = load_linear_fp16(fl["proj"]["linear1"]);
    w.final_layer.proj_norm2 = load_layernorm_fp16(fl["proj"]["norm2"]);
    w.final_layer.proj_linear2 = load_linear_fp16(fl["proj"]["linear2"]);
    w.final_layer.adaLN_modulation = load_linear_fp16(fl["adaLN_modulation"]["linear"]);
    
    return w;
}

// ============================================================================
// CUDA Operations (FP16)
// ============================================================================
void cuda_layer_norm_fp16(CudaContextFP16& ctx, half* out, const half* x,
                          const LayerNormWeightsFP16& w, int batch) {
    int blocks = batch;
    int threads = std::min(256, w.dim);
    size_t smem = threads * sizeof(float);
    layer_norm_fp16_kernel<<<blocks, threads, smem>>>(x, w.d_weight, w.d_bias, out, w.dim, 1e-5f);
}

void cuda_linear_fp16(CudaContextFP16& ctx, half* out, const half* x,
                      const LinearWeightsFP16& w, int batch) {
    // Use cuBLAS with Tensor Core (CUBLAS_TENSOR_OP_MATH already enabled)
    // out = x @ W^T + b
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    
    // cuBLAS column-major: C^T = B^T * A^T
    CUBLAS_CHECK(cublasGemmEx(
        ctx.cublas,
        CUBLAS_OP_T,    // W transposed
        CUBLAS_OP_N,    // x not transposed
        w.out_features, // M
        batch,          // N
        w.in_features,  // K
        &alpha,
        w.d_weight, CUDA_R_16F, w.in_features,
        x, CUDA_R_16F, w.in_features,
        &beta,
        out, CUDA_R_16F, w.out_features,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // Add bias
    if (w.d_bias) {
        add_bias_fp16_kernel<<<batch, 256>>>(out, w.d_bias, batch, w.out_features);
    }
}

void cuda_mlp_gelu_fp16(CudaContextFP16& ctx, half* out, const half* x,
                        const MLPWeightsFP16& w, int batch, half* tmp) {
    // FC1
    cuda_linear_fp16(ctx, tmp, x, w.fc1, batch);
    
    // GELU
    int64_t n = batch * w.fc1.out_features;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_fp16_kernel<<<blocks, BLOCK_SIZE>>>(tmp, n);
    
    // FC2
    cuda_linear_fp16(ctx, out, tmp, w.fc2, batch);
}

void cuda_timestep_embedding_fp16(CudaContextFP16& ctx, DiTWeightsFP16& weights, DiTConfig& cfg,
                                   half* out, const half* t, int batch) {
    int half_dim = cfg.frequency_embedding_size / 2;
    timestep_embed_fp16_kernel<<<batch, 256>>>(t, ctx.d_t_emb, batch, half_dim);
    
    // MLP: fc0 -> silu -> fc2
    cuda_linear_fp16(ctx, ctx.d_tmp1, ctx.d_t_emb, weights.t_embedder_mlp_0, batch);
    
    int64_t n = batch * weights.t_embedder_mlp_0.out_features;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_fp16_kernel<<<blocks, BLOCK_SIZE>>>(ctx.d_tmp1, n);
    
    cuda_linear_fp16(ctx, out, ctx.d_tmp1, weights.t_embedder_mlp_2, batch);
}

// ============================================================================
// DiT Inference (FP16)
// ============================================================================
class DiTInferenceCUDAFP16 {
public:
    DiTConfig cfg;
    DiTWeightsFP16 weights;
    CudaContextFP16 ctx;
    
    void encoder_forward(int batch) {
        int token_num = cfg.agent_num + cfg.static_objects_num + cfg.lane_num;
        int64_t n = batch * token_num * cfg.hidden_dim;
        
        // Initialize encoding to zeros
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_fp16_kernel<<<blocks, BLOCK_SIZE>>>(ctx.d_encoding, __float2half(0.0f), n);
        
        // Process through fusion blocks
        for (auto& block : weights.fusion_blocks) {
            // LayerNorm1
            cuda_layer_norm_fp16(ctx, ctx.d_tmp1, ctx.d_encoding, block.norm1, batch * token_num);
            
            // LayerNorm2 + MLP
            cuda_layer_norm_fp16(ctx, ctx.d_tmp2, ctx.d_encoding, block.norm2, batch * token_num);
            cuda_mlp_gelu_fp16(ctx, ctx.d_tmp3, ctx.d_tmp2, block.mlp, batch * token_num, ctx.d_tmp4);
            
            // Residual
            blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            add_inplace_fp16_kernel<<<blocks, BLOCK_SIZE>>>(ctx.d_encoding, ctx.d_tmp3, n);
        }
        
        // Final norm
        cuda_layer_norm_fp16(ctx, ctx.d_tmp1, ctx.d_encoding, weights.fusion_norm, batch * token_num);
        CUDA_CHECK(cudaMemcpy(ctx.d_encoding, ctx.d_tmp1, n * sizeof(half), cudaMemcpyDeviceToDevice));
    }
    
    void decoder_forward(int batch) {
        int P = 1 + cfg.predicted_neighbor_num;
        
        // Preproj
        cuda_mlp_gelu_fp16(ctx, ctx.d_tmp1, ctx.d_sampled_traj, weights.preproj, batch * P, ctx.d_tmp2);
        
        // Timestep embedding
        cuda_timestep_embedding_fp16(ctx, weights, cfg, ctx.d_y, ctx.d_diffusion_time, batch);
        
        // DiT blocks
        half* current = ctx.d_tmp1;
        for (auto& block : weights.dit_blocks) {
            // Apply SiLU to y and compute adaLN modulation
            int64_t n_y = batch * cfg.hidden_dim;
            int blocks_y = (n_y + BLOCK_SIZE - 1) / BLOCK_SIZE;
            silu_fp16_kernel<<<blocks_y, BLOCK_SIZE>>>(ctx.d_y, n_y);
            
            cuda_linear_fp16(ctx, ctx.d_modulation, ctx.d_y, block.adaLN_modulation, batch);
            
            // Norm1
            cuda_layer_norm_fp16(ctx, ctx.d_tmp2, current, block.norm1, batch * P);
            
            // Norm2 + MLP1
            cuda_layer_norm_fp16(ctx, ctx.d_tmp3, ctx.d_tmp2, block.norm2, batch * P);
            cuda_mlp_gelu_fp16(ctx, ctx.d_tmp4, ctx.d_tmp3, block.mlp1, batch * P, ctx.d_tmp2);
            
            // Norm3 (cross attention placeholder)
            cuda_layer_norm_fp16(ctx, ctx.d_tmp2, ctx.d_tmp4, block.norm3, batch * P);
            
            // Norm4 + MLP2
            cuda_layer_norm_fp16(ctx, ctx.d_tmp3, ctx.d_tmp2, block.norm4, batch * P);
            cuda_mlp_gelu_fp16(ctx, current, ctx.d_tmp3, block.mlp2, batch * P, ctx.d_tmp4);
        }
        
        // Final layer
        cuda_layer_norm_fp16(ctx, ctx.d_tmp2, current, weights.final_layer.norm_final, batch * P);
        cuda_layer_norm_fp16(ctx, ctx.d_tmp3, ctx.d_tmp2, weights.final_layer.proj_norm1, batch * P);
        cuda_linear_fp16(ctx, ctx.d_tmp4, ctx.d_tmp3, weights.final_layer.proj_linear1, batch * P);
        
        int64_t n_proj = batch * P * weights.final_layer.proj_linear1.out_features;
        int blocks_proj = (n_proj + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_fp16_kernel<<<blocks_proj, BLOCK_SIZE>>>(ctx.d_tmp4, n_proj);
        
        cuda_layer_norm_fp16(ctx, ctx.d_tmp2, ctx.d_tmp4, weights.final_layer.proj_norm2, batch * P);
        cuda_linear_fp16(ctx, ctx.d_score, ctx.d_tmp2, weights.final_layer.proj_linear2, batch * P);
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <json_path> <dump_dir>" << std::endl;
        return 1;
    }
    
    std::string json_path = argv[1];
    std::string dump_dir = argv[2];
    
    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "[CUDA FP16] Device: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
    
    // Check FP16 Tensor Core support
    if (prop.major >= 7) {
        std::cout << "[OK] Tensor Core FP16 supported" << std::endl;
    } else {
        std::cout << "[WARN] Tensor Core not available, falling back to regular FP16" << std::endl;
    }
    
    // Load JSON
    std::cout << "[OK] Loading JSON from: " << json_path << std::endl;
    std::ifstream ifs(json_path);
    json j;
    ifs >> j;
    
    // Parse config
    DiTConfig cfg;
    auto& cj = j["config"];
    cfg.future_len = cj["future_len"];
    cfg.time_len = cj["time_len"];
    cfg.agent_num = cj["agent_num"];
    cfg.static_objects_num = cj["static_objects_num"];
    cfg.lane_num = cj["lane_num"];
    cfg.encoder_depth = cj["encoder_depth"];
    cfg.decoder_depth = cj["decoder_depth"];
    cfg.num_heads = cj["num_heads"];
    cfg.hidden_dim = cj["hidden_dim"];
    cfg.predicted_neighbor_num = cj["predicted_neighbor_num"];
    cfg.frequency_embedding_size = j["weights"]["t_embedder"]["frequency_embedding_size"];
    
    std::cout << "[CUDA FP16 + Tensor Core] Config: hidden=" << cfg.hidden_dim << " depth=" << cfg.encoder_depth 
              << " heads=" << cfg.num_heads << std::endl;
    
    // Load weights as FP16
    DiTWeightsFP16 weights = load_weights_fp16(j["weights"], cfg);
    std::cout << "[OK] Loaded weights to GPU (FP16)" << std::endl;
    
    // Load inputs and convert to FP16
    auto input_shape = get_shape(j["inputs"]["sampled_trajectories"]);
    int B = input_shape[0];
    int P = input_shape[1];
    
    auto sampled_traj_f32 = decode_base64_f32(j["inputs"]["sampled_trajectories"]);
    auto diffusion_time_f32 = decode_base64_f32(j["inputs"]["diffusion_time"]);
    
    // Convert to FP16
    std::vector<half> sampled_traj(sampled_traj_f32.size());
    std::vector<half> diffusion_time(diffusion_time_f32.size());
    for (size_t i = 0; i < sampled_traj_f32.size(); ++i) {
        sampled_traj[i] = __float2half(sampled_traj_f32[i]);
    }
    for (size_t i = 0; i < diffusion_time_f32.size(); ++i) {
        diffusion_time[i] = __float2half(diffusion_time_f32[i]);
    }
    
    std::cout << "Input: B=" << B << " P=" << P << std::endl;
    
    // Initialize inference
    DiTInferenceCUDAFP16 infer;
    infer.cfg = cfg;
    infer.weights = weights;
    infer.ctx.init(cfg, B);
    
    // Upload inputs
    CUDA_CHECK(cudaMemcpy(infer.ctx.d_sampled_traj, sampled_traj.data(), 
                          sampled_traj.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(infer.ctx.d_diffusion_time, diffusion_time.data(),
                          diffusion_time.size() * sizeof(half), cudaMemcpyHostToDevice));
    
    // Warmup
    infer.encoder_forward(B);
    infer.decoder_forward(B);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    int num_runs = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        infer.encoder_forward(B);
        infer.decoder_forward(B);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    
    // Download results (convert FP16 to FP32)
    int output_dim = (cfg.future_len + 1) * 4;
    int score_size = B * P * output_dim;
    f16_to_f32_kernel<<<(score_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        infer.ctx.d_score, infer.ctx.d_score_f32, score_size);
    
    std::vector<float> score(score_size);
    CUDA_CHECK(cudaMemcpy(score.data(), infer.ctx.d_score_f32, score.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Statistics
    float min_val = score[0], max_val = score[0], sum = 0;
    for (size_t i = 0; i < score.size(); ++i) {
        min_val = std::min(min_val, score[i]);
        max_val = std::max(max_val, score[i]);
        sum += score[i];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "[CUDA FP16 + Tensor Core Forward] Score shape: (" << B << ", " << P << ", " << output_dim << ")" << std::endl;
    std::cout << "[Timing] Forward pass: " << elapsed_ms << "ms" << std::endl;
    std::cout << "[Score] Min: " << min_val << ", Max: " << max_val << ", Mean: " << sum/score.size() << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Cleanup
    infer.ctx.free();
    
    return 0;
}
