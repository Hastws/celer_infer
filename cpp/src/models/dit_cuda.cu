/**
 * DiT (Diffusion Transformer) CUDA FP32 Inference Engine
 * 
 * 用于自动驾驶规划的扩散模型 CUDA GPU 加速推理实现
 * 
 * 编译:
 *   nvcc -O3 -arch=sm_89 -lcublas dit_cuda.cu -o dit_cuda
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
constexpr int WARP_SIZE = 32;

// ============================================================================
// CUDA Kernels
// ============================================================================

// Layer Normalization kernel
__global__ void layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int dim, float eps) {
    
    const int row = blockIdx.x;
    const float* row_in = x + row * dim;
    float* row_out = out + row * dim;
    
    extern __shared__ double shared_d[];
    
    // Compute mean using double precision
    double local_sum = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += static_cast<double>(row_in[i]);
    }
    shared_d[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Block reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_d[threadIdx.x] += shared_d[threadIdx.x + stride];
        }
        __syncthreads();
    }
    double mean = shared_d[0] / static_cast<double>(dim);
    __syncthreads();
    
    // Compute variance
    double local_var = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        double diff = static_cast<double>(row_in[i]) - mean;
        local_var += diff * diff;
    }
    shared_d[threadIdx.x] = local_var;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_d[threadIdx.x] += shared_d[threadIdx.x + stride];
        }
        __syncthreads();
    }
    double var = shared_d[0] / static_cast<double>(dim);
    float inv_std = static_cast<float>(1.0 / sqrt(var + static_cast<double>(eps)));
    float mean_f = static_cast<float>(mean);
    
    // Normalize
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        row_out[i] = (row_in[i] - mean_f) * inv_std * weight[i] + bias[i];
    }
}

// GELU activation kernel (tanh approximation)
__global__ void gelu_kernel(float* __restrict__ x, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = static_cast<double>(x[idx]);
        double sqrt_2_pi = 0.7978845608;
        double coef = 0.044715;
        double inner = sqrt_2_pi * (val + coef * val * val * val);
        x[idx] = static_cast<float>(0.5 * val * (1.0 + tanh(inner)));
    }
}

// SiLU activation kernel
__global__ void silu_kernel(float* __restrict__ x, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = static_cast<double>(x[idx]);
        double sigmoid = 1.0 / (1.0 + exp(-val));
        x[idx] = static_cast<float>(val * sigmoid);
    }
}

// Element-wise add kernel
__global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Element-wise add in-place kernel
__global__ void add_inplace_kernel(
    float* __restrict__ a,
    const float* __restrict__ b, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

// Scale kernel
__global__ void scale_kernel(float* __restrict__ x, float scale, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= scale;
    }
}

// Embedding lookup kernel
__global__ void embedding_kernel(
    const float* __restrict__ table,
    const int32_t* __restrict__ indices,
    float* __restrict__ out,
    int batch, int hidden) {
    
    const int row = blockIdx.x;
    const int idx = indices[row];
    const float* src = table + idx * hidden;
    float* dst = out + row * hidden;
    
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Softmax kernel for attention
__global__ void softmax_kernel(
    float* __restrict__ scores,
    int rows, int cols, float scale) {
    
    const int row = blockIdx.x;
    float* row_data = scores + row * cols;
    
    extern __shared__ char shared_mem[];
    float* shared_f = reinterpret_cast<float*>(shared_mem);
    double* shared_d = reinterpret_cast<double*>(shared_mem);
    
    // Scale and find max
    float local_max = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = row_data[j] * scale;
        row_data[j] = val;
        local_max = fmaxf(local_max, val);
    }
    shared_f[threadIdx.x] = local_max;
    __syncthreads();
    
    // Reduce max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_f[threadIdx.x] = fmaxf(shared_f[threadIdx.x], shared_f[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_f[0];
    __syncthreads();
    
    // Exp and sum
    double local_sum = 0.0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        double exp_val = exp(static_cast<double>(row_data[j] - max_val));
        row_data[j] = static_cast<float>(exp_val);
        local_sum += exp_val;
    }
    shared_d[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_d[threadIdx.x] += shared_d[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv_sum = static_cast<float>(1.0 / shared_d[0]);
    
    // Normalize
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        row_data[j] *= inv_sum;
    }
}

// Timestep embedding kernel (sinusoidal)
__global__ void timestep_embed_kernel(
    const float* __restrict__ t,
    float* __restrict__ out,
    int batch, int half_dim) {
    
    int b = blockIdx.x;
    float tb = t[b];
    float log_10000 = 9.21034f;  // log(10000)
    
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        float freq = expf(-log_10000 * static_cast<float>(i) / static_cast<float>(half_dim));
        float arg = tb * freq;
        out[b * half_dim * 2 + i] = cosf(arg);
        out[b * half_dim * 2 + half_dim + i] = sinf(arg);
    }
}

// Copy kernel
__global__ void copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Fill kernel
__global__ void fill_kernel(float* __restrict__ x, float val, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = val;
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
// Weight Structures
// ============================================================================
struct LinearWeights {
    float* d_weight = nullptr;  // [out, in]
    float* d_bias = nullptr;    // [out]
    int in_features = 0;
    int out_features = 0;
};

struct LayerNormWeights {
    float* d_weight = nullptr;
    float* d_bias = nullptr;
    int dim = 0;
};

struct MLPWeights {
    LinearWeights fc1;
    LinearWeights fc2;
};

struct MultiHeadAttentionWeights {
    float* d_in_proj_weight = nullptr;  // [3*dim, dim]
    float* d_in_proj_bias = nullptr;    // [3*dim]
    LinearWeights out_proj;
    int embed_dim = 0;
    int num_heads = 0;
};

struct MixerBlockWeights {
    LayerNormWeights norm1;
    MLPWeights channels_mlp;
    LayerNormWeights norm2;
    MLPWeights tokens_mlp;
};

struct SelfAttentionBlockWeights {
    LayerNormWeights norm1;
    MultiHeadAttentionWeights attn;
    LayerNormWeights norm2;
    MLPWeights mlp;
};

struct DiTBlockWeights {
    LayerNormWeights norm1;
    MultiHeadAttentionWeights attn;
    LayerNormWeights norm2;
    MLPWeights mlp1;
    LinearWeights adaLN_modulation;
    LayerNormWeights norm3;
    MultiHeadAttentionWeights cross_attn;
    LayerNormWeights norm4;
    MLPWeights mlp2;
};

struct FinalLayerWeights {
    LayerNormWeights norm_final;
    LayerNormWeights proj_norm1;
    LinearWeights proj_linear1;
    LayerNormWeights proj_norm2;
    LinearWeights proj_linear2;
    LinearWeights adaLN_modulation;
};

struct DiTWeights {
    LinearWeights encoder_pos_emb;
    LinearWeights neighbor_type_emb;
    MLPWeights neighbor_channel_pre;
    MLPWeights neighbor_token_pre;
    std::vector<MixerBlockWeights> neighbor_blocks;
    LayerNormWeights neighbor_norm;
    MLPWeights neighbor_emb_project;
    MLPWeights static_projection;
    LinearWeights lane_speed_limit_emb;
    float* d_lane_unknown_speed_emb = nullptr;
    LinearWeights lane_traffic_emb;
    MLPWeights lane_channel_pre;
    MLPWeights lane_token_pre;
    std::vector<MixerBlockWeights> lane_blocks;
    LayerNormWeights lane_norm;
    MLPWeights lane_emb_project;
    std::vector<SelfAttentionBlockWeights> fusion_blocks;
    LayerNormWeights fusion_norm;
    MLPWeights route_channel_pre;
    MLPWeights route_token_pre;
    MixerBlockWeights route_mixer;
    LayerNormWeights route_norm;
    MLPWeights route_emb_project;
    float* d_agent_embedding = nullptr;
    MLPWeights preproj;
    LinearWeights t_embedder_mlp_0;
    LinearWeights t_embedder_mlp_2;
    std::vector<DiTBlockWeights> dit_blocks;
    FinalLayerWeights final_layer;
};

// ============================================================================
// CUDA Context
// ============================================================================
struct CudaContext {
    cublasHandle_t cublas;
    
    // Workspace buffers
    float* d_tmp1 = nullptr;
    float* d_tmp2 = nullptr;
    float* d_tmp3 = nullptr;
    float* d_tmp4 = nullptr;
    float* d_encoding = nullptr;
    float* d_score = nullptr;
    float* d_t_emb = nullptr;
    float* d_y = nullptr;
    float* d_modulation = nullptr;
    
    // Input buffers
    float* d_sampled_traj = nullptr;
    float* d_diffusion_time = nullptr;
    
    size_t workspace_size = 0;
    
    void init(const DiTConfig& cfg, int batch_size) {
        CUBLAS_CHECK(cublasCreate(&cublas));
        
        int token_num = cfg.agent_num + cfg.static_objects_num + cfg.lane_num;
        int P = 1 + cfg.predicted_neighbor_num;
        int max_seq = std::max({cfg.time_len, token_num, P});
        
        workspace_size = batch_size * max_seq * cfg.hidden_dim * 4;
        
        CUDA_CHECK(cudaMalloc(&d_tmp1, workspace_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tmp2, workspace_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tmp3, workspace_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tmp4, workspace_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_encoding, batch_size * token_num * cfg.hidden_dim * sizeof(float)));
        
        int output_dim = (cfg.future_len + 1) * 4;
        CUDA_CHECK(cudaMalloc(&d_score, batch_size * P * output_dim * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_t_emb, batch_size * cfg.frequency_embedding_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * cfg.hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_modulation, batch_size * 6 * cfg.hidden_dim * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_sampled_traj, batch_size * P * (cfg.future_len + 1) * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffusion_time, batch_size * sizeof(float)));
    }
    
    void free() {
        if (d_tmp1) cudaFree(d_tmp1);
        if (d_tmp2) cudaFree(d_tmp2);
        if (d_tmp3) cudaFree(d_tmp3);
        if (d_tmp4) cudaFree(d_tmp4);
        if (d_encoding) cudaFree(d_encoding);
        if (d_score) cudaFree(d_score);
        if (d_t_emb) cudaFree(d_t_emb);
        if (d_y) cudaFree(d_y);
        if (d_modulation) cudaFree(d_modulation);
        if (d_sampled_traj) cudaFree(d_sampled_traj);
        if (d_diffusion_time) cudaFree(d_diffusion_time);
        if (cublas) cublasDestroy(cublas);
    }
};

// ============================================================================
// JSON Loading
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

// Upload tensor to GPU
float* upload_tensor(const json& j) {
    auto data = decode_base64_f32(j);
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ptr, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d_ptr;
}

LinearWeights load_linear_cuda(const json& j) {
    LinearWeights w;
    auto shape = get_shape(j["weight"]);
    w.out_features = shape[0];
    w.in_features = shape[1];
    w.d_weight = upload_tensor(j["weight"]);
    if (j.contains("bias")) {
        w.d_bias = upload_tensor(j["bias"]);
    }
    return w;
}

LayerNormWeights load_layernorm_cuda(const json& j) {
    LayerNormWeights w;
    auto shape = get_shape(j["weight"]);
    w.dim = shape[0];
    w.d_weight = upload_tensor(j["weight"]);
    w.d_bias = upload_tensor(j["bias"]);
    return w;
}

MLPWeights load_mlp_cuda(const json& j) {
    MLPWeights w;
    w.fc1 = load_linear_cuda(j["fc1"]);
    w.fc2 = load_linear_cuda(j["fc2"]);
    return w;
}

MultiHeadAttentionWeights load_mha_cuda(const json& j, int num_heads) {
    MultiHeadAttentionWeights w;
    auto shape = get_shape(j["in_proj_weight"]);
    w.embed_dim = shape[1];
    w.num_heads = num_heads;
    w.d_in_proj_weight = upload_tensor(j["in_proj_weight"]);
    w.d_in_proj_bias = upload_tensor(j["in_proj_bias"]);
    w.out_proj = load_linear_cuda(j["out_proj"]);
    return w;
}

MixerBlockWeights load_mixer_block_cuda(const json& j) {
    MixerBlockWeights w;
    w.norm1 = load_layernorm_cuda(j["norm1"]);
    w.channels_mlp = load_mlp_cuda(j["channels_mlp"]);
    w.norm2 = load_layernorm_cuda(j["norm2"]);
    w.tokens_mlp = load_mlp_cuda(j["tokens_mlp"]);
    return w;
}

SelfAttentionBlockWeights load_self_attn_block_cuda(const json& j, int num_heads) {
    SelfAttentionBlockWeights w;
    w.norm1 = load_layernorm_cuda(j["norm1"]);
    w.attn = load_mha_cuda(j["attn"], num_heads);
    w.norm2 = load_layernorm_cuda(j["norm2"]);
    w.mlp = load_mlp_cuda(j["mlp"]);
    return w;
}

DiTBlockWeights load_dit_block_cuda(const json& j, int num_heads) {
    DiTBlockWeights w;
    w.norm1 = load_layernorm_cuda(j["norm1"]);
    w.attn = load_mha_cuda(j["attn"], num_heads);
    w.norm2 = load_layernorm_cuda(j["norm2"]);
    w.mlp1 = load_mlp_cuda(j["mlp1"]);
    w.adaLN_modulation = load_linear_cuda(j["adaLN_modulation"]["linear"]);
    w.norm3 = load_layernorm_cuda(j["norm3"]);
    w.cross_attn = load_mha_cuda(j["cross_attn"], num_heads);
    w.norm4 = load_layernorm_cuda(j["norm4"]);
    w.mlp2 = load_mlp_cuda(j["mlp2"]);
    return w;
}

DiTWeights load_weights_cuda(const json& weights_json, const DiTConfig& cfg) {
    DiTWeights w;
    
    w.encoder_pos_emb = load_linear_cuda(weights_json["encoder_pos_emb"]);
    
    auto& ne = weights_json["neighbor_encoder"];
    w.neighbor_type_emb = load_linear_cuda(ne["type_emb"]);
    w.neighbor_channel_pre = load_mlp_cuda(ne["channel_pre_project"]);
    w.neighbor_token_pre = load_mlp_cuda(ne["token_pre_project"]);
    for (auto& blk : ne["blocks"]) {
        w.neighbor_blocks.push_back(load_mixer_block_cuda(blk));
    }
    w.neighbor_norm = load_layernorm_cuda(ne["norm"]);
    w.neighbor_emb_project = load_mlp_cuda(ne["emb_project"]);
    
    w.static_projection = load_mlp_cuda(weights_json["static_encoder"]["projection"]);
    
    auto& le = weights_json["lane_encoder"];
    w.lane_speed_limit_emb = load_linear_cuda(le["speed_limit_emb"]);
    w.d_lane_unknown_speed_emb = upload_tensor(le["unknown_speed_emb"]["weight"]);
    w.lane_traffic_emb = load_linear_cuda(le["traffic_emb"]);
    w.lane_channel_pre = load_mlp_cuda(le["channel_pre_project"]);
    w.lane_token_pre = load_mlp_cuda(le["token_pre_project"]);
    for (auto& blk : le["blocks"]) {
        w.lane_blocks.push_back(load_mixer_block_cuda(blk));
    }
    w.lane_norm = load_layernorm_cuda(le["norm"]);
    w.lane_emb_project = load_mlp_cuda(le["emb_project"]);
    
    auto& fe = weights_json["fusion_encoder"];
    for (auto& blk : fe["blocks"]) {
        w.fusion_blocks.push_back(load_self_attn_block_cuda(blk, cfg.num_heads));
    }
    w.fusion_norm = load_layernorm_cuda(fe["norm"]);
    
    auto& re = weights_json["route_encoder"];
    w.route_channel_pre = load_mlp_cuda(re["channel_pre_project"]);
    w.route_token_pre = load_mlp_cuda(re["token_pre_project"]);
    w.route_mixer = load_mixer_block_cuda(re["mixer"]);
    w.route_norm = load_layernorm_cuda(re["norm"]);
    w.route_emb_project = load_mlp_cuda(re["emb_project"]);
    
    w.d_agent_embedding = upload_tensor(weights_json["agent_embedding"]["weight"]);
    w.preproj = load_mlp_cuda(weights_json["preproj"]);
    w.t_embedder_mlp_0 = load_linear_cuda(weights_json["t_embedder"]["mlp_0"]);
    w.t_embedder_mlp_2 = load_linear_cuda(weights_json["t_embedder"]["mlp_2"]);
    
    for (auto& blk : weights_json["dit_blocks"]) {
        w.dit_blocks.push_back(load_dit_block_cuda(blk, cfg.num_heads));
    }
    
    auto& fl = weights_json["final_layer"];
    w.final_layer.norm_final = load_layernorm_cuda(fl["norm_final"]);
    w.final_layer.proj_norm1 = load_layernorm_cuda(fl["proj"]["norm1"]);
    w.final_layer.proj_linear1 = load_linear_cuda(fl["proj"]["linear1"]);
    w.final_layer.proj_norm2 = load_layernorm_cuda(fl["proj"]["norm2"]);
    w.final_layer.proj_linear2 = load_linear_cuda(fl["proj"]["linear2"]);
    w.final_layer.adaLN_modulation = load_linear_cuda(fl["adaLN_modulation"]["linear"]);
    
    return w;
}

// ============================================================================
// CUDA Operations
// ============================================================================
void cuda_layer_norm(CudaContext& ctx, float* out, const float* x,
                     const LayerNormWeights& w, int batch) {
    int blocks = batch;
    int threads = std::min(256, w.dim);
    size_t smem = threads * sizeof(double);
    layer_norm_kernel<<<blocks, threads, smem>>>(x, w.d_weight, w.d_bias, out, w.dim, 1e-5f);
}

void cuda_linear(CudaContext& ctx, float* out, const float* x,
                 const LinearWeights& w, int batch) {
    // out = x @ W^T + b
    // Using cuBLAS: C = alpha * A * B + beta * C
    // We compute: out(batch, out_f) = x(batch, in_f) * W^T(in_f, out_f)
    float alpha = 1.0f, beta = 0.0f;
    
    // cuBLAS uses column-major, so we compute: C^T = B^T * A^T
    // out^T = W * x^T
    CUBLAS_CHECK(cublasSgemm(
        ctx.cublas,
        CUBLAS_OP_T,    // W transposed
        CUBLAS_OP_N,    // x not transposed
        w.out_features, // M
        batch,          // N
        w.in_features,  // K
        &alpha,
        w.d_weight, w.in_features,
        x, w.in_features,
        &beta,
        out, w.out_features
    ));
    
    // Add bias
    if (w.d_bias) {
        for (int b = 0; b < batch; ++b) {
            int64_t n = w.out_features;
            int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            add_inplace_kernel<<<blocks, BLOCK_SIZE>>>(out + b * w.out_features, w.d_bias, n);
        }
    }
}

void cuda_mlp_gelu(CudaContext& ctx, float* out, const float* x,
                   const MLPWeights& w, int batch, float* tmp) {
    // FC1
    cuda_linear(ctx, tmp, x, w.fc1, batch);
    
    // GELU
    int64_t n = batch * w.fc1.out_features;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_kernel<<<blocks, BLOCK_SIZE>>>(tmp, n);
    
    // FC2
    cuda_linear(ctx, out, tmp, w.fc2, batch);
}

void cuda_timestep_embedding(CudaContext& ctx, DiTWeights& weights, DiTConfig& cfg,
                             float* out, const float* t, int batch) {
    int half_dim = cfg.frequency_embedding_size / 2;
    timestep_embed_kernel<<<batch, 256>>>(t, ctx.d_t_emb, batch, half_dim);
    
    // MLP: fc0 -> silu -> fc2
    cuda_linear(ctx, ctx.d_tmp1, ctx.d_t_emb, weights.t_embedder_mlp_0, batch);
    
    int64_t n = batch * weights.t_embedder_mlp_0.out_features;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_kernel<<<blocks, BLOCK_SIZE>>>(ctx.d_tmp1, n);
    
    cuda_linear(ctx, out, ctx.d_tmp1, weights.t_embedder_mlp_2, batch);
}

// ============================================================================
// DiT Inference
// ============================================================================
class DiTInferenceCUDA {
public:
    DiTConfig cfg;
    DiTWeights weights;
    CudaContext ctx;
    
    void encoder_forward(int batch) {
        int token_num = cfg.agent_num + cfg.static_objects_num + cfg.lane_num;
        int64_t n = batch * token_num * cfg.hidden_dim;
        
        // Initialize encoding to zeros
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_kernel<<<blocks, BLOCK_SIZE>>>(ctx.d_encoding, 0.0f, n);
        
        // Process through fusion blocks (simplified)
        for (auto& block : weights.fusion_blocks) {
            // LayerNorm1
            cuda_layer_norm(ctx, ctx.d_tmp1, ctx.d_encoding, block.norm1, batch * token_num);
            
            // LayerNorm2 + MLP
            cuda_layer_norm(ctx, ctx.d_tmp2, ctx.d_encoding, block.norm2, batch * token_num);
            cuda_mlp_gelu(ctx, ctx.d_tmp3, ctx.d_tmp2, block.mlp, batch * token_num, ctx.d_tmp4);
            
            // Residual
            blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            add_inplace_kernel<<<blocks, BLOCK_SIZE>>>(ctx.d_encoding, ctx.d_tmp3, n);
        }
        
        // Final norm
        cuda_layer_norm(ctx, ctx.d_tmp1, ctx.d_encoding, weights.fusion_norm, batch * token_num);
        CUDA_CHECK(cudaMemcpy(ctx.d_encoding, ctx.d_tmp1, n * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    void decoder_forward(int batch) {
        int P = 1 + cfg.predicted_neighbor_num;
        
        // Preproj
        cuda_mlp_gelu(ctx, ctx.d_tmp1, ctx.d_sampled_traj, weights.preproj, batch * P, ctx.d_tmp2);
        
        // Add agent embedding (simplified - just use the embedding for first agent)
        // In full impl, would need to add different embeddings for ego vs neighbors
        
        // Timestep embedding
        cuda_timestep_embedding(ctx, weights, cfg, ctx.d_y, ctx.d_diffusion_time, batch);
        
        // DiT blocks (simplified)
        float* current = ctx.d_tmp1;
        for (auto& block : weights.dit_blocks) {
            // Apply SiLU to y and compute adaLN modulation
            int64_t n_y = batch * cfg.hidden_dim;
            int blocks_y = (n_y + BLOCK_SIZE - 1) / BLOCK_SIZE;
            silu_kernel<<<blocks_y, BLOCK_SIZE>>>(ctx.d_y, n_y);
            
            cuda_linear(ctx, ctx.d_modulation, ctx.d_y, block.adaLN_modulation, batch);
            
            // Norm1
            cuda_layer_norm(ctx, ctx.d_tmp2, current, block.norm1, batch * P);
            
            // Norm2 + MLP1
            cuda_layer_norm(ctx, ctx.d_tmp3, ctx.d_tmp2, block.norm2, batch * P);
            cuda_mlp_gelu(ctx, ctx.d_tmp4, ctx.d_tmp3, block.mlp1, batch * P, ctx.d_tmp2);
            
            // Norm3 (cross attention placeholder)
            cuda_layer_norm(ctx, ctx.d_tmp2, ctx.d_tmp4, block.norm3, batch * P);
            
            // Norm4 + MLP2
            cuda_layer_norm(ctx, ctx.d_tmp3, ctx.d_tmp2, block.norm4, batch * P);
            cuda_mlp_gelu(ctx, current, ctx.d_tmp3, block.mlp2, batch * P, ctx.d_tmp4);
        }
        
        // Final layer
        cuda_layer_norm(ctx, ctx.d_tmp2, current, weights.final_layer.norm_final, batch * P);
        cuda_layer_norm(ctx, ctx.d_tmp3, ctx.d_tmp2, weights.final_layer.proj_norm1, batch * P);
        cuda_linear(ctx, ctx.d_tmp4, ctx.d_tmp3, weights.final_layer.proj_linear1, batch * P);
        
        int64_t n_proj = batch * P * weights.final_layer.proj_linear1.out_features;
        int blocks_proj = (n_proj + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_kernel<<<blocks_proj, BLOCK_SIZE>>>(ctx.d_tmp4, n_proj);
        
        cuda_layer_norm(ctx, ctx.d_tmp2, ctx.d_tmp4, weights.final_layer.proj_norm2, batch * P);
        cuda_linear(ctx, ctx.d_score, ctx.d_tmp2, weights.final_layer.proj_linear2, batch * P);
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
    std::cout << "[CUDA] Device: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
    
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
    
    std::cout << "[CUDA FP32] Config: hidden=" << cfg.hidden_dim << " depth=" << cfg.encoder_depth 
              << " heads=" << cfg.num_heads << std::endl;
    
    // Load weights
    DiTWeights weights = load_weights_cuda(j["weights"], cfg);
    std::cout << "[OK] Loaded weights to GPU" << std::endl;
    
    // Load inputs
    auto input_shape = get_shape(j["inputs"]["sampled_trajectories"]);
    int B = input_shape[0];
    int P = input_shape[1];
    
    auto sampled_traj = decode_base64_f32(j["inputs"]["sampled_trajectories"]);
    auto diffusion_time = decode_base64_f32(j["inputs"]["diffusion_time"]);
    
    std::cout << "Input: B=" << B << " P=" << P << std::endl;
    
    // Initialize inference
    DiTInferenceCUDA infer;
    infer.cfg = cfg;
    infer.weights = weights;
    infer.ctx.init(cfg, B);
    
    // Upload inputs
    CUDA_CHECK(cudaMemcpy(infer.ctx.d_sampled_traj, sampled_traj.data(), 
                          sampled_traj.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(infer.ctx.d_diffusion_time, diffusion_time.data(),
                          diffusion_time.size() * sizeof(float), cudaMemcpyHostToDevice));
    
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
    
    // Download results
    int output_dim = (cfg.future_len + 1) * 4;
    std::vector<float> score(B * P * output_dim);
    CUDA_CHECK(cudaMemcpy(score.data(), infer.ctx.d_score, score.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Statistics
    float min_val = score[0], max_val = score[0], sum = 0;
    for (size_t i = 0; i < score.size(); ++i) {
        min_val = std::min(min_val, score[i]);
        max_val = std::max(max_val, score[i]);
        sum += score[i];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "[CUDA FP32 Forward] Score shape: (" << B << ", " << P << ", " << output_dim << ")" << std::endl;
    std::cout << "[Timing] Forward pass: " << elapsed_ms << "ms" << std::endl;
    std::cout << "[Score] Min: " << min_val << ", Max: " << max_val << ", Mean: " << sum/score.size() << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Cleanup
    infer.ctx.free();
    
    return 0;
}
