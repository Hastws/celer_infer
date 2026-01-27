// backend_config.hpp - Backend configuration and feature flags
// ============================================================================
// Defines optimization features that can be combined:
//   - SIMD: AVX2 vectorization
//   - FUSION: Operator fusion (RMSNorm+Linear, SiLU*Up, etc.)
//   - QUANT: INT8 quantization
//   - CUDA: GPU acceleration
// ============================================================================

#ifndef BACKEND_CONFIG_HPP
#define BACKEND_CONFIG_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace celer_infer {

// Feature flags (can be OR'd together)
enum BackendFeature : uint32_t {
  FEATURE_NONE     = 0,
  FEATURE_SIMD     = 1 << 0,  // AVX2 SIMD optimization
  FEATURE_FUSION   = 1 << 1,  // Operator fusion
  FEATURE_QUANT    = 1 << 2,  // INT8 quantization
  FEATURE_CUDA     = 1 << 3,  // GPU acceleration
};

// Backend configuration
struct BackendConfig {
  uint32_t features = FEATURE_NONE;
  
  // Quantization settings
  int quant_bits = 8;           // Quantization bit width
  bool symmetric_quant = true;  // Symmetric vs asymmetric quantization
  
  // CUDA settings
  int cuda_device = 0;          // GPU device ID
  bool cuda_use_tensor_cores = true;  // Use Tensor Cores if available
  
  // Helper methods
  bool has_simd() const { return features & FEATURE_SIMD; }
  bool has_fusion() const { return features & FEATURE_FUSION; }
  bool has_quant() const { return features & FEATURE_QUANT; }
  bool has_cuda() const { return features & FEATURE_CUDA; }
  
  // Get feature string for logging
  std::string feature_string() const {
    std::string s;
    if (features == FEATURE_NONE) return "baseline";
    if (has_simd()) s += "simd+";
    if (has_fusion()) s += "fusion+";
    if (has_quant()) s += "quant+";
    if (has_cuda()) s += "cuda+";
    if (!s.empty() && s.back() == '+') s.pop_back();
    return s;
  }
  
  // Parse from string like "simd+fusion"
  static BackendConfig from_string(const std::string& s) {
    BackendConfig cfg;
    if (s.find("simd") != std::string::npos) cfg.features |= FEATURE_SIMD;
    if (s.find("fusion") != std::string::npos) cfg.features |= FEATURE_FUSION;
    if (s.find("quant") != std::string::npos) cfg.features |= FEATURE_QUANT;
    if (s.find("cuda") != std::string::npos) cfg.features |= FEATURE_CUDA;
    return cfg;
  }
};

// All valid backend combinations for benchmarking
inline std::vector<BackendConfig> get_all_backend_configs() {
  std::vector<BackendConfig> configs;
  
  // CPU backends (all combinations of simd, fusion, quant)
  for (uint32_t i = 0; i < 8; ++i) {
    BackendConfig cfg;
    cfg.features = 0;
    if (i & 1) cfg.features |= FEATURE_SIMD;
    if (i & 2) cfg.features |= FEATURE_FUSION;
    if (i & 4) cfg.features |= FEATURE_QUANT;
    configs.push_back(cfg);
  }
  
  // CUDA backends (with/without fusion, quant)
  for (uint32_t i = 0; i < 4; ++i) {
    BackendConfig cfg;
    cfg.features = FEATURE_CUDA;
    if (i & 1) cfg.features |= FEATURE_FUSION;
    if (i & 2) cfg.features |= FEATURE_QUANT;
    configs.push_back(cfg);
  }
  
  return configs;
}

}  // namespace celer_infer

#endif  // BACKEND_CONFIG_HPP
