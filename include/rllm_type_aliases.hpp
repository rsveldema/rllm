#pragma once

#if USE_VULKAN_BACKEND
#include <_type_aliases.hpp>
using rlmm_float_small = kfloat;    
#else
using rlmm_float_small = _Float16;  
#endif




using rlmm_float = float;
static constexpr rlmm_float RLMM_ZERO = rlmm_float{0};
static constexpr rlmm_float RLMM_ONE = rlmm_float{1};
static constexpr rlmm_float RLMM_NEG_ONE = rlmm_float{-1};
