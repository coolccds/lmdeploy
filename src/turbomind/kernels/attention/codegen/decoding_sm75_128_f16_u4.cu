// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm75, half, uint4_t, 8, 128>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm75, half, uint4_t, 16, 128>>(const AttentionParams<half>& params);

}  // namespace turbomind

// ---- EXTRA INSTANTIATIONS: Sm75 + u4 + head_dim=256 (SIMT fallback) ----
#include <cuda_fp16.h>
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

// Try a few Qh values to avoid future undefined symbols.
// (Your missing symbol is Qh=8, but add 4/16 as well.)
using K256_Q4  = typename DecodingConfig<arch::Sm75, __half, uint4_t, 4, 256>::Kernel;
using K256_Q8  = typename DecodingConfig<arch::Sm75, __half, uint4_t, 8, 256>::Kernel;
using K256_Q16 = typename DecodingConfig<arch::Sm75, __half, uint4_t, 16, 256>::Kernel;

template bool invokeDecoding<K256_Q4>(const typename K256_Q4::ParamType&);
template bool invokeDecoding<K256_Q8>(const typename K256_Q8::ParamType&);
template bool invokeDecoding<K256_Q16>(const typename K256_Q16::ParamType&);

}  // namespace turbomind
// ---- END EXTRA INSTANTIATIONS ----


// ---- EXTRA INSTANTIATIONS: Sm75 + u4 KV + head_dim=256 (SIMT fallback) ----
#include <cuda_fp16.h>
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {
using K256_Q2_U4 = typename DecodingConfig<arch::Sm75, __half, uint4_t, 2, 256>::Kernel;
template bool invokeDecoding<K256_Q2_U4>(const typename K256_Q2_U4::ParamType&);
}  // namespace turbomind
// ---- END EXTRA INSTANTIATIONS ----


// ---- FIX UNDEFINED: Sm75 + u4 KV + head_dim=256 + Qh=3/5/6 (SIMT fallback) ----
#include <cuda_fp16.h>
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using K256_Q3_U4 = typename DecodingConfig<arch::Sm75, __half, uint4_t, 3, 256>::Kernel;
using K256_Q5_U4 = typename DecodingConfig<arch::Sm75, __half, uint4_t, 5, 256>::Kernel;
using K256_Q6_U4 = typename DecodingConfig<arch::Sm75, __half, uint4_t, 6, 256>::Kernel;

template bool invokeDecoding<K256_Q3_U4>(const typename K256_Q3_U4::ParamType&);
template bool invokeDecoding<K256_Q5_U4>(const typename K256_Q5_U4::ParamType&);
template bool invokeDecoding<K256_Q6_U4>(const typename K256_Q6_U4::ParamType&);

}  // namespace turbomind
// ---- END FIX ----


// ---- FIX UNDEFINED: Sm75 + u4 KV + head_dim=256 + Qh=7 (SIMT fallback) ----
#include <cuda_fp16.h>
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using K256_Q7_U4 = typename DecodingConfig<arch::Sm75, __half, uint4_t, 7, 256>::Kernel;
template bool invokeDecoding<K256_Q7_U4>(const typename K256_Q7_U4::ParamType&);

}  // namespace turbomind
// ---- END FIX ----

