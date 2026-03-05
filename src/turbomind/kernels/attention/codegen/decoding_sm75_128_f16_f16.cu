// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm75, half, half, 8, 128>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm75, half, half, 16, 128>>(const AttentionParams<half>& params);

}  // namespace turbomind

// ---- EXTRA INSTANTIATIONS: Sm75 + f16 KV + head_dim=256 (SIMT fallback) ----
#include <cuda_fp16.h>
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

// The missing symbol you saw corresponds to: Sm75 + T=__half + Tkv=__half + Qh=2 + HeadDim=256
using K256_Q2 = typename DecodingConfig<arch::Sm75, __half, __half, 2, 256>::Kernel;
template bool invokeDecoding<K256_Q2>(const typename K256_Q2::ParamType&);

// Optional: also instantiate a few nearby Qh values for safety
using K256_Q4 = typename DecodingConfig<arch::Sm75, __half, __half, 4, 256>::Kernel;
using K256_Q8 = typename DecodingConfig<arch::Sm75, __half, __half, 8, 256>::Kernel;
template bool invokeDecoding<K256_Q4>(const typename K256_Q4::ParamType&);
template bool invokeDecoding<K256_Q8>(const typename K256_Q8::ParamType&);

}  // namespace turbomind
// ---- END EXTRA INSTANTIATIONS ----


// ---- FIX UNDEFINED: Sm75 + u8 KV(unsigned char) + head_dim=256 + Qh=6 (SIMT fallback) ----
#include <cuda_fp16.h>
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

// This matches the failing import symbol: T=__half, Tkv=unsigned char('h'), Qh=6, HeadDim=256
using K256_Q6_U8 = typename DecodingConfig<arch::Sm75, __half, unsigned char, 6, 256>::Kernel;
template bool invokeDecoding<K256_Q6_U8>(const typename K256_Q6_U8::ParamType&);

}  // namespace turbomind
// ---- END FIX ----


// ---- FIX UNDEFINED: Sm75 + f16 KV + head_dim=256 missing Qh (SIMT fallback) ----
#include <cuda_fp16.h>
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

// Missing ones you listed: __half KV with Qh=3/5/6/7 (do NOT duplicate Qh=2/4/8 if you already instantiated them earlier)
using K256_Q3_F16 = typename DecodingConfig<arch::Sm75, __half, __half, 3, 256>::Kernel;
using K256_Q5_F16 = typename DecodingConfig<arch::Sm75, __half, __half, 5, 256>::Kernel;
using K256_Q6_F16 = typename DecodingConfig<arch::Sm75, __half, __half, 6, 256>::Kernel;
using K256_Q7_F16 = typename DecodingConfig<arch::Sm75, __half, __half, 7, 256>::Kernel;

template bool invokeDecoding<K256_Q3_F16>(const typename K256_Q3_F16::ParamType&);
template bool invokeDecoding<K256_Q5_F16>(const typename K256_Q5_F16::ParamType&);
template bool invokeDecoding<K256_Q6_F16>(const typename K256_Q6_F16::ParamType&);
template bool invokeDecoding<K256_Q7_F16>(const typename K256_Q7_F16::ParamType&);

// Missing ones you listed: unsigned char KV with Qh=2/3/4/5/7/8 (you already instantiated Qh=6 earlier)
using K256_Q2_U8 = typename DecodingConfig<arch::Sm75, __half, unsigned char, 2, 256>::Kernel;
using K256_Q3_U8 = typename DecodingConfig<arch::Sm75, __half, unsigned char, 3, 256>::Kernel;
using K256_Q4_U8 = typename DecodingConfig<arch::Sm75, __half, unsigned char, 4, 256>::Kernel;
using K256_Q5_U8 = typename DecodingConfig<arch::Sm75, __half, unsigned char, 5, 256>::Kernel;
using K256_Q7_U8 = typename DecodingConfig<arch::Sm75, __half, unsigned char, 7, 256>::Kernel;
using K256_Q8_U8 = typename DecodingConfig<arch::Sm75, __half, unsigned char, 8, 256>::Kernel;

template bool invokeDecoding<K256_Q2_U8>(const typename K256_Q2_U8::ParamType&);
template bool invokeDecoding<K256_Q3_U8>(const typename K256_Q3_U8::ParamType&);
template bool invokeDecoding<K256_Q4_U8>(const typename K256_Q4_U8::ParamType&);
template bool invokeDecoding<K256_Q5_U8>(const typename K256_Q5_U8::ParamType&);
template bool invokeDecoding<K256_Q7_U8>(const typename K256_Q7_U8::ParamType&);
template bool invokeDecoding<K256_Q8_U8>(const typename K256_Q8_U8::ParamType&);

}  // namespace turbomind
// ---- END FIX ----

