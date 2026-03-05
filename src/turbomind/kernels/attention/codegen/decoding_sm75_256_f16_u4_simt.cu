// Explicit instantiation to satisfy linker symbols for sm75 head_dim=256 SIMT u4 path

#include <cuda_fp16.h>

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

// This must match the mangled symbol you saw:
// Sm75 + T=__half + Tkv=uint4_t + Qh=8 + HeadDim=256
using Kernel = typename DecodingConfig<arch::Sm75, __half, uint4_t, 8, 256>::Kernel;

// Force template instantiation into the final binary
template bool invokeDecoding<Kernel>(const typename Kernel::ParamType&);

}  // namespace turbomind
