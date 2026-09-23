# CPU kernels

Preserve w * conj(x), packed BF16 complex values, and scalar/non-ARM fallbacks.
Legacy runtime toggles: GGML_IFAIRY_LUT, GGML_IFAIRY_LUT_DEBUG,
GGML_IFAIRY_LUT_IMPL=auto|lut16|lut_c, GGML_IFAIRY_VEC_DOT_ACT_TENSOR.
No kernel tuning is part of this extraction. Validate direct and LUT builds.
