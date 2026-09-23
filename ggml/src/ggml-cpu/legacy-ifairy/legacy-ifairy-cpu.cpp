#include "legacy-ifairy-cpu.h"

#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "quants.h"
#include "wide-linear.h"

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
#    include "ggml-ifairy-lut-impl.h"
#    include "ggml-ifairy-lut.h"

#    include <limits.h>

#    include <algorithm>
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static constexpr size_t GGML_LEGACY_IFAIRY_CPU_CACHE_LINE = 64;

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
enum ggml_legacy_ifairy_lut_impl {
    GGML_LEGACY_IFAIRY_LUT_IMPL_AUTO  = 0,
    GGML_LEGACY_IFAIRY_LUT_IMPL_LUT16 = 1,
    GGML_LEGACY_IFAIRY_LUT_IMPL_LUT_C = 2,
};

struct ggml_legacy_ifairy_lut_config {
    bool dbg;
    bool lut_enabled;
    enum ggml_legacy_ifairy_lut_impl impl;
};

static enum ggml_legacy_ifairy_lut_impl ggml_legacy_ifairy_lut_impl_from_env(const char * env_name, bool dbg) {
    enum ggml_legacy_ifairy_lut_impl impl = GGML_LEGACY_IFAIRY_LUT_IMPL_AUTO;
    const char * impl_env                 = getenv(env_name);
    if (impl_env && impl_env[0] != '\0' && strcmp(impl_env, "0") != 0 && strcmp(impl_env, "auto") != 0) {
        if (strcmp(impl_env, "lut16") == 0) {
            impl = GGML_LEGACY_IFAIRY_LUT_IMPL_LUT16;
        } else if (strcmp(impl_env, "lut_c") == 0) {
            impl = GGML_LEGACY_IFAIRY_LUT_IMPL_LUT_C;
        } else if (dbg) {
            GGML_LOG_WARN("legacy_ifairy_lut: unknown %s=%s (expected auto|lut16|lut_c)\n", env_name, impl_env);
        }
    }
    return impl;
}

static struct ggml_legacy_ifairy_lut_config ggml_legacy_ifairy_lut_config_from_env(void) {
    struct ggml_legacy_ifairy_lut_config cfg;

    cfg.dbg         = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG");
    cfg.lut_enabled = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT");
    cfg.impl        = ggml_legacy_ifairy_lut_impl_from_env("GGML_IFAIRY_LUT_IMPL", cfg.dbg);

    return cfg;
}

static bool ggml_legacy_ifairy_cpu_can_mul_mat(const struct ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_MUL_MAT || !ggml_ifairy_env_enabled("GGML_IFAIRY_LUT")) {
        return false;
    }
    const struct ggml_tensor * w = dst->src[0];
    const struct ggml_tensor * x = dst->src[1];
    // The weight packer handles one contiguous 2D matrix. Keep broadcasts,
    // views with gaps in the weights, and IFAIRY64 on their existing paths.
    if (!w || !x || w->type != GGML_TYPE_IFAIRY || w->op != GGML_OP_NONE || !ggml_is_matrix(w) || !ggml_is_matrix(x) ||
        !ggml_is_matrix(dst) || !ggml_is_contiguous(w) || x->nb[0] != ggml_type_size(x->type) ||
        dst->nb[0] != sizeof(float) || w->ne[0] <= 0 || w->ne[0] > INT_MAX || w->ne[1] <= 0 || w->ne[1] > INT_MAX ||
        x->ne[1] <= 0 || x->ne[1] > INT_MAX) {
        return false;
    }
    return ggml_ifairy_lut_can_mul_mat(w, x, dst);
}

static bool ggml_legacy_ifairy_cpu_compute_mul_mat(const struct ggml_compute_params *    params,
                                                   struct ggml_tensor *                  dst,
                                                   const ggml_legacy_ifairy_lut_config & cfg) {
    const struct ggml_tensor * w     = dst->src[0];
    const struct ggml_tensor * x     = dst->src[1];
    const auto *               extra = static_cast<const ifairy_lut_extra *>(w->extra);
    const size_t               need  = ggml_ifairy_lut_get_wsize(w, x, dst, params->nth);
    if (!extra || !extra->packed_w || !params->wdata || params->wsize < need) {
        return false;
    }

    const int    m          = (int) w->ne[1];
    const int    k          = (int) w->ne[0];
    const int    n          = (int) x->ne[1];
    const size_t blocks     = k / QK_IFAIRY;
    const size_t q_stride   = ggml_row_size(GGML_TYPE_IFAIRY_Q16, k);
    const size_t q_bytes    = x->type == GGML_TYPE_F32 ? GGML_PAD((size_t) n * q_stride, 64) : 0;
    const size_t lut_bytes  = (size_t) n * blocks * QK_IFAIRY_GROUPS_PER_BLOCK * k_ifairy_lut_group_bytes;
    auto *       lut        = static_cast<uint8_t *>(params->wdata) + q_bytes;
    auto *       scales     = reinterpret_cast<float *>(lut + lut_bytes);
    const void * act        = x->data;
    size_t       act_stride = x->nb[1];
    const bool   lut_c      = cfg.impl == GGML_LEGACY_IFAIRY_LUT_IMPL_LUT_C && x->type == GGML_TYPE_F32;

    if (x->type == GGML_TYPE_F32) {
        // A table entry sums two signed activations: native x86 Q8 (127)
        // would saturate int8 LUT entries. Use the existing LUT-safe Q8
        // quantizers: tensor scale for auto/lut16, block scale for lut_c.
        for (int col = params->ith; col < n; col += params->nth) {
            const auto * row   = reinterpret_cast<const float *>(static_cast<const char *>(x->data) + col * x->nb[1]);
            void *       q_row = static_cast<char *>(params->wdata) + col * q_stride;
            if (lut_c) {
                quantize_row_ifairy_q16_lut_c(row, q_row, k);
            } else {
                quantize_row_ifairy_q16_tensor(row, q_row, k);
            }
        }
        act        = params->wdata;
        act_stride = q_stride;
        ggml_barrier(params->threadpool);
    } else {
        // External Q16 tensors may use the full int8 range. All workers
        // make the same decision before any barrier; fall back without
        // clipping or changing the caller's activation quantization.
        for (int col = 0; col < n; ++col) {
            const auto * row =
                reinterpret_cast<const block_ifairy_q16 *>(static_cast<const char *>(x->data) + col * x->nb[1]);
            for (size_t b = 0; b < blocks; ++b) {
                for (int j = 0; j < QK_IFAIRY; ++j) {
                    const int r = row[b].x_real[j];
                    const int i = row[b].x_imag[j];
                    if ((r > 63 && r < 193) || (i > 63 && i < 193)) {
                        return false;
                    }
                }
            }
        }
    }

    ggml_ifairy_lut_preprocess_ex_lut16(m, k, n, act, act_stride, scales, lut, params->ith, params->nth);
    ggml_barrier(params->threadpool);

    const int64_t tiles = ((int64_t) m + 15) / 16;
    const int64_t tile0 = tiles * params->ith / params->nth;
    const int64_t tile1 = tiles * (params->ith + 1) / params->nth;
    const int64_t row0  = tile0 * 16;
    const int     rows  = (int) (std::min<int64_t>(tile1 * 16, m) - row0);
    if (rows > 0) {
        const auto * packed = static_cast<const ifairy_lut_wtile_16 *>(extra->packed_w) + tile0 * blocks;
        auto *       out    = reinterpret_cast<float *>(static_cast<char *>(dst->data) + row0 * dst->nb[0]);
        ggml_ifairy_lut_qgemm_lut16(rows, k, n, packed, lut, scales, out, dst->nb[1], dst->nb[0], true, false);
    }
    if (params->ith == 0 && cfg.dbg) {
        GGML_LOG_INFO("ifairy_lut: executed MUL_MAT %s M=%d N=%d K=%d threads=%d quant=%s\n", dst->name, m, n, k,
                      params->nth, x->type != GGML_TYPE_F32 ? "prequantized" : (lut_c ? "block42.6" : "tensor42.6"));
    }
    return true;
}
#endif

bool ggml_legacy_ifairy_cpu_supports_op(const struct ggml_tensor * dst) {
#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    if (ggml_legacy_ifairy_cpu_can_mul_mat(dst)) {
        return true;
    }
#endif
    return dst != nullptr && dst->op == GGML_OP_IFAIRY_WIDE_LINEAR_W2;
}

int ggml_legacy_ifairy_cpu_n_tasks(const struct ggml_tensor * dst, int n_threads) {
    if (!ggml_legacy_ifairy_cpu_supports_op(dst)) {
        return 0;
    }
    return n_threads;
}

size_t ggml_legacy_ifairy_cpu_work_size(const struct ggml_tensor * dst, int n_tasks) {
    GGML_UNUSED(n_tasks);

    if (!ggml_legacy_ifairy_cpu_supports_op(dst)) {
        return 0;
    }

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    if (dst->op == GGML_OP_MUL_MAT) {
        // This includes the direct path's quantization buffer, so an
        // unavailable weight pack can safely fall back during execution.
        return ggml_ifairy_lut_get_wsize(dst->src[0], dst->src[1], dst, n_tasks);
    }
#endif

    const struct ggml_tensor * x = dst->src[0];
    GGML_ASSERT(x && x->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] % ggml_blck_size(GGML_TYPE_IFAIRY64) == 0);

    const size_t q_row_size = ggml_row_size(GGML_TYPE_IFAIRY64_Q16, x->ne[0]);
    const size_t q_bytes    = GGML_PAD((size_t) ggml_nrows(x) * q_row_size, GGML_LEGACY_IFAIRY_CPU_CACHE_LINE);

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    const size_t lut_bytes = ggml_ifairy_wide_linear_w2_lut_wsize(dst);
    return q_bytes > lut_bytes ? q_bytes : lut_bytes;
#else
    return q_bytes;
#endif
}

void ggml_legacy_ifairy_cpu_prepare_graph(const struct ggml_cgraph * cgraph) {
#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    const struct ggml_legacy_ifairy_lut_config cfg = ggml_legacy_ifairy_lut_config_from_env();
    if (!cfg.lut_enabled) {
        return;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (!node) {
            continue;
        }

        if (node->op == GGML_OP_IFAIRY_WIDE_LINEAR_W2) {
            for (int src = 1; src <= 4; ++src) {
                struct ggml_tensor * weight = node->src[src];
                const struct ifairy_lut_extra * extra = weight ? (const struct ifairy_lut_extra *) weight->extra : nullptr;
                if (weight && (!extra || !extra->packed_w)) {
                    ggml_ifairy_lut_transform_tensor(weight, nullptr);
                }
            }
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        struct ggml_tensor * src0 = node->src[0];
        if (!ggml_legacy_ifairy_cpu_can_mul_mat(node)) {
            continue;
        }

        ggml_ifairy_lut_transform_tensor(src0, nullptr);
    }
#else
    GGML_UNUSED(cgraph);
#endif
}

bool ggml_legacy_ifairy_cpu_try_quantize_mul_mat_src1(
    const struct ggml_compute_params * params,
    const struct ggml_tensor *         src0,
    const struct ggml_tensor *         src1,
    enum ggml_type                     vec_dot_type,
    char *                             wdata,
    size_t                             nbw1,
    size_t                             nbw2,
    size_t                             nbw3) {
    if (!params || !src0 || !src1 || !wdata) {
        return false;
    }
    if (src0->type != GGML_TYPE_IFAIRY || vec_dot_type != GGML_TYPE_IFAIRY_Q16 || src1->type != GGML_TYPE_F32) {
        return false;
    }

    const char * env = getenv("GGML_IFAIRY_VEC_DOT_ACT_TENSOR");
    if (!env || strcmp(env, "0") == 0) {
        return false;
    }

    if (params->ith != 0) {
        return true;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                quantize_row_ifairy_q16_tensor((float *) ((char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11),
                                               (void *) (wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1), ne10);
            }
        }
    }

    return true;
}

static bool ggml_legacy_ifairy_cpu_compute_wide_linear_w2(
    const struct ggml_compute_params * params,
    struct ggml_tensor *                dst,
    bool                                use_lut,
    bool                                lut_c) {
    if (!dst || dst->op != GGML_OP_IFAIRY_WIDE_LINEAR_W2) {
        return false;
    }

#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    if (use_lut && ggml_compute_forward_ifairy_wide_linear_w2_lut(params, dst, lut_c)) {
        return true;
    }
#else
    GGML_UNUSED(use_lut);
    GGML_UNUSED(lut_c);
#endif

    ggml_compute_forward_ifairy_wide_linear_w2(params, dst);
    return true;
}

bool ggml_legacy_ifairy_cpu_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
#ifdef GGML_USE_LEGACY_IFAIRY_CPU_LUT
    const struct ggml_legacy_ifairy_lut_config cfg = ggml_legacy_ifairy_lut_config_from_env();
    const bool use_lut = cfg.lut_enabled;
    const bool lut_c   = cfg.impl == GGML_LEGACY_IFAIRY_LUT_IMPL_LUT_C;
    if (ggml_legacy_ifairy_cpu_can_mul_mat(dst)) {
        return ggml_legacy_ifairy_cpu_compute_mul_mat(params, dst, cfg);
    }
#else
    const bool use_lut = false;
    const bool lut_c   = false;
#endif
    return ggml_legacy_ifairy_cpu_compute_wide_linear_w2(params, dst, use_lut, lut_c);
}
