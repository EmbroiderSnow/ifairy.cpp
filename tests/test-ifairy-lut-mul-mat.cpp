// Exercise the graph dispatch, not just the LUT primitives.
#include "../ggml/src/ggml-cpu/legacy-ifairy/legacy-ifairy-cpu.h"
#include "../ggml/src/ggml-cpu/legacy-ifairy/lut/ggml-ifairy-lut.h"
#include "../ggml/src/ggml-cpu/quants.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

static std::atomic<int> executions{ 0 };

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static void log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (strstr(text, "ifairy_lut: executed MUL_MAT")) {
        ++executions;
    }
}

struct context {
    ggml_context * ptr = ggml_init({ 32 * 1024 * 1024, nullptr, false });

    ~context() {
        ggml_ifairy_lut_free();
        ggml_free(ptr);
    }
};

static float packed(float real, float imag) {
    const ggml_bf16_t pair[] = { ggml_fp32_to_bf16(real), ggml_fp32_to_bf16(imag) };
    float             result;
    memcpy(&result, pair, sizeof(result));
    return result;
}

static void run_case(int          m,
                     int          n,
                     int          k,
                     int          threads,
                     const char * impl,
                     bool         f32,
                     bool         unsafe          = false,
                     bool         strided_weights = false,
                     bool         strided_act     = false,
                     int          batches         = 1,
                     const char * enabled         = "1") {
    env("GGML_IFAIRY_LUT", enabled);
    env("GGML_IFAIRY_LUT_IMPL", impl);
    env("GGML_IFAIRY_LUT_DEBUG", "1");
    env("GGML_IFAIRY_VEC_DOT_ACT_TENSOR", nullptr);
    context ctx;
    require(ctx.ptr != nullptr, "context allocation failed");
    auto *     w_base   = ggml_new_tensor_2d(ctx.ptr, GGML_TYPE_IFAIRY, k, m * (strided_weights ? 2 : 1));
    auto *     w        = strided_weights ? ggml_view_2d(ctx.ptr, w_base, k, m, 2 * w_base->nb[1], 0) : w_base;
    const auto act_type = f32 ? GGML_TYPE_F32 : GGML_TYPE_IFAIRY_Q16;
    auto *     x_base   = ggml_new_tensor_3d(ctx.ptr, act_type, k, n * (strided_act ? 2 : 1), batches);
    auto *     x        = strided_act ? ggml_view_2d(ctx.ptr, x_base, k, n, 2 * x_base->nb[1], 0) : x_base;
    auto *     out      = ggml_mul_mat(ctx.ptr, w, x);
    auto *     graph    = ggml_new_graph(ctx.ptr);
    ggml_build_forward_expand(graph, out);
    const int                 blocks = k / QK_IFAIRY;
    std::vector<block_ifairy> weights((size_t) m * blocks);
    for (int row = 0; row < m; ++row) {
        for (int b = 0; b < blocks; ++b) {
            auto & wb = weights[(size_t) row * blocks + b];
            wb.d_real = ggml_fp32_to_fp16((1 + row % 5) / 32.0f);
            // Legacy direct vecdot uses one weight scale per row.
            wb.d_imag = ggml_fp32_to_fp16((1 + row % 3) / 16.0f);
            for (size_t j = 0; j < sizeof(wb.qs); ++j) {
                wb.qs[j] = (uint8_t) ((row * 73 + b * 37 + j * 19) & 255);
            }
        }
        memcpy((char *) w->data + row * w->nb[1], weights.data() + (size_t) row * blocks, w_base->nb[1]);
    }

    // Reuse the same graph with new activations to catch stale LUTs.
    for (int iteration = 0; iteration < 2; ++iteration) {
        std::vector<block_ifairy_q16> quantized((size_t) n * batches * blocks);
        std::vector<float>            input(k);
        for (int col = 0; col < n * batches; ++col) {
            for (int j = 0; j < k; ++j) {
                const float scale = 1.0f + static_cast<float>(j / QK_IFAIRY) * 0.5f;
                input[j]          = packed(scale * ((j * 17 + col * 11 + iteration * 7) % 61 - 30) / 32.0f,
                                           scale * ((j * 13 + col * 19 + iteration * 3) % 59 - 29) / 32.0f);
            }
            auto * q = quantized.data() + (size_t) col * blocks;
            if (strcmp(impl, "lut_c") == 0) {
                quantize_row_ifairy_q16_lut_c(input.data(), q, k);
            } else {
                quantize_row_ifairy_q16_tensor(input.data(), q, k);
            }
            if (unsafe) {
                for (int b = 0; b < blocks; ++b) {
                    // Include both signs and same-sign pairs that would saturate a LUT.
                    for (int j = 0; j < QK_IFAIRY; ++j) {
                        q[b].x_real[j] = (uint8_t) (int8_t) ((j / 2) % 2 ? 127 : -127);
                        q[b].x_imag[j] = (uint8_t) (int8_t) ((j / 3) % 2 ? -127 : 127);
                    }
                }
            }
            char * dst = (char *) x->data + (col % n) * x->nb[1] + (col / n) * x->nb[2];
            memcpy(dst, f32 ? (const void *) input.data() : (const void *) q, ggml_row_size(act_type, k));
        }

        executions = 0;
        require(ggml_graph_compute_with_ctx(ctx.ptr, graph, threads) == GGML_STATUS_SUCCESS, "graph compute failed");
        const bool expect_lut = enabled && strcmp(enabled, "0") != 0 && !unsafe && !strided_weights && batches == 1;
        require(executions == (expect_lut ? 1 : 0), "LUT execution/fallback dispatch mismatch");

        // Independent complex dot-product oracle using the exact quantized
        // inputs. Decode weight codes directly; never call the LUT kernels.
        for (int col = 0; col < n * batches; ++col) {
            for (int row = 0; row < m; ++row) {
                double real = 0.0;
                double imag = 0.0;
                for (int b = 0; b < blocks; ++b) {
                    const auto & wb = weights[(size_t) row * blocks + b];
                    const auto & qb = quantized[(size_t) col * blocks + b];
                    const double wr = ggml_fp16_to_fp32(wb.d_real);
                    const double wi = ggml_fp16_to_fp32(wb.d_imag);
                    const double xr = ggml_fp16_to_fp32(qb.d_real);
                    const double xi = ggml_fp16_to_fp32(qb.d_imag);
                    for (int j = 0; j < QK_IFAIRY; ++j) {
                        const int    code         = (wb.qs[(j / 64) * 16 + (j & 15)] >> (2 * ((j >> 4) & 3))) & 3;
                        const double real_codes[] = { -wr, wr, 0.0, 0.0 };
                        const double imag_codes[] = { 0.0, 0.0, -wi, wi };
                        const double a            = real_codes[code];
                        const double b_im         = imag_codes[code];
                        const double c            = (int8_t) qb.x_real[j] * xr;
                        const double d            = (int8_t) qb.x_imag[j] * xi;
                        real += a * c + b_im * d;
                        imag += b_im * c - a * d;
                    }
                }
                ggml_bf16_t actual[2];
                memcpy(actual, (char *) out->data + row * out->nb[0] + (col % n) * out->nb[1] + (col / n) * out->nb[2],
                       sizeof(actual));
                const double reference[] = { real, imag };
                for (int part = 0; part < 2; ++part) {
                    const double value = ggml_bf16_to_fp32(actual[part]);
                    if (!std::isfinite(value) ||
                        std::abs(value - reference[part]) > 0.008 * std::abs(reference[part]) + 1e-5) {
                        fprintf(stderr,
                                "M=%d N=%d K=%d t=%d impl=%s f32=%d unsafe=%d strides=%d/%d batches=%d on=%s "
                                "iteration=%d row=%d col=%d part=%d got=%g expected=%g\n",
                                m, n, k, threads, impl, f32, unsafe, strided_weights, strided_act, batches,
                                enabled ? enabled : "unset", iteration, row, col, part, value, reference[part]);
                        require(false, "complex product disagrees with oracle");
                    }
                }
            }
        }
    }
}

static void test_missing_resources() {
    env("GGML_IFAIRY_LUT", "1");
    context ctx;
    auto *  w   = ggml_new_tensor_2d(ctx.ptr, GGML_TYPE_IFAIRY, 256, 17);
    auto *  x   = ggml_new_tensor_2d(ctx.ptr, GGML_TYPE_F32, 256, 3);
    auto *  out = ggml_mul_mat(ctx.ptr, w, x);
    memset(w->data, 0, ggml_nbytes(w));
    memset(out->data, 0xa5, ggml_nbytes(out));
    std::vector<uint8_t> scratch(ggml_legacy_ifairy_cpu_work_size(out, 1));
    require(scratch.size() >= ggml_row_size(GGML_TYPE_IFAIRY_Q16, ggml_nelements(x)),
            "plan omits direct fallback workspace");
    ggml_compute_params params{ 0, 1, scratch.size(), scratch.data(), nullptr };
    require(!ggml_legacy_ifairy_cpu_compute(&params, out), "missing pack must decline LUT");
    require(ggml_ifairy_lut_transform_tensor(w, nullptr), "test pack failed");
    params.wsize = 0;
    require(!ggml_legacy_ifairy_cpu_compute(&params, out), "missing workspace must decline LUT");
    const auto * bytes = static_cast<const uint8_t *>(out->data);
    require(std::all_of(bytes, bytes + ggml_nbytes(out), [](uint8_t byte) { return byte == 0xa5; }),
            "declined LUT modified the output");
}

int main() {
#if !((defined(__aarch64__) && defined(__ARM_NEON)) || defined(__x86_64__) || defined(_M_X64))
    puts("LUT MUL_MAT dispatch is not enabled on this architecture; direct fallbacks remain available.");
    return 77;
#endif
    try {
        ggml_log_set(log_callback, nullptr);
        test_missing_resources();
        for (int threads : { 1, 4, 8 }) {
            for (const char * impl : { "auto", "lut16", "lut_c" }) {
                for (const auto & shape : {
                         std::vector<int>{ 1,  1, 256  },
                         { 17, 3, 512  },
                         { 65, 9, 1536 }
                }) {
                    for (bool f32 : { false, true }) {
                        run_case(shape[0], shape[1], shape[2], threads, impl, f32);
                    }
                }
            }
            run_case(17, 3, 512, threads, "auto", false, true);
            run_case(17, 3, 512, threads, "auto", false, false, true);
            run_case(17, 3, 512, threads, "auto", true, false, false, true);
            run_case(17, 3, 512, threads, "auto", false, false, false, false, 2);
            run_case(17, 3, 512, threads, "auto", false, false, false, false, 1, "0");
            run_case(17, 3, 512, threads, "auto", false, false, false, false, 1, nullptr);
        }
        ggml_log_set(nullptr, nullptr);
        puts(
            "LUT MUL_MAT: 144 graph executions passed (dispatch, complex oracle, tails, strides, broadcasts, range "
            "fallback).");
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "LUT MUL_MAT regression failed: %s\n", error.what());
        return 1;
    }
}
