#include "llama-model.h"

#include "ggml-cpp.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>


static bool llama_backend_dev_is_metal(ggml_backend_dev_t dev) {
    if (!dev) {
        return false;
    }

    ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
    const char *       reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return reg_name && strcmp(reg_name, "Metal") == 0;
}

static bool llama_backend_dev_supports_row4(ggml_backend_dev_t dev, uint32_t schema_version);


const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_140M:          return "140M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_360M:          return "360M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_558M:          return "558M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_950M:          return "950M";
        case LLM_TYPE_0_3B:          return "0.3B";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_2B:          return "1.2B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_1_8B:          return "1.8B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_2_9B:          return "2.9B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_36B:           return "36B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_120B:          return "120B";
        case LLM_TYPE_142B:          return "142B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_290B:          return "290B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_405B:          return "405B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LLM_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LLM_TYPE_A13B:          return "A13B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_235B_A22B:     return "235B.A22B";
        case LLM_TYPE_300B_A47B:     return "300B.A47B";
        case LLM_TYPE_355B_A32B:     return "355B.A32B";
        case LLM_TYPE_E2B:           return "E2B";
        case LLM_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                int n_expert_used = hparams.n_expert_used;
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_ADD_ID:
            {
                int n_expert_used = hparams.n_expert_used;
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_add_id(ctx, a, w, c);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor = ggml_div(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                int n_embd_head = hparams.n_embd_head_v;
                int n_head = hparams.n_head();
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_rope_ext(
                    ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_SSM_CONV:
            {
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0] - 1 + n_seq_tokens, w->ne[1], n_seqs);
                op_tensor = ggml_ssm_conv(ctx, conv_x, w);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                // w is ssm_a, which is used to distinguish Mamba-1 and Mamba-2
                const int64_t d_state      = w->ne[0] == 1 ? hparams.ssm_d_state : w->ne[0];
                const int64_t n_head       = w->ne[1];
                const int64_t head_dim     = hparams.ssm_d_inner / n_head;
                const int64_t n_group      = hparams.ssm_n_group ? hparams.ssm_n_group : 1;
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
                ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * dt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * B   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * C   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
                op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C, ids);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S = 123;
                const int64_t H = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs = 123;
                ggml_tensor  * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * tf = w;
                ggml_tensor  * td = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd = hparams.n_embd;
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd, w->ne[1], 1, 1);
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        case GGML_OP_SCALE:
            {
                op_tensor = ggml_scale(ctx, w, 1.0f);
            } break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

static bool buft_list_has_non_cpu_dev(const buft_list_t & buft_list) {
    for (const auto & cur : buft_list) {
        if (cur.first && ggml_backend_dev_type(cur.first) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return true;
        }
    }
    return false;
}

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t & buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices, bool use_extra_bufts) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    for (auto * dev : devices) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
        if (buft) {
            buft_list.emplace_back(dev, buft);
            break;
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    return buft_list;
}

struct llama_model::impl {
    impl() {}
    ~impl() {}

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored
    std::vector<ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_ptr> bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;
};

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {}

void llama_model::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model::load_arch(llama_model_loader & ml) {
    arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

void llama_model::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.meta.get();

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    {
        bool has_row4_marker = false;
        for (int i = 0; i < gguf_get_n_kv(ctx); ++i) {
            const char * key = gguf_get_key(ctx, i);
            if (strncmp(key, "row4.", 5) == 0) {
                has_row4_marker = true;
                if (strcmp(key, "row4.schema_version") != 0 && strcmp(key, "row4.weight_layout") != 0 &&
                    strcmp(key, "row4.execution_profile") != 0 && strcmp(key, "row4.codebook") != 0 &&
                    strcmp(key, "row4.numeric_profile") != 0 && strcmp(key, "row4.qkv_order") != 0 &&
                    strcmp(key, "row4.ffn_order") != 0 && strcmp(key, "row4.lm_head_layout") != 0) {
                    throw std::runtime_error(format("unsupported Row4 metadata key: %s", key));
                }
            }
        }
        if (!has_row4_marker) {
            for (const auto & weight : ml.weights_map) {
                if (weight.first.find(".row4.") != std::string::npos ||
                    weight.first.compare(0, strlen("output.w8."), "output.w8.") == 0) {
                    has_row4_marker = true;
                    break;
                }
            }
        }

        uint32_t    schema_version = 0;
        std::string weight_layout;
        std::string execution_profile;
        std::string codebook;
        std::string numeric_profile;
        std::string qkv_order;
        std::string ffn_order;
        std::string lm_head_layout;

        const bool has_schema            = ml.get_key(LLM_KV_ROW4_SCHEMA_VERSION, schema_version, false);
        const bool has_layout            = ml.get_key(LLM_KV_ROW4_WEIGHT_LAYOUT, weight_layout, false);
        const bool has_execution_profile = ml.get_key(LLM_KV_ROW4_EXECUTION_PROFILE, execution_profile, false);
        const bool has_codebook          = ml.get_key(LLM_KV_ROW4_CODEBOOK, codebook, false);
        const bool has_numeric           = ml.get_key(LLM_KV_ROW4_NUMERIC_PROFILE, numeric_profile, false);
        const bool has_qkv               = ml.get_key(LLM_KV_ROW4_QKV_ORDER, qkv_order, false);
        const bool has_ffn               = ml.get_key(LLM_KV_ROW4_FFN_ORDER, ffn_order, false);
        const bool has_lm_head           = ml.get_key(LLM_KV_ROW4_LM_HEAD_LAYOUT, lm_head_layout, false);
        uint32_t   general_file_type     = 0;
        const bool has_general_file_type = ml.get_key(LLM_KV_GENERAL_FILE_TYPE, general_file_type, false);

        const bool any_row4 = has_row4_marker || has_schema || has_layout || has_execution_profile || has_codebook ||
                              has_numeric || has_qkv || has_ffn || has_lm_head ||
                              (has_general_file_type && general_file_type == LLAMA_FTYPE_MOSTLY_ROW4);
        if (any_row4) {
            if (arch != LLM_ARCH_QWEN3) {
                throw std::runtime_error("Row4 requires general.architecture=qwen3");
            }
            if (!(has_schema && has_layout && has_codebook && has_numeric && has_qkv && has_ffn && has_lm_head)) {
                throw std::runtime_error(
                    "incomplete Row4 descriptor: row4.schema_version, weight_layout, codebook, numeric_profile, "
                    "qkv_order, ffn_order, and lm_head_layout are all required");
            }

            uint32_t general_alignment = 0;
            if (!has_general_file_type || general_file_type != LLAMA_FTYPE_MOSTLY_ROW4) {
                throw std::runtime_error(
                    format("Row4 requires general.file_type=%u", (unsigned) LLAMA_FTYPE_MOSTLY_ROW4));
            }

            const bool common_descriptor_matches =
                codebook == "uv_axis_v1" && numeric_profile == "bf16_a8_away_i32_bf16_v1" && qkv_order == "q_k_v" &&
                ffn_order == "gate_up" && lm_head_layout == "s8_m16k128_rowmajor_v1";
            bool descriptor_matches = false;
            if (schema_version == 1) {
                if (has_execution_profile) {
                    throw std::runtime_error("Row4 schema v1 forbids row4.execution_profile");
                }
                if (!ml.get_key(LLM_KV_GENERAL_ALIGNMENT, general_alignment, false) || general_alignment != 64) {
                    throw std::runtime_error("Row4 schema v1 requires general.alignment=64");
                }
                descriptor_matches = weight_layout == "m16k128_split8_v1" && common_descriptor_matches;
            } else if (schema_version == 2) {
                if (!has_execution_profile) {
                    throw std::runtime_error("Row4 schema v2 requires row4.execution_profile");
                }
                if (!ml.get_key(LLM_KV_GENERAL_ALIGNMENT, general_alignment, false) || general_alignment != 128) {
                    throw std::runtime_error("Row4 schema v2 requires general.alignment=128");
                }
                descriptor_matches = weight_layout == "m32k256_pair2_split8_v2" &&
                                     execution_profile == "metal_only_v1" && common_descriptor_matches;
            }

            if (!descriptor_matches) {
                throw std::runtime_error(
                    format("unsupported Row4 descriptor: schema=%u layout=%s execution=%s codebook=%s numeric=%s "
                           "qkv=%s ffn=%s lm_head=%s",
                           schema_version, weight_layout.c_str(),
                           has_execution_profile ? execution_profile.c_str() : "<absent>", codebook.c_str(),
                           numeric_profile.c_str(), qkv_order.c_str(), ffn_order.c_str(), lm_head_layout.c_str()));
            }

            row4_enabled        = true;
            row4_schema_version = schema_version;
        }
    }

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,    hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,  hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH, hparams.n_embd_features);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);
    }

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(
        hparams.recurrent_layer_arr.begin(),
        hparams.recurrent_layer_arr.end(),
        llm_arch_is_recurrent(ml.get_arch()));

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);

    std::fill(hparams.swa_layers.begin(), hparams.swa_layers.end(), 0);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    // by default assume that the sliding-window layers use the same scaling type as the non-sliding-window layers
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k = 2 * hparams.n_embd / hparams.n_head();

        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

        hparams.n_embd_head_v = 2 * hparams.n_embd / hparams.n_head();

        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON) {
            if (hparams.n_rot != hparams.n_embd_head_k) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k));
            }
        }
    } else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k = 0;
        hparams.n_embd_head_v = 0;
    }

    // for differentiating model types
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // arch-specific KVs
    switch (arch) {
        case LLM_ARCH_IFAIRY:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24:
                        type = LLM_TYPE_700M;
                        break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
            }
            break;
        default:
            throw std::runtime_error("this repository supports only the ifairy architecture");
    }

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);
}

void llama_model::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool llama_model::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & n_gpu_layers = params.n_gpu_layers;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer = hparams.n_layer;

    const bool use_mmap_buffer = true;

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s)\n", __func__,
                   ml.use_mmap ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts);
    for (auto * dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev, std::move(buft_list));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr ||
                    std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i];
            size_t             total;
            size_t             free;
            ggml_backend_dev_memory(dev, &free, &total);
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }
    const int i_gpu_start         = std::max((int) hparams.n_layer - n_gpu_layers, (int) 0);
    const int act_gpu_layers      = devices.empty() ? 0 : std::min(n_gpu_layers, (int) n_layer + 1);
    auto      get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < (int) hparams.n_layer && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il,
                            ggml_backend_dev_name(cpu_dev), is_swa);
            return { cpu_dev, &pimpl->cpu_buft_list };
        }
        const int layer_gpu =
            std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start) / act_gpu_layers) -
            splits.begin();
        auto * dev = devices.at(layer_gpu);
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev),
                        is_swa);
        return { dev, &pimpl->gpu_buft_list.at(dev) };
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer in the general case
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // For a fully Metal-offloaded Bundle model, keep the embedding lookup on Metal as well
    // so an otherwise unnecessary CPU split is not introduced at the start of the graph.

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer);

    if (row4_enabled) {
        std::vector<ggml_backend_dev_t> row4_devs;
        for (const auto & layer_dev : pimpl->dev_layer) {
            if (std::find(row4_devs.begin(), row4_devs.end(), layer_dev.dev) == row4_devs.end()) {
                row4_devs.push_back(layer_dev.dev);
            }
        }
        if (std::find(row4_devs.begin(), row4_devs.end(), pimpl->dev_output.dev) == row4_devs.end()) {
            row4_devs.push_back(pimpl->dev_output.dev);
        }
        for (ggml_backend_dev_t dev : row4_devs) {
            if (row4_schema_version == 2 && !llama_backend_dev_is_metal(dev)) {
                throw std::runtime_error(format(
                    "Qwen3 Row4 schema v2 uses execution_profile=metal_only_v1; every Row4 layer and the W8 output "
                    "must be placed on Metal, but device %s was selected",
                    dev ? ggml_backend_dev_name(dev) : "none"));
            }
            if (!llama_backend_dev_supports_row4(dev, row4_schema_version)) {
                throw std::runtime_error(format(
                    "Qwen3 Row4 schema v%u requires ROW4_LINEAR and W8A8_LINEAR support; device %s is unsupported",
                    row4_schema_version, dev ? ggml_backend_dev_name(dev) : "none"));
            }
        }
    }

    // one ggml context per buffer type
    int max_n_tensors = ml.n_tensors;
    max_n_tensors += 1;            // duplicated output tensor
    max_n_tensors += n_layer * 2;  // duplicated rope freq tensors
    const size_t ctx_size = ggml_tensor_overhead() * max_n_tensors;

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ctx_size,
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }

            ctx_map[buft] = ctx;
            pimpl->ctxs.emplace_back(ctx);

            return ctx;
        }
        return it->second;
    };

    const auto TENSOR_DUPLICATED   = llama_model_loader::TENSOR_DUPLICATED;
    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;
    const auto TENSOR_SKIP         = llama_model_loader::TENSOR_SKIP;

    // create tensors for the weights
    {
        // note: cast to int64_t since we will use these for the tensor dimensions
        const int64_t n_embd       = hparams.n_embd;
        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa();
        const int64_t n_ff         = hparams.n_ff();
        const int64_t n_embd_gqa   = n_embd_v_gqa;
        const int64_t n_vocab      = vocab.n_tokens();
        const int64_t n_expert     = hparams.n_expert;

        if (n_expert > 0 && hparams.n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        int                        n_moved_tensors       = 0;
        ggml_tensor *              first_moved_tensor    = nullptr;
        ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
        ggml_backend_buffer_type_t first_moved_to_buft   = nullptr;

        auto create_tensor = [&](const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne,
                                 int flags) -> ggml_tensor * {
            ggml_tensor * t_meta = ml.get_tensor_meta(tn.str().c_str());

            if (!t_meta) {
                if (flags & TENSOR_NOT_REQUIRED) {
                    return nullptr;
                }
                throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
            }

            // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
            // the tensor is duplicated
            // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
            llm_tensor tn_tensor = tn.tensor;
            if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && flags & TENSOR_DUPLICATED) {
                tn_tensor = LLM_TENSOR_OUTPUT;
            }

            llm_tensor_info info;
            try {
                info = llm_tensor_info_for(tn_tensor);
            } catch (const std::out_of_range & e) {
                throw std::runtime_error(format("missing tensor info mapping for %s", tn.str().c_str()));
            }

            // skip unused tensors
            if (info.op == GGML_OP_NONE || flags & TENSOR_SKIP) {
                const size_t nbytes = ggml_nbytes(t_meta);
                LLAMA_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", tn.str().c_str(), nbytes);

                ml.size_data -= nbytes;
                ml.n_created++;

                return nullptr;
            }

            // tensors with "bias" suffix are always used with GGML_OP_ADD or GGML_OP_ADD_ID
            ggml_op op;
            bool    bias = tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0;
            if (bias) {
                if (info.op == GGML_OP_MUL_MAT_ID) {
                    op = GGML_OP_ADD_ID;
                } else {
                    op = GGML_OP_ADD;
                }
            } else {
                op = info.op;
            }
            const bool row4_storage = row4_enabled && tn.suffix != nullptr &&
                                      (strcmp(tn.suffix, "row4.codes") == 0 || strcmp(tn.suffix, "row4.scales") == 0 ||
                                       strcmp(tn.suffix, "w8.codes") == 0 || strcmp(tn.suffix, "w8.scales") == 0);
            if (row4_storage) {
                // These tensors are opaque inputs to ROW4_LINEAR/W8A8_LINEAR, not generic matrix weights.
                op = GGML_OP_NONE;
            }

            // sanity checks
            if (info.layer == LLM_TENSOR_LAYER_INPUT || info.layer == LLM_TENSOR_LAYER_OUTPUT) {
                if (tn.bid != -1) {
                    GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());
                }
            } else {
                if (tn.bid == -1) {
                    GGML_ABORT("repeating layer tensor %s used without a layer number", tn.str().c_str());
                }
            }

            // select the buffer type for this tensor
            buft_list_t *      buft_list;
            ggml_backend_dev_t target_dev;
            switch (info.layer) {
                case LLM_TENSOR_LAYER_INPUT:
                    buft_list  = pimpl->dev_input.buft_list;
                    target_dev = pimpl->dev_input.dev;
                    break;
                case LLM_TENSOR_LAYER_OUTPUT:
                    buft_list  = pimpl->dev_output.buft_list;
                    target_dev = pimpl->dev_output.dev;
                    break;
                case LLM_TENSOR_LAYER_REPEATING:
                    buft_list  = pimpl->dev_layer.at(tn.bid).buft_list;
                    target_dev = pimpl->dev_layer.at(tn.bid).dev;
                    break;
                default:
                    GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
            }

            ggml_backend_buffer_type_t buft = nullptr;

            // check overrides
            if (ml.tensor_buft_overrides) {
                std::string tensor_name = tn.str();
                for (const auto * overrides = ml.tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
                    std::regex pattern(overrides->pattern);
                    if (std::regex_search(tensor_name, pattern)) {
                        if (overrides->buft == ggml_backend_cpu_buffer_type()) {
                            // when overriding to a CPU buffer, consider the extra buffer types
                            buft = select_weight_buft(hparams, t_meta, op, pimpl->cpu_buft_list);
                        } else {
                            buft = overrides->buft;
                        }

                        LLAMA_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n", tensor_name.c_str(),
                                        ggml_nbytes(t_meta) / 1024 / 1024, ggml_type_name(t_meta->type),
                                        ggml_backend_buft_name(buft));
                        break;
                    }
                }
            }

            if (row4_storage) {
                if (buft && ggml_backend_buft_get_device(buft) != target_dev) {
                    throw std::runtime_error(
                        format("Row4 tensor override for %s would split a codes/scales bundle across devices",
                               tn.str().c_str()));
                }
                // Codes and scales use the same target device and its ordinary storage. No persistent prepack exists.
                buft = ggml_backend_dev_buffer_type(target_dev);
                if (!buft) {
                    throw std::runtime_error(format("target device %s has no buffer for Row4 tensor %s",
                                                    ggml_backend_dev_name(target_dev), tn.str().c_str()));
                }
            }

            if (!buft) {
                buft = select_weight_buft(hparams, t_meta, op, *buft_list);
                if (!buft) {
                    throw std::runtime_error(
                        format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
                }
            }

            // avoid using a host buffer when using mmap
            auto * buft_dev = ggml_backend_buft_get_device(buft);
            if (ml.use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error("no CPU backend found");
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);
            }

            if (row4_enabled && row4_schema_version == 2) {
                ggml_backend_dev_t final_dev = ggml_backend_buft_get_device(buft);
                if (!llama_backend_dev_is_metal(final_dev)) {
                    throw std::runtime_error(format(
                        "Qwen3 Row4 schema v2 uses execution_profile=metal_only_v1; model tensor %s was assigned "
                        "to non-Metal buffer type %s",
                        tn.str().c_str(), ggml_backend_buft_name(buft)));
                }
            }

            if (buft != buft_list->front().second) {
                n_moved_tensors++;
                if (!first_moved_tensor) {
                    first_moved_tensor    = t_meta;
                    first_moved_from_buft = buft_list->front().second;
                    first_moved_to_buft   = buft;
                }
            }

            ggml_context * ctx = ctx_for_buft(buft);

            // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
            if (flags & TENSOR_DUPLICATED) {
                ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str());
                if (t) {
                    return t;
                }
            }
            return ml.create_tensor(ctx, tn, ne, flags);
        };

        layers.resize(n_layer);

        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);
        switch (arch) {
            case LLM_ARCH_IFAIRY:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM), { n_embd * 2 }, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT), { n_embd * 2, n_vocab }, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM, i), { n_embd * 2 }, 0);
                        layer.attn_sub_norm = create_tensor(tn(LLM_TENSOR_ATTN_SUB_NORM, i), { n_embd * 2 }, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, i), { n_embd, n_embd }, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, i), { n_embd, n_embd_gqa / 2 }, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, i), { n_embd, n_embd_gqa / 2 }, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, i), { n_embd, n_embd }, 0);

                        layer.ffn_norm     = create_tensor(tn(LLM_TENSOR_FFN_NORM, i), { n_embd * 2 }, 0);
                        layer.ffn_sub_norm = create_tensor(tn(LLM_TENSOR_FFN_SUB_NORM, i), { n_ff * 2 }, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, i), { n_embd, n_ff }, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, i), { n_ff, n_embd }, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, i), { n_embd, n_ff }, 0);
                    }
                }
                break;
            default:
                throw std::runtime_error("this repository supports only the ifairy architecture");
        }

        if (n_moved_tensors > 0) {
            LLAMA_LOG_DEBUG(
                "%s: tensor '%s' (%s) (and %d others) cannot be used with preferred buffer type %s, using %s instead\n",
                __func__, first_moved_tensor->name, ggml_type_name(first_moved_tensor->type), n_moved_tensors - 1,
                ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
        }
    }

    ml.done_getting_tensors();

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_bufs;
    ctx_bufs.reserve(ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ctx_map.size() * ml.files.size();
    pimpl->bufs.reserve(n_max_backend_buffer);

    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context *             ctx  = it.second;

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft                = buft == ggml_backend_dev_buffer_type(dev);

        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer, then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last;  // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t          max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf =
                    ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                pimpl->bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            pimpl->bufs.emplace_back(buf);
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init(ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        if (pimpl->bufs.empty()) {
            throw std::runtime_error("failed to allocate buffer");
        }

        for (auto & buf : buf_map) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ctx_bufs.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
        }

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers),
                       max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & buf : pimpl->bufs) {
        LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()),
                       ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (auto & ctx : pimpl->ctxs) {
        for (auto * cur = ggml_get_first_tensor(ctx.get()); cur != NULL; cur = ggml_get_next_tensor(ctx.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // load tensor data
    for (auto & it : ctx_bufs) {
        ggml_context * ctx  = it.first;
        auto &         bufs = it.second;
        if (!ml.load_all_data(ctx, bufs, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback,
                              params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch             = %s\n", __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only       = %d\n", __func__, hparams.vocab_only);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n", __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd           = %u\n", __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_layer          = %u\n", __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_head           = %s\n", __func__,
                       print_f([&](uint32_t il) { return hparams.n_head(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv        = %s\n", __func__,
                       print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot            = %u\n", __func__, hparams.n_rot);
        LLAMA_LOG_INFO("%s: n_swa            = %u\n", __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any       = %u\n", __func__, hparams.is_swa_any());
        LLAMA_LOG_INFO("%s: n_embd_head_k    = %u\n", __func__, hparams.n_embd_head_k);
        LLAMA_LOG_INFO("%s: n_embd_head_v    = %u\n", __func__, hparams.n_embd_head_v);
        LLAMA_LOG_INFO("%s: n_gqa            = %s\n", __func__,
                       print_f([&](uint32_t il) { return hparams.n_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa     = %s\n", __func__,
                       print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa     = %s\n", __func__,
                       print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps       = %.1e\n", __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps   = %.1e\n", __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv      = %.1e\n", __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias = %.1e\n", __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale    = %.1e\n", __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: f_attn_scale     = %.1e\n", __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: n_ff             = %s\n", __func__,
                       print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_expert         = %u\n", __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used    = %u\n", __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: causal attn      = %d\n", __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type     = %d\n", __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type        = %d\n", __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling     = %s\n", __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n", __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train = %g\n", __func__, hparams.rope_freq_scale_train);
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn  = %u\n", __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_finetuned   = %s\n", __func__, hparams.rope_finetuned ? "yes" : "unknown");
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out        = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (auto label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]    = %s\n", __func__, i++, label.c_str());
            }
        }
    }

    if (arch == LLM_ARCH_MAMBA || arch == LLM_ARCH_MAMBA2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_FALCON_H1 ||
        arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_NEMOTRON_H) {
        LLAMA_LOG_INFO("%s: ssm_d_conv       = %u\n", __func__, hparams.ssm_d_conv);
        LLAMA_LOG_INFO("%s: ssm_d_inner      = %u\n", __func__, hparams.ssm_d_inner);
        LLAMA_LOG_INFO("%s: ssm_d_state      = %u\n", __func__, hparams.ssm_d_state);
        LLAMA_LOG_INFO("%s: ssm_dt_rank      = %u\n", __func__, hparams.ssm_dt_rank);
        LLAMA_LOG_INFO("%s: ssm_n_group      = %u\n", __func__, hparams.ssm_n_group);
        LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms   = %d\n", __func__, hparams.ssm_dt_b_c_rms);
    }

    LLAMA_LOG_INFO("%s: model type       = %s\n", __func__, type_name().c_str());
    if (pimpl->n_elements >= 1e12) {
        LLAMA_LOG_INFO("%s: model params     = %.2f T\n", __func__, pimpl->n_elements * 1e-12);
    } else if (pimpl->n_elements >= 1e9) {
        LLAMA_LOG_INFO("%s: model params     = %.2f B\n", __func__, pimpl->n_elements * 1e-9);
    } else if (pimpl->n_elements >= 1e6) {
        LLAMA_LOG_INFO("%s: model params     = %.2f M\n", __func__, pimpl->n_elements * 1e-6);
    } else {
        LLAMA_LOG_INFO("%s: model params     = %.2f K\n", __func__, pimpl->n_elements * 1e-3);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name     = %s\n", __func__, name.c_str());

    if (row4_enabled) {
        LLAMA_LOG_INFO("%s: row4 schema      = %u\n", __func__, row4_schema_version);
        LLAMA_LOG_INFO("%s: row4 layout      = %s\n", __func__,
                       row4_schema_version == 2 ? "m32k256_pair2_split8_v2" : "m16k128_split8_v1");
        if (row4_schema_version == 2) {
            LLAMA_LOG_INFO("%s: row4 execution   = metal_only_v1\n", __func__);
        }
        LLAMA_LOG_INFO("%s: row4 numeric     = bf16_a8_away_i32_bf16_v1\n", __func__);
    }

    if (arch == LLM_ARCH_DEEPSEEK) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n", __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n", __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n", __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n", __func__, hparams.expert_weights_scale);
    }

    if (arch == LLM_ARCH_DEEPSEEK2) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n", __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_lora_q             = %d\n", __func__, hparams.n_lora_q);
        LLAMA_LOG_INFO("%s: n_lora_kv            = %d\n", __func__, hparams.n_lora_kv);
        LLAMA_LOG_INFO("%s: n_embd_head_k_mla    = %d\n", __func__, hparams.n_embd_head_k_mla);
        LLAMA_LOG_INFO("%s: n_embd_head_v_mla    = %d\n", __func__, hparams.n_embd_head_v_mla);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n", __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n", __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n", __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n", __func__, hparams.expert_weights_norm);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n", __func__,
                       llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul    = %.4f\n", __func__, hparams.rope_yarn_log_mul);
    }

    if (arch == LLM_ARCH_QWEN2MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n", __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp       = %d\n", __func__, hparams.n_ff_shexp);
    }

    if (arch == LLM_ARCH_QWEN3MOE || arch == LLM_ARCH_OPENAI_MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n", __func__, hparams.n_ff_exp);
    }

    if (arch == LLM_ARCH_MINICPM || arch == LLM_ARCH_GRANITE || arch == LLM_ARCH_GRANITE_MOE ||
        arch == LLM_ARCH_GRANITE_HYBRID) {
        LLAMA_LOG_INFO("%s: f_embedding_scale = %f\n", __func__, hparams.f_embedding_scale);
        LLAMA_LOG_INFO("%s: f_residual_scale  = %f\n", __func__, hparams.f_residual_scale);
        LLAMA_LOG_INFO("%s: f_attention_scale = %f\n", __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: n_ff_shexp        = %d\n", __func__, hparams.n_ff_shexp);
    }

    if (arch == LLM_ARCH_BAILINGMOE) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n", __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n", __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n", __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n", __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n", __func__, hparams.expert_weights_norm);
    }

    if (arch == LLM_ARCH_SMALLTHINKER) {
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n", __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n", __func__,
                       llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template <typename F> static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx{ ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf{ ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor *           op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

static bool llama_backend_dev_supports_row4(ggml_backend_dev_t dev, uint32_t schema_version) {
    if (!dev) {
        return false;
    }

    ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
    const char *       reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    const bool         is_metal = reg_name && strcmp(reg_name, "Metal") == 0;

    if (schema_version == 2 && !is_metal) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    if (!is_metal && !(reg_name && strcmp(reg_name, "CPU") == 0)) {
        return false;
    }
#else
    if (!is_metal) {
        return false;
    }
#endif

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!buft) {
        return false;
    }

    auto row4_probe = [schema_version](ggml_context * ctx) {
        const int64_t k     = schema_version == 2 ? 256 : 128;
        ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
        ggml_tensor * codes = nullptr;
        if (schema_version == 2) {
            codes = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES_PAIR2, 128, 8, 1, 4);
        } else {
            codes = ggml_new_tensor_4d(ctx, GGML_TYPE_ROW4_CODES, 64, 4, 1, 8);
        }
        ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 128);
        return ggml_row4_linear(ctx, x, codes, scales, 128, k);
    };
    auto w8_probe = [](ggml_context * ctx) {
        ggml_tensor * x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
        ggml_tensor * codes  = ggml_new_tensor_4d(ctx, GGML_TYPE_I8, 128, 16, 1, 8);
        ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        return ggml_w8a8_linear(ctx, x, codes, scales, 128, 128);
    };

    return buft_supported(buft, dev, row4_probe) && buft_supported(buft, dev, w8_probe);
}

template <typename F> static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t         cur_dev  = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(*pimpl->dev_layer.at(il).buft_list, [&](ggml_context * ctx) {
        ggml_tensor * cur       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
        ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
        return ggml_add(ctx, cur, layer_dir);
    });
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
                           [name](const std::pair<std::string, ggml_tensor *> & it) { return it.first == name; });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float llama_model::get_rope_freq_base(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_per_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

template <bool iswa>

// TODO: move up next to build_starcoder

struct llm_graph_context_mamba : public llm_graph_context {
    llm_graph_context_mamba(const llm_graph_params & params) : llm_graph_context(params) {}

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp,
                                    ggml_tensor *        cur,
                                    const llama_model &  model,
                                    const llama_ubatch & ubatch,
                                    int                  il) {
        const auto * mctx_cur = inp->mctx;

        const auto kv_head = mctx_cur->get_head();

        const auto & layer = model.layers[il];

        const int64_t d_conv         = hparams.ssm_d_conv;
        const int64_t d_inner        = hparams.ssm_d_inner;
        const int64_t d_state        = hparams.ssm_d_state;
        const int64_t dt_rank        = hparams.ssm_dt_rank;
        const int64_t head_dim       = 1;
        const int64_t n_seqs         = ubatch.n_seqs;
        // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
        const bool    ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;

        const int64_t n_seq_tokens = ubatch.n_seq_tokens;

        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
        ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

        ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
        conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
        ggml_tensor * xz = build_lora_mm(layer.ssm_in, cur);
        // split the above in two
        // => {d_inner, n_seq_tokens, n_seqs}
        ggml_tensor * x  = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
        ggml_tensor * z  = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2],
                                        d_inner * ggml_element_size(xz));

        // conv
        {
            // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
            ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);

            // copy last (d_conv - 1) columns back into the state cache
            ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1],
                                                   conv_x->nb[2], n_seq_tokens * (conv_x->nb[0]));

            ggml_build_forward_expand(
                gf, ggml_cpy(ctx0, last_conv,
                             ggml_view_1d(ctx0, conv_states_all, (d_conv - 1) * (d_inner) * (n_seqs),
                                          kv_head * (d_conv - 1) * (d_inner) *ggml_element_size(conv_states_all))));

            // 1D convolution
            // The equivalent is to make a self-overlapping view of conv_x
            // over d_conv columns at each stride in the 3rd dimension,
            // then element-wise multiply that with the conv1d weight,
            // then sum the elements of each row,
            // (the last two steps are a dot product over rows (also doable with mul_mat))
            // then permute away the ne[0] dimension,
            // and then you're left with the resulting x tensor.
            // For simultaneous sequences, all sequences need to have the same length.
            x = ggml_ssm_conv(ctx0, conv_x, layer.ssm_conv1d);

            // bias
            x = ggml_add(ctx0, x, layer.ssm_conv1d_b);

            x = ggml_silu(ctx0, x);
        }

        // ssm
        {
            // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
            ggml_tensor * x_db = build_lora_mm(layer.ssm_x, x);
            // split
            ggml_tensor * dt   = ggml_view_3d(ctx0, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
            ggml_tensor * B =
                ggml_view_4d(ctx0, x_db, d_state, /* n_group */ 1, n_seq_tokens, n_seqs, d_state * x_db->nb[0],
                             x_db->nb[1], x_db->nb[2], ggml_element_size(x_db) * dt_rank);
            ggml_tensor * C =
                ggml_view_4d(ctx0, x_db, d_state, /* n_group */ 1, n_seq_tokens, n_seqs, d_state * x_db->nb[0],
                             x_db->nb[1], x_db->nb[2], ggml_element_size(x_db) * (dt_rank + d_state));

            // Some Mamba variants (e.g. FalconMamba, Jamba) apply RMS norm in B, C & Dt layers
            if (ssm_dt_b_c_rms || (layer.ssm_dt_norm && layer.ssm_b_norm && layer.ssm_c_norm)) {
                dt = build_norm(dt, layer.ssm_dt_norm, NULL, LLM_NORM_RMS, il);
                B  = build_norm(B, layer.ssm_b_norm, NULL, LLM_NORM_RMS, il);
                C  = build_norm(C, layer.ssm_c_norm, NULL, LLM_NORM_RMS, il);
            }

            // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
            dt = build_lora_mm(layer.ssm_dt, dt);
            dt = ggml_add(ctx0, dt, layer.ssm_dt_b);

            cur = x;
            x   = ggml_reshape_4d(ctx0, x, head_dim, n_head, n_seq_tokens, n_seqs);

            ggml_tensor * A = layer.ssm_a;

            // use the states and the indices provided by build_recurrent_state
            // (this is necessary in order to properly use the states before they are overwritten,
            //  while avoiding to make unnecessary copies of the states)
            auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
                ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());

                // Custom operator to optimize the parallel associative scan
                // as described in the Annex D of the Mamba paper.
                // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
                return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
            };

            ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);

            // store last states
            ggml_build_forward_expand(
                gf, ggml_cpy(ctx0, ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, x->nb[3] * x->ne[3]),
                             ggml_view_1d(ctx0, ssm_states_all, d_state * d_inner * n_seqs,
                                          kv_head * d_state * d_inner * ggml_element_size(ssm_states_all))));

            ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[2], x->nb[3], 0);

            // TODO: skip computing output earlier for unused tokens

            y = ggml_add(ctx0, y, ggml_mul(ctx0, cur, layer.ssm_d));
            y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

            // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
            cur = build_lora_mm(layer.ssm_out, y);
        }

        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);

        return cur;
    }

    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp,
                                     ggml_tensor *        cur,
                                     const llama_model &  model,
                                     const llama_ubatch & ubatch,
                                     int                  il) const {
        const auto * mctx_cur = inp->mctx;

        const auto kv_head = mctx_cur->get_head();

        const int64_t d_conv   = hparams.ssm_d_conv;
        const int64_t d_inner  = hparams.ssm_d_inner;
        const int64_t d_state  = hparams.ssm_d_state;
        const int64_t head_dim = d_inner / n_head;
        const int64_t n_group  = hparams.ssm_n_group;
        const int64_t n_seqs   = ubatch.n_seqs;

        const int64_t n_seq_tokens = ubatch.n_seq_tokens;

        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
        ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

        ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
        conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs);

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        // d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

        // {n_embd, d_in_proj} @ {n_embd, n_seq_tokens, n_seqs} => {d_in_proj, n_seq_tokens, n_seqs}
        ggml_tensor * zxBCdt = build_lora_mm(model.layers[il].ssm_in, cur);

        // split the above in three
        ggml_tensor * z   = ggml_view_4d(ctx0, zxBCdt, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * zxBCdt->nb[0],
                                         zxBCdt->nb[1], zxBCdt->nb[2], 0);
        ggml_tensor * xBC = ggml_view_3d(ctx0, zxBCdt, d_inner + 2 * n_group * d_state, n_seq_tokens, n_seqs,
                                         zxBCdt->nb[1], zxBCdt->nb[2], d_inner * ggml_element_size(zxBCdt));
        ggml_tensor * dt  = ggml_view_3d(ctx0, zxBCdt, n_head, n_seq_tokens, n_seqs, zxBCdt->nb[1], zxBCdt->nb[2],
                                         (2 * d_inner + 2 * n_group * d_state) * ggml_element_size(zxBCdt));

        // conv
        {
            // => {d_conv - 1 + n_seq_tokens, d_inner + 2*n_group*d_state, n_seqs}
            ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);

            // copy last (d_conv - 1) columns back into the state cache
            ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs,
                                                   conv_x->nb[1], conv_x->nb[2], n_seq_tokens * (conv_x->nb[0]));

            ggml_build_forward_expand(
                gf, ggml_cpy(
                        ctx0, last_conv,
                        ggml_view_1d(ctx0, conv_states_all, (d_conv - 1) * (d_inner + 2 * n_group * d_state) * (n_seqs),
                                     kv_head * (d_conv - 1) * (d_inner + 2 * n_group * d_state) *
                                         ggml_element_size(conv_states_all))));

            // 1D convolution
            // The equivalent is to make a self-overlapping view of conv_x
            // over d_conv columns at each stride in the 3rd dimension,
            // then element-wise multiply that with the conv1d weight,
            // then sum the elements of each row,
            // (the last two steps are a dot product over rows (also doable with mul_mat))
            // then permute away the ne[0] dimension,
            // and then you're left with the resulting x tensor.
            // For simultaneous sequences, all sequences need to have the same length.
            xBC = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);

            // bias
            xBC = ggml_add(ctx0, xBC, model.layers[il].ssm_conv1d_b);

            xBC = ggml_silu(ctx0, xBC);
        }

        // ssm
        {
            // These correspond to V K Q in SSM/attention duality
            ggml_tensor * x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * xBC->nb[0],
                                           xBC->nb[1], xBC->nb[2], 0);
            ggml_tensor * B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                           xBC->nb[1], xBC->nb[2], d_inner * ggml_element_size(xBC));
            ggml_tensor * C =
                ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0], xBC->nb[1],
                             xBC->nb[2], (d_inner + n_group * d_state) * ggml_element_size(xBC));

            // {n_head, n_seq_tokens, n_seqs}
            dt = ggml_add(ctx0, ggml_cont(ctx0, dt), model.layers[il].ssm_dt_b);

            ggml_tensor * A = model.layers[il].ssm_a;

            // use the states and the indices provided by build_recurrent_state
            // (this is necessary in order to properly use the states before they are overwritten,
            //  while avoiding to make unnecessary copies of the states)
            auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
                ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());

                // TODO: use semistructured matrices to implement state-space duality
                // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
                return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
            };

            ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);

            // store last states
            ggml_build_forward_expand(
                gf, ggml_cpy(ctx0, ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, ggml_nelements(x) * x->nb[0]),
                             ggml_view_1d(ctx0, ssm_states_all, d_state * d_inner * n_seqs,
                                          kv_head * d_state * d_inner * ggml_element_size(ssm_states_all))));

            ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x->nb[1],
                                           n_head * x->nb[1], n_seq_tokens * n_head * x->nb[1], 0);

            // TODO: skip computing output earlier for unused tokens

            y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
            y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

            // grouped RMS norm
            if (model.layers[il].ssm_norm) {
                y = ggml_reshape_4d(ctx0, y, d_inner / n_group, n_group, n_seq_tokens, n_seqs);
                y = build_norm(y, model.layers[il].ssm_norm, NULL, LLM_NORM_RMS, il);
            }

            y = ggml_reshape_3d(ctx0, y, d_inner, n_seq_tokens, n_seqs);

            // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
            cur = build_lora_mm(model.layers[il].ssm_out, y);
        }

        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
        cb(cur, "mamba_out", il);

        return cur;
    }
};

// ref: https://allenai.org/olmo
// based on the original build_llama() function, changes:
//   * non-parametric layer norm
//   * clamp qkv
//   * removed bias
//   * removed MoE

// based on the build_qwen2moe() function, changes:
//   * removed shared experts
//   * removed bias
//   * added q, k norm

struct llm_build_ifairy : public llm_graph_context {
    llm_build_ifairy(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * inpL = build_inp_embd(model.tok_embd);

        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv();

        const float kq_scale =
            hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head) / 2) : hparams.f_attention_scale;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            ggml_tensor * cur = ifairy_build_norm(inpL, model.layers[il].attn_norm, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head / 2, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head / 2, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head / 2, n_head_kv, n_tokens);

            Qcur = ggml_ifairy_rope(ctx0, Qcur, inp_pos, n_rot, 0);
            cb(Qcur, "Qcur_rope", il);

            Kcur = ggml_ifairy_rope(ctx0, Kcur, inp_pos, n_rot, 0);
            cb(Kcur, "Kcur_rope", il);

            Vcur = ggml_ifairy_split(ctx0, Vcur);
            cb(Vcur, "Vcur_split", il);

            cur = ifairy_build_attn(inp_attn, Qcur, Kcur, Vcur, kq_scale, il);
            cb(cur, "attn_out", il);

            cur = ifairy_build_norm(cur, model.layers[il].attn_sub_norm, il);
            cb(cur, "attn_post_norm", il);

            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_proj", il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            cur = ggml_ifairy_add(ctx0, cur, inpSA);
            cb(cur, "attn_res", il);

            ggml_tensor * ffn_inp = cur;

            cur = ifairy_build_norm(ffn_inp, model.layers[il].ffn_norm, il);
            cb(cur, "ffn_norm", il);

            cur = ifairy_build_ffn(cur, model.layers[il].ffn_up, model.layers[il].ffn_gate, model.layers[il].ffn_down,
                                   model.layers[il].ffn_sub_norm, il);
            cb(cur, "ffn_out", il);

            cur = ggml_ifairy_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_res", il);

            //cur = build_cvec(cur, il);
            //cb(cur, "l_out", il);

            inpL = cur;
        }

        ggml_tensor * cur = ifairy_build_norm(inpL, model.output_norm, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;
        GGML_ASSERT(model.output);
        ggml_tensor * lm_head = model.output ? model.output : model.tok_embd;
        cur                   = ggml_ifairy_split(ctx0, cur);
        cur                   = build_lora_mm(lm_head, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

// ref: https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1/blob/main/modeling_rwkv6qwen2.py

// ref: https://github.com/facebookresearch/chameleon
// based on the original build_llama() function, changes:
//   * qk-norm
//   * swin-norm
//   * removed bias
//   * removed MoE

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        //case LLM_ARCH_GEMMA_EMBEDDING: // TODO: disabled until the cacheless SWA logic is fixed [TAG_NO_CACHE_ISWA]
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
            {
                res = nullptr;
            } break;
        // Models that need standard caching should rely on recurrent/hybrid
        // checks
        default:
            {
                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch)) {

                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    llama_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    llama_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    if (arch == LLM_ARCH_FALCON_H1) {
                        filter_attn = [&](int32_t) { return true; };
                        filter_recr = [&](int32_t) { return true; };
                    } else if (arch == LLM_ARCH_NEMOTRON_H) {
                        filter_attn = [&](int32_t il) {
                            return !hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [&](int32_t il) {
                            return hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
                        };
                    }

                    const auto padding = llama_kv_cache::get_padding(cparams);

                    cparams.n_ctx = GGML_PAD(cparams.n_ctx, padding);

                    res = new llama_memory_hybrid(
                        /* model             */ *this,
                        /* attn_type_k       */ params.type_k,
                        /* attn_type_v       */ params.type_v,
                        /* attn_v_trans      */ !cparams.flash_attn,
                        /* attn_kv_size      */ cparams.n_ctx,
                        /* attn_n_pad        */ padding,
                        /* attn_n_swa        */ hparams.n_swa,
                        /* attn_swa_type     */ hparams.swa_type,
                        /* recurrent_type_k  */ GGML_TYPE_F32,
                        /* recurrent_type_v  */ GGML_TYPE_F32,
                        /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                        /* n_seq_max         */ cparams.n_seq_max,
                        /* offload           */ cparams.offload_kqv,
                        /* unified           */ cparams.kv_unified,
                        /* filter_attn       */ std::move(filter_attn),
                        /* filter_recr       */ std::move(filter_recr));
                } else {
                    const auto padding = llama_kv_cache::get_padding(cparams);

                    uint32_t n_ctx_per_stream = cparams.n_ctx;

                    if (!cparams.kv_unified) {
                        n_ctx_per_stream = (cparams.n_ctx + cparams.n_seq_max - 1)/cparams.n_seq_max;
                        n_ctx_per_stream = GGML_PAD(n_ctx_per_stream, padding);

                        cparams.n_ctx = n_ctx_per_stream*cparams.n_seq_max;
                    } else {
                        n_ctx_per_stream = GGML_PAD(n_ctx_per_stream, padding);

                        cparams.n_ctx = n_ctx_per_stream;
                    }

                    uint32_t kv_size = n_ctx_per_stream;
                    if (cparams.prefix_sliding_window > 0 && cparams.prefix_sliding_prefix_cap > 0) {
                        const uint64_t bounded = (uint64_t) cparams.prefix_sliding_prefix_cap +
                                                 cparams.prefix_sliding_window - 1 + cparams.n_ubatch;
                        kv_size = GGML_PAD((uint32_t) std::min<uint64_t>(n_ctx_per_stream, bounded), padding);
                    }

                    LLAMA_LOG_DEBUG("%s: n_ctx = %u (padded)\n", __func__, cparams.n_ctx);

                    llama_memory_i::layer_reuse_cb reuse = nullptr;

                    if (arch == LLM_ARCH_GEMMA3N) {
                        reuse = [&](int32_t il) {
                            if (il >= (int32_t) hparams.n_layer_kv_from_start) {
                                return (int32_t) hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                            }

                            return -1;
                        };
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        res = new llama_kv_cache_iswa(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                params.swa_full,
                                cparams.kv_unified,
                                kv_size,
                                cparams.n_seq_max,
                                cparams.n_ubatch,
                                padding,
                                nullptr,
                                reuse);
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new llama_kv_cache(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                kv_size,
                                cparams.n_seq_max,
                                padding,
                                hparams.n_swa,
                                hparams.swa_type,
                                cparams.prefix_sliding_window,
                                cparams.prefix_sliding_prefix_cap,
                                cparams.n_ubatch,
                                nullptr,
                                nullptr);
                    }
                }
            }
    }

    return res;
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm;

    switch (arch) {
        case LLM_ARCH_IFAIRY:
            {
                llm = std::make_unique<llm_build_ifairy>(*this, params);
            }
            break;
        default:
            throw std::runtime_error("this repository supports only the ifairy architecture");
    }

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b);

    return llm->res->get_gf();
}


//
// interface implementation
//

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ 999,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
    };

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_free_model(llama_model * model) {
    llama_model_free(model);
}

void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    return model->hparams.n_swa;
}

uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t llama_n_ctx_train(const llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const llama_model * model) {
    return llama_model_n_head(model);
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_NEMOTRON_H:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_PLM:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GLM4:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_IFAIRY:

        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_ARCEE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_SEED_OSS:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
            return LLAMA_ROPE_TYPE_MROPE;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

uint64_t llama_model_size(const llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:        return true;
        case LLM_ARCH_T5ENCODER: return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}
