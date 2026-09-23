// End-to-end iFairy load/prefill/decode and architecture-isolation regression.
#include "../ggml/src/ggml-quants.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct fixture {
    gguf_context *                    meta = gguf_init_empty();
    ggml_context *                    ctx  = ggml_init({ 32 * ggml_tensor_overhead(), nullptr, true });
    std::vector<std::vector<uint8_t>> data;
    std::filesystem::path             path =
        std::filesystem::temp_directory_path() /
        ("ifairy-model-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");

    fixture() {
        gguf_set_val_str(meta, "general.architecture", "ifairy");
        gguf_set_val_u32(meta, "ifairy.context_length", 32);
        gguf_set_val_u32(meta, "ifairy.embedding_length", 256);
        gguf_set_val_u32(meta, "ifairy.block_count", 1);
        gguf_set_val_u32(meta, "ifairy.feed_forward_length", 256);
        gguf_set_val_u32(meta, "ifairy.attention.head_count", 4);
        gguf_set_val_u32(meta, "ifairy.attention.head_count_kv", 4);
        gguf_set_val_f32(meta, "ifairy.attention.layer_norm_epsilon", 1e-5f);
        gguf_set_val_u32(meta, "ifairy.vocab_size", 32);
        gguf_set_val_str(meta, "tokenizer.ggml.model", "no_vocab");
        add("token_embd", GGML_TYPE_F32, 256, 32, true);
        add("output", GGML_TYPE_F16, 512, 32);
        add("output_norm", GGML_TYPE_F32, 512);
        for (const char * name : { "attn_norm", "attn_sub_norm", "ffn_norm", "ffn_sub_norm" }) {
            add((std::string("blk.0.") + name).c_str(), GGML_TYPE_F32, 512);
        }
        for (const char * name : { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" }) {
            add((std::string("blk.0.") + name).c_str(), GGML_TYPE_IFAIRY, 256, 256);
        }
    }

    ~fixture() {
        ggml_free(ctx);
        gguf_free(meta);
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    void add(const char * name, ggml_type type, int64_t k, int64_t m = 1, bool packed = false) {
        auto * tensor = ggml_new_tensor_2d(ctx, type, k, m);
        ggml_set_name(tensor, name);
        gguf_add_tensor(meta, tensor);
        data.emplace_back(ggml_nbytes(tensor), 0);
        auto & bytes = data.back();
        if (type == GGML_TYPE_IFAIRY) {
            for (size_t i = 0; i < bytes.size() / sizeof(block_ifairy); ++i) {
                block_ifairy block{};
                block.d_real = ggml_fp32_to_fp16(0.002f);
                block.d_imag = ggml_fp32_to_fp16(0.003f);
                for (size_t j = 0; j < sizeof(block.qs); ++j) {
                    block.qs[j] = static_cast<uint8_t>((i * 37 + j * 19) & 255);
                }
                memcpy(bytes.data() + i * sizeof(block), &block, sizeof(block));
            }
        } else if (packed) {
            for (int64_t i = 0; i < k * m; ++i) {
                ggml_bf16_t pair[] = { ggml_fp32_to_bf16(0.01f * (1 + i % 17)),
                                       ggml_fp32_to_bf16(0.02f * (1 + i % 11)) };
                memcpy(bytes.data() + i * sizeof(float), pair, sizeof(pair));
            }
        } else if (type == GGML_TYPE_F32) {
            for (int64_t i = 0; i < k * m; ++i) {
                float value = 1.0f;
                memcpy(bytes.data() + i * sizeof(float), &value, sizeof(value));
            }
        } else {
            for (int64_t i = 0; i < k * m; ++i) {
                auto value = ggml_fp32_to_fp16(0.001f * (static_cast<int>(i % 23) - 11));
                memcpy(bytes.data() + i * sizeof(value), &value, sizeof(value));
            }
        }
        gguf_set_tensor_data(meta, name, bytes.data());
    }

    void write(const char * arch) {
        gguf_set_val_str(meta, "general.architecture", arch);
        require(gguf_write_to_file(meta, path.string().c_str(), false), "fixture write failed");
    }
};

static std::vector<float> infer(llama_model * model, bool batched) {
    auto params            = llama_context_default_params();
    params.n_ctx           = 32;
    params.n_batch         = 8;
    params.n_ubatch        = 8;
    params.n_threads       = 2;
    params.n_threads_batch = 2;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto * ctx             = llama_init_from_model(model, params);
    require(ctx != nullptr, "context creation failed");
    llama_token tokens[]    = { 1, 2, 3, 4 };
    const int   prompt_size = batched ? 3 : 1;
    for (int i = 0; i < 3; i += prompt_size) {
        require(llama_decode(ctx, llama_batch_get_one(tokens + i, prompt_size)) == 0, "prefill failed");
    }
    require(llama_decode(ctx, llama_batch_get_one(tokens + 3, 1)) == 0, "decode failed");
    const auto * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "missing logits");
    std::vector<float> result(logits, logits + 32);
    llama_free(ctx);
    for (float value : result) {
        require(std::isfinite(value), "non-finite logits");
    }
    return result;
}

int main() {
    try {
        ggml_backend_load_all();
        llama_backend_init();
        fixture f;
        auto    params      = llama_model_default_params();
        params.n_gpu_layers = 0;
        f.write("ifairy");
        auto * model = llama_model_load_from_file(f.path.string().c_str(), params);
        require(model != nullptr, "iFairy load failed");
        auto  batched  = infer(model, true);
        auto  serial   = infer(model, false);
        auto  repeated = infer(model, true);
        float max_diff = 0.0f;
        for (size_t i = 0; i < batched.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(batched[i] - serial[i]));
            require(batched[i] == repeated[i], "same-path output is not deterministic");
        }
        require(max_diff < 0.05f, "prefill/decode logits disagree");
        llama_model_free(model);
        for (const char * arch : { "fairy2i", "llama", "qwen3", "unknown" }) {
            f.write(arch);
            auto * rejected = llama_model_load_from_file(f.path.string().c_str(), params);
            if (rejected) {
                llama_model_free(rejected);
                throw std::runtime_error("non-iFairy model was accepted");
            }
        }
        llama_backend_free();
        printf("iFairy model load, prefill, decode, repeatability and isolation passed; max difference %.8f\n",
               max_diff);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
