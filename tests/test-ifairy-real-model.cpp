// Optional real-checkpoint probe: finite logits, reproducibility and prefill mode comparison.
#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#    include <sys/resource.h>
#endif

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static nlohmann::json probe(llama_model * model, const std::string & prefix, bool serial, int threads) {
    const auto * vocab   = llama_model_get_vocab(model);
    const int    n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 0, "empty vocabulary");
    nlohmann::json results      = nlohmann::json::array();
    int            prompt_index = 0;
    for (const std::string prompt : { "The capital of France is", "Once upon a time, in a small village" }) {
        auto params            = llama_context_default_params();
        params.n_ctx           = 2048;
        params.n_batch         = 512;
        params.n_ubatch        = 512;
        params.n_threads       = threads;
        params.n_threads_batch = threads;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.offload_kqv     = false;
        params.op_offload      = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free);
        require(ctx != nullptr, "context creation failed");
        std::vector<llama_token> tokens(512);
        const int count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(),
                                         static_cast<int32_t>(tokens.size()), true, false);
        require(count > 0 && count <= static_cast<int>(tokens.size()), "tokenization failed");
        tokens.resize(count);
        const int chunk = serial ? 1 : count;
        for (int i = 0; i < count; i += chunk) {
            require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data() + i, chunk)) == 0, "prefill failed");
        }
        const std::string path = prefix + "-prompt-" + std::to_string(prompt_index++) + ".f32";
        std::ofstream     dump(path, std::ios::binary);
        require(dump.is_open(), "cannot open logits dump");
        std::vector<llama_token> generated;
        float                    low  = std::numeric_limits<float>::infinity();
        float                    high = -std::numeric_limits<float>::infinity();
        for (int step = 0; step <= 32; ++step) {
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            require(logits != nullptr, "missing logits");
            for (int i = 0; i < n_vocab; ++i) {
                require(std::isfinite(logits[i]), "non-finite logits");
                low  = std::min(low, logits[i]);
                high = std::max(high, logits[i]);
            }
            dump.write(reinterpret_cast<const char *>(logits), static_cast<std::streamsize>(n_vocab) * sizeof(float));
            require(dump.good(), "logits write failed");
            if (step < 32) {
                auto token = static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
                generated.push_back(token);
                require(llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) == 0, "decode failed");
            }
        }
        dump.close();
        require(!dump.fail(), "logits close failed");
        results.push_back({
            { "prompt",        prompt         },
            { "prompt_ids",    tokens         },
            { "generated_ids", generated      },
            { "logit_vectors", 33             },
            { "finite_logits", 33LL * n_vocab },
            { "min_logit",     low            },
            { "max_logit",     high           },
            { "logits_file",   path           }
        });
    }
    return results;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL.gguf DUMP_PREFIX batched|serial THREADS\n", argv[0]);
        return 1;
    }
    try {
        const std::string mode = argv[3];
        require(mode == "batched" || mode == "serial", "invalid prefill mode");
        const int threads = std::stoi(argv[4]);
        require(threads > 0, "threads must be positive");
        ggml_backend_load_all();
        llama_backend_init();
        nlohmann::json output;
        {
            auto params         = llama_model_default_params();
            params.n_gpu_layers = 0;
            std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], params),
                                                                            llama_model_free);
            require(model != nullptr, "model load failed");
            output = {
                { "mode", mode },
                { "threads", threads },
                { "n_vocab", llama_vocab_n_tokens(llama_model_get_vocab(model.get())) },
                { "results", probe(model.get(), argv[2], mode == "serial", threads) }
            };
        }
#ifndef _WIN32
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            output["max_rss_native"] = usage.ru_maxrss;  // KiB on Linux/Android; bytes on macOS.
        }
#endif
        llama_backend_free();
        printf("%s\n", output.dump(2).c_str());
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "real-model probe failed: %s\n", error.what());
        llama_backend_free();
        return 1;
    }
}
