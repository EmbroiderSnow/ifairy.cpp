"""Real-checkpoint probe for the llama.h ABI at revision a6256d6 (Linux x86_64).

Usage: python3 check_logits.py BUILD_DIR MODEL.gguf [batched|serial]
Uses the standard library only; does not modify the checkpoint or runtime.
Set GGML_IFAIRY_LUT and GGML_IFAIRY_LUT_IMPL before starting the process.
"""

import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys


class ModelParams(C.Structure):
    _fields_ = [
        ('devices', C.c_void_p), ('tensor_buft_overrides', C.c_void_p),
        ('n_gpu_layers', C.c_int32), ('split_mode', C.c_int), ('main_gpu', C.c_int32),
        ('tensor_split', C.c_void_p), ('progress_callback', C.c_void_p),
        ('progress_callback_user_data', C.c_void_p), ('kv_overrides', C.c_void_p),
        *[(name, C.c_bool) for name in
          ('vocab_only', 'use_mmap', 'use_mlock', 'check_tensors', 'use_extra_bufts')],
    ]


class ContextParams(C.Structure):
    _fields_ = [
        *[(name, C.c_uint32) for name in ('n_ctx', 'n_batch', 'n_ubatch', 'n_seq_max',
                                        'prefix_sliding_window', 'prefix_sliding_prefix_cap')],
        ('n_threads', C.c_int32), ('n_threads_batch', C.c_int32),
        *[(name, C.c_int) for name in ('rope_scaling_type', 'pooling_type',
                                     'attention_type', 'flash_attn_type')],
        *[(name, C.c_float) for name in ('rope_freq_base', 'rope_freq_scale', 'yarn_ext_factor',
                                       'yarn_attn_factor', 'yarn_beta_fast', 'yarn_beta_slow')],
        ('yarn_orig_ctx', C.c_uint32), ('defrag_thold', C.c_float),
        ('cb_eval', C.c_void_p), ('cb_eval_user_data', C.c_void_p),
        ('type_k', C.c_int), ('type_v', C.c_int),
        ('abort_callback', C.c_void_p), ('abort_callback_data', C.c_void_p),
        *[(name, C.c_bool) for name in ('embeddings', 'offload_kqv', 'no_perf',
                                      'op_offload', 'swa_full', 'kv_unified')],
    ]


TokenPtr = C.POINTER(C.c_int32)


class Batch(C.Structure):
    _fields_ = [('n_tokens', C.c_int32), ('token', TokenPtr), ('embd', C.c_void_p),
                ('pos', TokenPtr), ('n_seq_id', TokenPtr),
                ('seq_id', C.POINTER(TokenPtr)), ('logits', C.c_void_p)]


def bind(lib, name, result, *args):
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = args
    return fn


def main():
    build, model_path = sys.argv[1:3]
    mode = sys.argv[3] if len(sys.argv) > 3 else 'batched'
    if mode not in ('batched', 'serial'):
        raise ValueError(mode)
    lib = C.CDLL(str(Path(build).resolve() / 'bin/libllama.so'))
    bind(lib, 'llama_backend_init', None)()
    bind(lib, 'llama_backend_free', None)
    bind(lib, 'llama_model_default_params', ModelParams)
    bind(lib, 'llama_context_default_params', ContextParams)
    bind(lib, 'llama_model_load_from_file', C.c_void_p, C.c_char_p, ModelParams)
    bind(lib, 'llama_model_free', None, C.c_void_p)
    bind(lib, 'llama_model_get_vocab', C.c_void_p, C.c_void_p)
    bind(lib, 'llama_vocab_n_tokens', C.c_int32, C.c_void_p)
    bind(lib, 'llama_init_from_model', C.c_void_p, C.c_void_p, ContextParams)
    bind(lib, 'llama_free', None, C.c_void_p)
    bind(lib, 'llama_tokenize', C.c_int32, C.c_void_p, C.c_char_p, C.c_int32,
         TokenPtr, C.c_int32, C.c_bool, C.c_bool)
    bind(lib, 'llama_batch_get_one', Batch, TokenPtr, C.c_int32)
    bind(lib, 'llama_decode', C.c_int32, C.c_void_p, Batch)
    bind(lib, 'llama_get_logits_ith', C.POINTER(C.c_float), C.c_void_p, C.c_int32)

    mp = lib.llama_model_default_params()
    mp.n_gpu_layers = 0
    model = lib.llama_model_load_from_file(os.fsencode(model_path), mp)
    if not model:
        raise RuntimeError('model load failed')
    results = []
    try:
        vocab = lib.llama_model_get_vocab(model)
        n_vocab = lib.llama_vocab_n_tokens(vocab)
        if n_vocab != 32000:
            raise RuntimeError(f'unexpected vocabulary size {n_vocab}')
        for prompt_index, prompt in enumerate(('The capital of France is', 'Once upon a time, in a small village')):
            cp = lib.llama_context_default_params()
            cp.n_ctx, cp.n_batch, cp.n_ubatch = 2048, 512, 512
            cp.n_threads = cp.n_threads_batch = 8
            cp.flash_attn_type = 0
            cp.offload_kqv = cp.op_offload = False
            ctx = lib.llama_init_from_model(model, cp)
            if not ctx:
                raise RuntimeError('context creation failed')
            try:
                encoded = prompt.encode()
                tokens = (C.c_int32 * 512)()
                count = lib.llama_tokenize(vocab, encoded, len(encoded), tokens, 512, True, False)
                if not 0 < count <= 512:
                    raise RuntimeError(f'tokenization failed: {count}')
                prompt_ids = list(tokens[:count])

                def decode(ids):
                    data = (C.c_int32 * len(ids))(*ids)
                    rc = lib.llama_decode(ctx, lib.llama_batch_get_one(data, len(ids)))
                    if rc:
                        raise RuntimeError(f'decode returned {rc}')

                if mode == 'batched':
                    decode(prompt_ids)
                else:
                    for token in prompt_ids:
                        decode([token])
                digest = hashlib.sha256()
                generated = []
                low, high = math.inf, -math.inf
                for step in range(33):
                    ptr = lib.llama_get_logits_ith(ctx, -1)
                    if not ptr:
                        raise RuntimeError('missing logits')
                    values = ptr[:n_vocab]
                    if not all(map(math.isfinite, values)):
                        raise RuntimeError(f'non-finite logits at step {step}')
                    digest.update(C.string_at(ptr, n_vocab * C.sizeof(C.c_float)))
                    if step == 0:
                        prefill_top5 = sorted(range(n_vocab), key=values.__getitem__, reverse=True)[:5]
                        prefill_top5 = [[token, values[token]] for token in prefill_top5]
                        dump_dir = os.getenv('IFAIRY_PROBE_DUMP_DIR')
                        if dump_dir:
                            Path(dump_dir).mkdir(parents=True, exist_ok=True)
                            Path(dump_dir, f'prompt-{prompt_index}.f32').write_bytes(
                                C.string_at(ptr, n_vocab * C.sizeof(C.c_float)))
                    low, high = min(low, min(values)), max(high, max(values))
                    if step < 32:
                        token = max(range(n_vocab), key=values.__getitem__)
                        generated.append(token)
                        decode([token])
                results.append(dict(prompt=prompt, prompt_ids=prompt_ids, generated_ids=generated,
                                    logit_vectors=33, finite_logits=33*n_vocab,
                                    prefill_top5=prefill_top5,
                                    logits_sha256=digest.hexdigest(), min_logit=low, max_logit=high))
            finally:
                lib.llama_free(ctx)
    finally:
        lib.llama_model_free(model)
        lib.llama_backend_free()
    print(json.dumps(dict(build=build, mode=mode, n_vocab=n_vocab,
                          lut=os.getenv('GGML_IFAIRY_LUT'), impl=os.getenv('GGML_IFAIRY_LUT_IMPL'),
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          results=results), indent=2))


if __name__ == '__main__':
    main()
