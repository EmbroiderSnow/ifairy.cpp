"""Sample actual model MUL_MAT outputs against an independent complex-sum oracle.

Uses the repository's Linux x86_64 ABI and existing LUT quantizer, then decodes
weight codes and accumulates w * conj(x) in Python double precision. Run with
GGML_IFAIRY_LUT=1 and GGML_IFAIRY_LUT_IMPL=auto. Diagnostic callbacks change graph
partitioning, so this run is never used for timing.
"""
import ctypes as C
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys

spec = importlib.util.spec_from_file_location('probe_abi', Path(__file__).parents[1] / 'local-x86-20260923/check_logits.py')
abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abi)
bind = abi.bind

class Tensor(C.Structure):
    pass
TP = C.POINTER(Tensor)
Tensor._fields_ = [('type', C.c_int), ('buffer', C.c_void_p), ('ne', C.c_int64 * 4), ('nb', C.c_size_t * 4),
                  ('op', C.c_int), ('op_params', C.c_int32 * 16), ('flags', C.c_int32), ('src', TP * 10),
                  ('view_src', TP), ('view_offs', C.c_size_t), ('data', C.c_void_p), ('name', C.c_char * 64),
                  ('extra', C.c_void_p), ('padding', C.c_char * 8)]
lib = C.CDLL(str(Path(sys.argv[1]).resolve() / 'bin/libllama.so'))
bind(lib, 'ggml_op_name', C.c_char_p, C.c_int)
bind(lib, 'quantize_row_ifairy_q16_tensor', None, C.c_void_p, C.c_void_p, C.c_int64)
rows = []
failures = []

def bf16(word):
    return struct.unpack('<f', struct.pack('<I', word << 16))[0]

def callback(ptr, ask, unused):
    try:
        t = ptr.contents
        selected = lib.ggml_op_name(t.op) == b'MUL_MAT' and t.src[0] and t.src[0].contents.type == 40
        if ask:
            return bool(selected)
        w, x = t.src[0].contents, t.src[1].contents
        m, k, n = w.ne[1], w.ne[0], x.ne[1]
        assert x.type == 0 and k % 256 == 0 and t.nb[0] == 4
        block_count = k // 256
        maximum = 0.0
        worst_ratio = 0.0
        count = 0
        scale_changes = 0
        for col in sorted({0, n - 1}):
            qbuf = C.create_string_buffer(block_count * 516)
            lib.quantize_row_ifairy_q16_tensor(x.data + col * x.nb[1], qbuf, k)
            qbytes = qbuf.raw
            for row in sorted({0, m // 2, m - 1}):
                wbytes = C.string_at(w.data + row * w.nb[1], block_count * 68)
                real_terms, imag_terms = [], []
                for b in range(block_count):
                    wb = wbytes[b*68:(b+1)*68]
                    qb = qbytes[b*516:(b+1)*516]
                    wr, wi = struct.unpack('<ee', wb[64:68])
                    xr, xi = struct.unpack('<ee', qb[512:516])
                    scale_changes += wb[64:68] != wbytes[64:68]
                    for j in range(256):
                        code = (wb[(j//64)*16 + (j&15)] >> (2*((j>>4)&3))) & 3
                        a = -wr if code == 0 else wr if code == 1 else 0.0
                        b_im = -wi if code == 2 else wi if code == 3 else 0.0
                        qr, qi = qb[j], qb[256+j]
                        c = (qr if qr < 128 else qr-256) * xr
                        d = (qi if qi < 128 else qi-256) * xi
                        real_terms.append(a*c + b_im*d)
                        imag_terms.append(b_im*c - a*d)
                actual = struct.unpack('<HH', C.string_at(t.data + col*t.nb[1] + row*4, 4))
                for actual_word, reference in zip(actual, (math.fsum(real_terms), math.fsum(imag_terms))):
                    value = bf16(actual_word)
                    assert math.isfinite(value)
                    error = abs(value-reference)
                    maximum = max(maximum, error)
                    worst_ratio = max(worst_ratio, error/(0.008*abs(reference)+1e-5))
                    count += 1
        row = dict(name=t.name.decode(), M=m, N=n, K=k, components=count, max_abs=maximum,
                   max_tolerance_ratio=worst_ratio, weight_scale_changes=scale_changes)
        rows.append(row)
        if worst_ratio > 1:
            failures.append(row)
        return True
    except Exception as error:
        failures.append({'error':repr(error)})
        return False

cb = C.CFUNCTYPE(C.c_bool, TP, C.c_bool, C.c_void_p)(callback)
bind(lib, 'llama_backend_init', None)()
bind(lib, 'llama_backend_free', None)
bind(lib, 'llama_model_default_params', abi.ModelParams)
bind(lib, 'llama_context_default_params', abi.ContextParams)
bind(lib, 'llama_model_load_from_file', C.c_void_p, C.c_char_p, abi.ModelParams)
bind(lib, 'llama_model_free', None, C.c_void_p)
bind(lib, 'llama_init_from_model', C.c_void_p, C.c_void_p, abi.ContextParams)
bind(lib, 'llama_free', None, C.c_void_p)
bind(lib, 'llama_model_get_vocab', C.c_void_p, C.c_void_p)
bind(lib, 'llama_tokenize', C.c_int32, C.c_void_p, C.c_char_p, C.c_int32, abi.TokenPtr, C.c_int32, C.c_bool, C.c_bool)
bind(lib, 'llama_batch_get_one', abi.Batch, abi.TokenPtr, C.c_int32)
bind(lib, 'llama_decode', C.c_int32, C.c_void_p, abi.Batch)
bind(lib, 'llama_get_logits_ith', C.POINTER(C.c_float), C.c_void_p, C.c_int32)
mp = lib.llama_model_default_params(); mp.n_gpu_layers = 0
model = lib.llama_model_load_from_file(os.fsencode(sys.argv[2]), mp)
assert model
cp = lib.llama_context_default_params()
cp.n_ctx, cp.n_batch, cp.n_ubatch = 2048, 512, 512
cp.n_threads = cp.n_threads_batch = 8
cp.flash_attn_type = 0
cp.offload_kqv = cp.op_offload = False
cp.cb_eval = C.cast(cb, C.c_void_p).value
ctx = lib.llama_init_from_model(model, cp)
assert ctx
try:
    prompt = b'The capital of France is'
    tokens = (C.c_int32*512)()
    n = lib.llama_tokenize(lib.llama_model_get_vocab(model), prompt, len(prompt), tokens, 512, True, False)
    assert n > 0
    assert lib.llama_decode(ctx, lib.llama_batch_get_one(tokens, n)) == 0
    logits = lib.llama_get_logits_ith(ctx, -1)[:32000]
    token = (C.c_int32*1)(max(range(32000), key=logits.__getitem__))
    assert lib.llama_decode(ctx, lib.llama_batch_get_one(token, 1)) == 0
finally:
    lib.llama_free(ctx); lib.llama_model_free(model); lib.llama_backend_free()
print(json.dumps({'matmuls':len(rows), 'components':sum(x['components'] for x in rows), 'failures':failures,
                  'max_tolerance_ratio':max(x['max_tolerance_ratio'] for x in rows), 'rows':rows}, indent=2))
assert len(rows) == 336 and not failures
