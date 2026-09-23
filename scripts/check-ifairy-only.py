#!/usr/bin/env python3
"""Check that production sources contain no removed architecture implementation."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
errors = []
for directory in ('src', 'include', 'common', 'ggml', 'gguf-py'):
    for path in (root / directory).rglob('*'):
        if not path.is_file() or path.suffix not in {'.h', '.c', '.cpp', '.py', '.txt', '.toml'}:
            continue
        if re.search('fairy2i', path.read_text(errors='replace'), re.IGNORECASE):
            errors.append(str(path.relative_to(root)))
builders = re.findall(r'^struct (llm_build_\w+)\s*:', (root / 'src/llama-model.cpp').read_text(), re.MULTILINE)
if builders != ['llm_build_ifairy']:
    errors.append(f'unexpected graph builders: {builders}')
for path in ('ggml/src/ggml-cpu/fairy2i', 'gguf-py/fairy2i', 'ggml/src/ggml-metal', 'ggml/src/ggml-opencl'):
    if (root / path).exists():
        errors.append(f'unexpected implementation directory: {path}')
if errors:
    print('\n'.join(errors), file=sys.stderr)
    sys.exit(1)
print('PASS: no Fairy2i production implementation; only the iFairy graph builder remains.')
