# Stage 1: official vLLM 0.30.0 XPU image + the mmap-PLE python tree
# (with the QSA XPU bring-up fixes baked in) + hash-locked Intel GPU
# driver 26.35.39758.10 (the host-mirroring-bug fix).
# BUILD CONTEXT: the package root (build.sh invokes it from there).
FROM docker.io/vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9

COPY driver-debs/ /driver-debs/
COPY driver-packages.lock.json /driver-packages.lock.json
COPY install_driver.py /install_driver.py
RUN python3 /install_driver.py /driver-debs /driver-packages.lock.json

COPY base-image/fs/vllm/ /opt/venv/lib/python3.12/site-packages/vllm/
# The official image keeps a source tree at /workspace/vllm that shadows
# site-packages at runtime — overlay it too so both copies match.
RUN if [ -d /workspace/vllm/vllm ]; then cp -a /opt/venv/lib/python3.12/site-packages/vllm/. /workspace/vllm/vllm/; fi

# sanity: every python file parses
RUN python3 - <<'PY2'
import ast, pathlib
n = 0
for root in (pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm"),
             pathlib.Path("/workspace/vllm/vllm")):
    if root.exists():
        for p in root.rglob("*.py"):
            ast.parse(p.read_text()); n += 1
print("parsed", n, "python files")
PY2
