# eaglefix = :latest + Qwen4ExpMTP eagle-group annotation fix
# (prefix-cache reuse for hybrid Qwen4Exp + MTP). Python-only overlay;
# kernels unchanged. Base image: localhost/qfn38-fp8-4xb70:latest.
FROM localhost/qfn38-fp8-4xb70:latest
COPY fs-overlay/vllm/v1/core/kv_cache_utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
RUN if [ -d /workspace/vllm/vllm ]; then \
      cp /opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py /workspace/vllm/vllm/v1/core/kv_cache_utils.py; fi
RUN python3 -c "import ast; ast.parse(open('/opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py').read()); print('overlay parses')"
COPY test_eagle_annotation.py /opt/eaglefix/test_eagle_annotation.py
