#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# PyTorch's scaled_dot_product_attention can use native FlashAttention kernels
# without the third-party flash-attn package when CUDA/GPU/dtype support it.
# The optional block below only installs flash-attn if a compatible wheel was
# uploaded beside this script or into the platform share directory.
install_optional_flash_attn() {
    local wheel=""
    for d in "${SCRIPT_DIR}" "${RUNTIME_SCRIPT_SHARE_DIR:-}"; do
        if [ -n "$d" ] && [ -d "$d" ]; then
            wheel="$(find "$d" -maxdepth 1 -type f -name 'flash_attn*.whl' | head -n 1)"
            if [ -n "$wheel" ]; then
                echo "[run.sh] Installing uploaded flash-attn wheel: $wheel"
                python3 -m pip install --user --no-index --no-deps "$wheel" || true
                export PYTHONPATH="$(python3 -m site --user-site):${PYTHONPATH}"
                return
            fi
        fi
    done

    if [ "${INSTALL_FLASH_ATTN_FROM_PIP:-0}" = "1" ]; then
        echo "[run.sh] INSTALL_FLASH_ATTN_FROM_PIP=1, trying pip install flash-attn"
        python3 -m pip install --user --no-build-isolation flash-attn || true
        export PYTHONPATH="$(python3 -m site --user-site):${PYTHONPATH}"
    fi
}

install_optional_flash_attn
python3 -c "import importlib.util, torch; print('[run.sh] torch', torch.__version__, 'cuda', torch.version.cuda, 'flash_attn_pkg', importlib.util.find_spec('flash_attn') is not None)"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --use_amp \
    --amp_dtype auto \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --num_epochs 2 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
