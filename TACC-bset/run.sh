#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_amp \
    --amp_dtype auto \
    --use_compile \
    --compile_mode default \
    --compile_disable_cudagraphs \
    --use_calendar_time_features \
    --query_generator_type ui_aware \
    --ui_shortcut_type bi_cross_attn \
    --ui_shortcut_alpha_init 1.0 \
    --dense_tokenizer_type field_attn \
    --reinit_sparse_after_epoch 1 \
    --reinit_cardinality_threshold 0 \
    --first_decoded_shortcut_type mixer \
    --first_decoded_alpha_init 0.2 \
    --target_attn_type din \
    --seq_max_lens "seq_a:512,seq_b:512,seq_c:512,seq_d:1024" \
    --use_seq_time_stats \
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
