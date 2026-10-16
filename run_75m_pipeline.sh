#!/bin/bash
# Train the ~75M hybrid (Transformer + Mythos) on every corpus we have, then
# chat fine-tune it on every conversation/SFT set. Idempotent: re-running skips
# finished phases, and the pretrain phase resumes from its latest checkpoint.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$(dirname "$0")/.venv/bin/python" ]; then
    PY="$(dirname "$0")/.venv/bin/python"
  elif [ -x "$(dirname "$0")/../.venv/bin/python" ]; then
    PY="$(dirname "$0")/../.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
fi
PRETRAIN_OUT=checkpoints/hybrid/hybrid-75m
CHAT_OUT=checkpoints/hybrid/hybrid-75m-chat
ARCH="--dim 768 --heads 12 --kv-heads 3 --hidden-dim 1920 --prelude-layers 3 --stages 3 --coda-layers 2 --max-loop-iters 4"

if [ ! -f "$PRETRAIN_OUT/final.pt" ]; then
  RESUME=""
  [ -f "$PRETRAIN_OUT/latest.pt" ] && RESUME="--resume $PRETRAIN_OUT/latest.pt"
  echo "=== Phase 1: pretrain ~75M hybrid on ALL pretraining text (data/pretrain-all) - $(date) ${RESUME:+[$RESUME]}"
  # shellcheck disable=SC2086
  $PY -u hybrid/train_hybrid.py \
    --train-text data/pretrain-all/train.txt \
    --validation-text data/pretrain-all/validation.txt \
    --tokenizer data/scratch_tokenizer.json \
    --output "$PRETRAIN_OUT" \
    --steps 59221 --seq-len 128 \
    --save-interval 1000 --eval-interval 2000 \
    --threads 8 $ARCH $RESUME
else
  echo "=== Phase 1 already complete, skipping - $(date)"
fi

if [ ! -f "$CHAT_OUT/final.pt" ]; then
  echo "=== Phase 2: chat fine-tune on ALL 9 conversation/SFT sets - $(date)"
  $PY -u hybrid/finetune_hybrid.py \
    --data data/ultrachat_10000.jsonl \
    --data data/ultrachat.jsonl \
    --data data/grounded_qa_mix.jsonl \
    --data data/conversation_mix.jsonl \
    --data data/conversation_focused.jsonl \
    --data data/chat_sft.jsonl \
    --data data/chat_sft_greet.jsonl \
    --data examples/basic_qa.jsonl \
    --data examples/conversation_core.jsonl \
    --init-from "$PRETRAIN_OUT/final.pt" \
    --output "$CHAT_OUT" \
    --steps 6000 --seq-len 224 --max-examples 6000 --shuffle \
    --save-interval 500 --threads 8
else
  echo "=== Phase 2 already complete, skipping - $(date)"
fi

echo "=== ALL DONE - $(date)"
echo "Chat:  python hybrid/chat_hybrid.py --model $CHAT_OUT/final.pt"
echo "Agent: python agent/chat_agent_hybrid.py --model $CHAT_OUT/final.pt"
