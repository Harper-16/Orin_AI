# AE-Assistant — Project Info

**What:** tiny CPU-first chat LMs trained from scratch + a Wikipedia-grounded agent.
**Where:** `AE-Assistant/` is the repo root (code lives here, not in `Model-Az/`).
**Status:** research / learning project — small models make mistakes on unfamiliar Q&A.

## Models

| Path | Params | Idea |
|------|--------|------|
| `scratch/` | ~15.7M | plain decoder Transformer (RMSNorm, SwiGLU, RoPE, GQA) |
| `mythos/` | ~17M | recurrent-depth "thinking loops" + self-verifying rethink |
| `hybrid/` | 17M / 75M | Transformer + Mythos fused per-stage with learned gate |
| `qwen_lora/` | 0.6B LoRA | optional Qwen3-0.6B adapter path (HF weights, own license) |
| `agent/` | — | intent router + Wikipedia (live) / SQuAD index (offline) + grounding check |

## Layout
agent/ chat_agent*.py, intent_router.py, wiki_agent.py, wiki_tools.py
common/ conversation_helpers.py, knowledge_store.py
data_prep/ download_*.py, prepare_*.py, build_*.py
scratch/ | mythos/ | hybrid/ model.py + train_*.py + finetune_*.py + chat_*.py
qwen_lora/ train.py, chat.py
examples/ tiny committed JSONL (basic_qa, conversation_core, smalltalk fixes)
plot_*.py loss-curve helpers
train75.sh / run_75m_pipeline.sh 75M full pipeline + watchdog

## Generated (NEVER commit — see .gitignore)
checkpoints/ (~23G), data/ (~353M), adapters/ (~169M), logs/

## Quickstart
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python data_prep/download_pretrain_data.py --max-documents 3000 --max-total-chars 6000000
python data_prep/download_data.py --max-examples 2000
python scratch/build_scratch_tokenizer.py
python scratch/train_from_scratch.py --steps 3000
python hybrid/train_hybrid.py --steps 5000
python hybrid/finetune_hybrid.py --data data/chat_sft.jsonl --init-from checkpoints/hybrid/hybrid-17m/final.pt --output checkpoints/hybrid/hybrid-chat --steps 2500 --seq-len 64
python hybrid/chat_hybrid.py --model checkpoints/hybrid/hybrid-chat/final.pt
python agent/chat_agent_hybrid.py --show-trace
./train75.sh start | status | stop

## Data licenses (code is MIT — data keeps its own)
FineWeb-Edu: ODC-By-1.0 · UltraChat200k: MIT · DailyDialog: CC BY-NC-SA 4.0 (non-commercial) · SQuAD2: CC BY-SA 4.0 · Qwen3-0.6B: Qwen license · Wikipedia runtime: CC BY-SA
