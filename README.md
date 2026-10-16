# Laptop Transformer trained from scratch

This project builds and trains its own small decoder-only Transformer. The model parameters start from random initialization; no pretrained model weights are loaded by the scratch training or chat programs.

The default network has about 15.7 million parameters, 8 Transformer layers, width 384, grouped-query causal attention, RMSNorm, SwiGLU, and rotary position embeddings. It uses 0.1 dropout during training; dropout is disabled during inference. It runs on CPU and fits ordinary laptop memory. The laptop used for this run has about 7.5 GiB RAM and no CUDA GPU.

A laptop-sized model trained on a small corpus will not have broad knowledge or dependable reasoning. This project is a genuine scratch-trained model and a learning platform; the included small Q&A set can teach a limited number of simple examples, but it will still make mistakes on unfamiliar questions.

## Training data

- Pretraining text comes from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), configuration sample-10BT. Its dataset card lists ODC-By-1.0. The downloader streams a shuffled sample and caps it by document count and character count; it does not download or train on the entire Internet.
- The starter chat examples come from [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), whose dataset card lists MIT. The included downloader caps the sample.
- The conversation-specialized pass uses [DailyDialog](https://huggingface.co/datasets/roskoN/dailydialog), a short multi-turn dialogue corpus. Its original dataset card lists CC BY-NC-SA 4.0 and says it is noncommercial; follow that license if you use or share the resulting model.
- The supervised starter set is in examples/basic_qa.jsonl. It covers greetings, common facts, multi-turn examples, and 429 generated, programmatically checked addition, subtraction, and multiplication examples for values from 0 to 12.
- Grounded factual Q&A uses Stanford SQuAD 2.0. Stanford's site lists CC BY-SA 4.0. The preparation script retains each Wikipedia passage with its question and builds a local search index. SQuAD is a bounded Wikipedia sample, not general coverage of the Internet.

The downloaded text and tokenizer are kept in data/ and ignored by Git. The source manifest records the FineWeb-Edu sample and its limits.

## Install

Use Python 3.10+ and install PyTorch for your machine first. For CPU-only Linux:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
~~~

If PyTorch is already installed, skip its install command. The data download needs internet. Training and chatting can run offline once the data and tokenizer have been downloaded.

## Build and train the scratch model

Run these commands from this folder; every command already includes the
subfolder that holds the script (see the project layout below).

### 1. Download a bounded pretraining sample

~~~bash
python data_prep/download_pretrain_data.py --max-documents 3000 --max-total-chars 6000000
~~~

This writes a shuffled training/validation split and a source manifest in data/pretrain/. You can use the script defaults for a somewhat larger sample, or lower the limits for a quick run.

### 2. Download chat examples

~~~bash
python data_prep/download_data.py --max-examples 2000 --max-total-chars 20000000
~~~

### 3. Train a new tokenizer

~~~bash
python scratch/build_scratch_tokenizer.py
~~~

It creates an 8,192-token byte-level BPE tokenizer from the pretraining text and chat examples.

### 4. Pretrain from random weights

~~~bash
python scratch/train_from_scratch.py --steps 10000 --seq-len 128 --batch-size 1 --threads 4
~~~

This writes resumable checkpoints to checkpoints/scratch/scratch-17m/. On the CPU laptop used during development, 10,000 steps took roughly 30 minutes; time varies by machine. To continue a 10,000-step run to 20,000 total steps:

~~~bash
python scratch/train_from_scratch.py --steps 20000 --seq-len 128 --batch-size 1 --threads 4 --resume checkpoints/scratch/scratch-17m/final.pt
~~~

### 5. Train it to follow chat turns

~~~bash
python scratch/finetune_from_scratch.py --data data/ultrachat.jsonl --pretrained checkpoints/scratch/scratch-17m/final.pt --output checkpoints/scratch/scratch-chat --steps 2000 --seq-len 256 --batch-size 1 --threads 4 --learning-rate 5e-5
~~~

This updates the full Transformer weights. It saves checkpoints in checkpoints/scratch/scratch-chat/.

### 6. Prepare factual Q&A and a local evidence index

~~~bash
python data_prep/prepare_conversation_mix.py --starter-repeats 5
python data_prep/prepare_grounded_qa.py --max-qa 8000 --max-dialogues 8000
~~~

The second command downloads Stanford's SQuAD 2.0 training file if needed. It mixes answerable and unanswerable passage-based questions with your conversation examples and builds data/squad_knowledge.sqlite3. The dataset license requires attribution and share-alike terms; keep its source details in data/squad_knowledge.source.json.

### 7. Fine-tune on conversations and grounded answers

~~~bash
python scratch/finetune_from_scratch.py --data data/grounded_qa_mix.jsonl --pretrained checkpoints/scratch/scratch-chat-expanded/final.pt --output checkpoints/scratch/scratch-chat-grounded --steps 4000 --seq-len 256 --batch-size 1 --threads 4 --learning-rate 5e-6 --dropout 0.1 --save-interval 500
~~~

This creates a new checkpoint folder and leaves your earlier model untouched. Fine-tuning can teach the small model to use supplied evidence, but it cannot guarantee correct answers to every question.

### 8. Chat with the trained model

~~~bash
python scratch/chat_from_scratch.py --model checkpoints/scratch/scratch-chat-grounded/final.pt --temperature 0.35
~~~

When the local index finds a matching passage, chat gives it to the Transformer and displays the Wikipedia article title. Replies are still generated by the model, with sampling on by default. It runs offline after the dataset, index, tokenizer, and checkpoint are in place. Type /clear to start a fresh conversation or /exit to quit.

### Continue with a larger sample

To add more material after the starter run, download a larger bounded sample and continue from the existing scratch checkpoint. This keeps the original weights available while writing the expanded model to a new folder.

~~~bash
python data_prep/download_pretrain_data.py --output-dir data/pretrain-25m --max-documents 12000 --max-total-chars 25000000
python data_prep/download_data.py --output data/ultrachat_10000.jsonl --max-examples 10000 --max-total-chars 50000000
python scratch/train_from_scratch.py --train-text data/pretrain-25m/train.txt --validation-text data/pretrain-25m/validation.txt --tokenizer data/scratch_tokenizer.json --output checkpoints/scratch/scratch-17m-expanded --steps 20000 --seq-len 128 --batch-size 1 --threads 4 --eval-interval 1000 --save-interval 2000 --resume checkpoints/scratch/scratch-17m/final.pt
python scratch/finetune_from_scratch.py --data data/ultrachat_10000.jsonl --tokenizer data/scratch_tokenizer.json --pretrained checkpoints/scratch/scratch-17m-expanded/final.pt --output checkpoints/scratch/scratch-chat-expanded --steps 3000 --seq-len 256 --batch-size 1 --threads 4 --learning-rate 5e-5
python scratch/finetune_from_scratch.py --data examples/basic_qa.jsonl --tokenizer data/scratch_tokenizer.json --pretrained checkpoints/scratch/scratch-chat-expanded/final.pt --output checkpoints/scratch/scratch-chat-expanded --steps 2000 --seq-len 128 --batch-size 1 --threads 4 --learning-rate 3e-5
python scratch/chat_from_scratch.py --model checkpoints/scratch/scratch-chat-expanded/final.pt
~~~

The pretraining command resumes at step 10,000 and trains to 20,000 total steps. The expanded Q&A checkpoint is saved separately from the baseline chat checkpoint.

### Conversation tuning experiments

`data_prep/download_conversation_data.py` can prepare short DailyDialog turns, and `data_prep/prepare_conversation_mix.py` can mix them with the hand-written starter examples. The fine-tuner now keeps complete recent messages and uses the same two-exchange history window as the chat program. Dropout defaults to 0.1 for older checkpoints; choose another value with `--dropout` (for example, `--dropout 0.2`).

On this laptop, a longer conversation-only fine-tune made the scratch model repeat the same generic answer to unrelated prompts. That checkpoint was kept separate and is not selected by the launcher. Keep the existing chat checkpoint as your working model unless a new fine-tune has been tried in chat and clearly improves it. DailyDialog’s original license is CC BY-NC-SA 4.0 and restricts commercial use.

The scratch model is still very small. Local passage retrieval can provide supporting information for matching questions, but answers may still be wrong when the passage is weak or unrelated.

## Project layout

Every command in this README is run from this folder, and each command already
includes its subfolder. Each model family lives in its own folder; shared
helpers live in `common/`; datasets, examples and checkpoints stay on top:

~~~
AE-Assistant/
├── common/                 shared helpers used by every model
│   ├── knowledge_store.py      local evidence index + grounded-answer prompts
│   └── conversation_helpers.py offline handlers for greetings and arithmetic
├── scratch/                the from-scratch Transformer (main model)
│   ├── scratch_model.py          architecture and text generation
│   ├── build_scratch_tokenizer.py fresh tokenizer training
│   ├── train_from_scratch.py     random-init language-model pretraining
│   ├── finetune_from_scratch.py  full-weight fine-tuning that keeps prior turns
│   └── chat_from_scratch.py      local terminal chat
├── mythos/                 recurrent-depth (Mythos) architecture
│   ├── mythos_model.py           Prelude + looped Recurrent Block + Coda
│   ├── mythos_reasoning.py       self-verifying "rethink" controller
│   ├── train_mythos.py           pretraining with a loop-count curriculum
│   └── chat_mythos.py            chat with an optional rethink loop
├── hybrid/                 Transformer + Mythos in one network
│   ├── hybrid_model.py           parallel path + recurrent path, gate-fused
│   ├── train_hybrid.py           pretraining with a loop-count curriculum
│   ├── finetune_hybrid.py        full-weight chat fine-tuning
│   └── chat_hybrid.py            chat, with `--mix-with` across architectures
├── agent/                  retrieval agent: search → read → judge → refine
│   ├── wiki_agent.py             the agent loop
│   ├── wiki_tools.py             live Wikipedia and offline SQuAD backends
│   ├── intent_router.py          small talk vs. question routing
│   ├── chat_agent.py             terminal chat for the retrieval agent
│   └── chat_agent_hybrid.py      hybrid model wired into the agent
├── data_prep/              dataset download and preparation
│   ├── download_pretrain_data.py     bounded FineWeb-Edu streaming download
│   ├── download_data.py              bounded UltraChat sampling
│   ├── download_conversation_data.py bounded DailyDialog preparation
│   ├── prepare_chat_sft.py           greeting/small-talk fine-tuning set
│   ├── prepare_conversation_mix.py   dialogues mixed with the starter examples
│   └── prepare_grounded_qa.py        passage QA and the local SQuAD index
├── qwen_lora/              optional Qwen3-0.6B LoRA baseline (train.py, chat.py)
├── data/                   downloaded text, tokenizer and SQLite index (git-ignored)
├── examples/               hand-written starter sets (basic_qa.jsonl, ...)
├── adapters/               LoRA adapters written by qwen_lora/train.py (git-ignored)
└── checkpoints/            trained weights, grouped by model:
    ├── scratch/              scratch-17m, scratch-chat*, ...
    ├── mythos/               mythos-17m
    └── hybrid/               hybrid-17m, hybrid-chat*, hybrid-grounded
~~~

## Mythos: recurrent-depth thinking

`mythos/mythos_model.py` adds a second architecture that reuses the same tokenizer,
data and helpers, but replaces the plain stack of blocks with a
**recurrent-depth Transformer** inspired by the public "Mythos"
reconstruction:

- **Prelude** — a few standard blocks, run once.
- **Recurrent Block** — ONE weight-shared block, looped up to `max_loop_iters`
  times. Each step refreshes the state with an LTI-stable injection
  (`h <- A*h + B*e + Block(h, e)`, where the spectral radius of `A` is kept
  below 1), a loop-index embedding, and an optional **Depth-Wise LoRA** so
  different iterations can do different work.
- **Coda** — a few standard blocks, run once.
- Optional **ACT halting** lets easy positions stop looping early, with a
  small ponder cost during training.

Because the loop count is chosen at inference, the *same* checkpoint can
think longer about harder questions. A default ~11M-parameter Mythos reaches
2 + 6 + 2 = 10 effective blocks deep from just five unique blocks.

`mythos/chat_mythos.py` turns that into a **self-verifying rethink loop**: it samples
candidate answers at increasing depth and scores them with honest signals
(tool/deterministic answers, retrieved-passage support, sample agreement, and
a repetition/sanity check), deepening only while the best answer keeps
improving. Verified replies print their loop count, score and agreement.

~~~bash
# Pretrain the RDT (random init; loop depth ramps 1 -> max during training)
python mythos/train_mythos.py --steps 5000 --seq-len 128 --max-loop-iters 6 --threads 4

# Chat with the self-verifying rethink loop (on by default)
python mythos/chat_mythos.py --model checkpoints/mythos/mythos-17m/final.pt --conversation-helpers
~~~

## Wiki agent: search, read, judge, refine

`agent/chat_agent.py` turns the project into a small retrieval agent. Instead of
trusting one lookup, it loops: **plan a query -> read passages -> judge
whether the evidence answers the question -> refine the query and search
again** until it is confident, refinement stops helping, or it runs out of
steps. If it never finds solid evidence it says so instead of guessing.

~~~bash
python agent/chat_agent.py                      # live Wikipedia (en)
python agent/chat_agent.py --offline            # local SQuAD index
python agent/chat_agent.py --show-trace         # print every search it runs
~~~

**How it finds the right page.** It separates the *subject entity* from the
*property* being asked about, then reads the subject's article **by exact
title** rather than trusting search ranking (search happily returns a
disambiguation page or a same-named novel above the article you meant).

**How it picks the answer sentence.** It scores sentences by:
- the subject/property pairing (`"Its capital ... is Paris"`),
- verb-form matches (`who *wrote*` -> `"written by"`),
- answer-type fit (a date for *when*, a number for *how many*, a name for
  *who*), and
- never returning a question or a fragment.

It then pulls a short answer out of the chosen sentence (`Paris`,
`William Shakespeare`, `2 September 1945`) and prints the supporting sentence
and source with a confidence. Answers below `--good-enough` are labelled *low
confidence* rather than presented as fact.

Runtime commands: `/trace on|off`, `/steps <n>`, `/exit`. Wikipedia requests
are throttled and cached by default; `--lang` switches language editions.

## Agent + model: retrieve, read, verify

`agent/chat_agent_hybrid.py` joins the two halves of the project. The **agent**
finds grounded evidence, the **hybrid model** reads it and writes the answer,
and a **grounding check** decides whether to trust the model or fall back.

1. **Route** — small talk ("hi", "thanks") skips retrieval and is answered by
   the model on its own. Questions go to the agent.
2. **Retrieve** — the agent searches Wikipedia recurrently until it has a
   passage that answers the question.
3. **Read** — the hybrid model reads `(question + passage)` and answers,
   sampling several candidates at different recurrence depths.
4. **Verify** — the model's content words must actually appear in the
   evidence. If they do not, the extractive answer is shown instead and
   labelled as a fallback. The model is never allowed to assert something the
   source does not support.

~~~bash
# Teach the model to read a passage and answer from it (grounded SFT)
python hybrid/finetune_hybrid.py \
    --data data/grounded_qa_mix.jsonl --data data/chat_sft_greet.jsonl \
    --init-from checkpoints/hybrid/hybrid-chat-greet/final.pt \
    --output checkpoints/hybrid/hybrid-grounded --steps 1400 --seq-len 224 \
    --max-examples 6000 --shuffle --threads 4

# Retrieve -> read -> verify
python agent/chat_agent_hybrid.py --model checkpoints/hybrid/hybrid-grounded/final.pt --show-trace
~~~

Because the fallback is always available, the system is correct-by-construction
even while the reader is still weak: a badly grounded model answer is discarded
rather than shown. `--min-grounding` sets how strict that check is, and
`/model` toggles between the model reader and the extractive agent so you can
compare them on the same question.



## Hybrid: Transformer + Mythos in one network

`hybrid/hybrid_model.py` mixes the two architectures so they cooperate instead of
competing. Every `HybridStage` runs **both** paths on the same hidden state and
fuses them with a learned per-channel gate:

- a unique **standard Transformer block** (parallel depth — good at memorising
  patterns), and
- a weight-shared **looped recurrent block** (recurrent depth — good at
  composing, and at thinking longer on hard inputs).

The gate lets the network itself decide, per channel, how much to trust
parallel memory versus recurrent reasoning, and `n_loops` still sets thinking
time at inference. Layout: `Prelude -> [HybridStage x N] -> Coda`.

~~~bash
# Pretrain the hybrid from random weights (default ~16.5M params, 14 blocks deep at max loops)
python hybrid/train_hybrid.py --steps 5000 --seq-len 128 --stages 2 --max-loop-iters 4 --threads 4

# Chat, mixing in a separately trained Transformer and/or Mythos model
python hybrid/chat_hybrid.py --model checkpoints/hybrid/hybrid-17m/final.pt \
    --mix-with checkpoints/scratch/scratch-chat-grounded/final.pt \
    --conversation-helpers
~~~

`--mix-with` (repeatable) rebuilds each extra checkpoint (it auto-detects
`transformer`, `mythos` or `hybrid` from its config) and lets the
self-verifying rethink loop draw candidate answers from every model, so the
final answer is *verified across architectures*. Type `/sources` at the prompt
to see which models are being mixed.

Scaling knobs: `--stages` (unique depth per path), `--prelude-layers` /
`--coda-layers`, `--max-loop-iters` (recurrent depth), `--depth-lora-rank`, and
`--standard-weight` (starting balance of the fusion gate).

Runtime commands: `/think` toggles the rethink loop, `/loops <n>` caps the
thinking depth, and `/clear` and `/exit` behave as before.

## Training the 75M hybrid on everything

`run_75m_pipeline.sh` trains a ~74M-parameter hybrid — a bigger version of the
architecture above (`--dim 768 --heads 12 --kv-heads 3 --hidden-dim 1920`,
3 prelude / 3 hybrid stages / 2 coda blocks, up to 20 effective blocks deep at
max loops) — on **all** of the project's data:

1. **Pretrain** on `data/pretrain-all/`, the concatenation of both
   FineWeb-Edu samples (7,580,233 tokens; 59,221 steps × 128 tokens ≈ one
   pass, drawn at random from the whole corpus).
2. **Chat fine-tune** on all nine conversation sets at once (both UltraChat
   samples, grounded Q&A, the DailyDialog mixes, `chat_sft*` and the
   hand-written starter sets — 148,098 assistant-turn examples, shuffled and
   capped to 6,000 so every file contributes).

~~~bash
bash run_75m_pipeline.sh        # starts (or resumes) the full run
tail -f logs/75m_pipeline.log   # watch progress
~~~

The phases are idempotent: re-running the script skips a finished phase, and
an interrupted pretraining phase resumes from
`checkpoints/hybrid/hybrid-75m/latest.pt` (a checkpoint is written every
1,000 steps). On the CPU laptop used during development (8 threads, measured
~65 train tokens/second) the full pipeline takes roughly 40 hours — about 33
hours of pretraining plus 7 hours of fine-tuning — and peaks near 2 GB of RAM.

Once trained, `hybrid/chat_hybrid.py` selects the 75M chat checkpoint first,
and `agent/chat_agent_hybrid.py` uses it as its default reader:

~~~bash
python hybrid/chat_hybrid.py
python agent/chat_agent_hybrid.py --show-trace
~~~

The same width/depth flags let you size any hybrid variant
(`--dim/--heads/--kv-heads/--hidden-dim/--prelude-layers/--stages/--coda-layers/--max-loop-iters`).

## Chatting on its own (no preset replies)

The greetings in `common/conversation_helpers.py` are an **optional** convenience
(`--conversation-helpers`); they are code, not the model. By default the chat
programs show whatever the model itself generates. For the model to actually
greet and small-talk on its own, it must be **chat fine-tuned** first — a
pretrained checkpoint only knows how to continue raw text.

Two steps teach it to chat from its own weights:

~~~bash
# 1. Build a greeting/small-talk heavy supervised set (mix in your own conversations)
python data_prep/prepare_chat_sft.py --repeats 3 \
    --extra examples/conversation_core.jsonl \
    --extra examples/basic_qa.jsonl

# 2. Full-weight chat fine-tune of the hybrid from your pretrained checkpoint
python hybrid/finetune_hybrid.py --data data/chat_sft.jsonl \
    --init-from checkpoints/hybrid/hybrid-17m/final.pt \
    --output checkpoints/hybrid/hybrid-chat --steps 2500 --seq-len 64 --threads 4

# 3. Chat — WITHOUT --conversation-helpers, so every reply comes from the model
python hybrid/chat_hybrid.py --model checkpoints/hybrid/hybrid-chat/final.pt
~~~

`hybrid/finetune_hybrid.py` shares its example building, label masking (assistant
turns only) and checkpointing with `scratch/finetune_from_scratch.py`, so all three
models are trained the same way. `--data` is repeatable, so you can fine-tune
on several conversation files (e.g. add `data/conversation_mix.jsonl` for real
multi-turn dialogue). The same recipe works for the Mythos model by swapping
`create_hybrid`/`HybridConfig` for `create_mythos`/`MythosConfig`.

