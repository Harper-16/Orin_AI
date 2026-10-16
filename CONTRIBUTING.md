# Contributing

Thanks for helping with AE-Assistant (tiny scratch-trained chat models + a
Wikipedia-grounded agent).

## Quick rules

1. Keep it CPU-first and laptop-friendly — no GPU-only code paths.
2. Never commit generated artefacts: `checkpoints/`, `adapters/`, `data/`,
   `logs/`, `*.pt`, `*.jsonl` (outside `examples/`), `*.sqlite3`, `*.log`,
   `*.png`. `git status` should show only code/docs before you push.
3. Match the existing style: `#!/usr/bin/env python3`, module docstring,
   `parse_args()` + `main()`, `if __name__ == "__main__": main()`.
4. Datasets keep their own licenses (see `LICENSE` notes + `README.md`).
   Don't vendor downloaded corpora — commit scripts + small `examples/` only.

## Dev loop

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m py_compile $(git ls-files '*.py')
```
