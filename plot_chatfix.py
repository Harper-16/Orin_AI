#!/usr/bin/env python3
"""Plot a chat fine-tune loss tail pasted from the terminal.

Usage:
  python plot_chatfix.py [logfile] [output.png]
  - logfile: text file with lines like "step  5680/6000 | chat loss 3.4256"
    (default: logs/chatfix_tail.log)
  - output: default logs/chatfix_loss.png

If you only have terminal output, save it first, e.g.:
  python -u hybrid/finetune_hybrid.py ... 2>&1 | tee logs/chatfix.log
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/chatfix_tail.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "logs/chatfix_loss.png"

step_re = re.compile(r"step\s+(\d+)/\d+\s*\|\s*chat loss ([0-9.]+)")
steps, losses = [], []
with open(LOG) as h:
    for line in h:
        m = step_re.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))

if not steps:
    raise SystemExit(f"No 'step N/M | chat loss X' lines in {LOG}")

window = 5
smooth = [sum(losses[max(0, i - window // 2):i + window // 2 + 1])
          / len(losses[max(0, i - window // 2):i + window // 2 + 1])
          for i in range(len(losses))]

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(steps, losses, "o-", color="#ffb703", ms=4, lw=1, alpha=0.7, label="chat loss (raw)")
ax.plot(steps, smooth, "-", color="#9d0208", lw=2, label=f"smoothed ({window}-pt)")
mean = sum(losses) / len(losses)
ax.axhline(mean, ls="--", color="#6c757d", label=f"mean {mean:.2f}")
ax.set_title(f"hybrid-75m-chatfix fine-tune tail — steps {steps[0]}-{steps[-1]} (n={len(steps)})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("step")
ax.set_ylabel("chat loss (nats)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}  n={len(steps)} mean={mean:.2f} last={losses[-1]:.2f} min={min(losses):.2f}")
