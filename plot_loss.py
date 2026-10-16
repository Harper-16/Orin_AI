#!/usr/bin/env python3
"""Plot the 75M pretraining loss curve from the pipeline log.

Usage: python plot_loss.py [output.png]   (default: logs/75m_loss_curve.png)
"""
import re
import math
import datetime
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "logs/75m_pipeline.log"
TOTAL = 59221
OUT = sys.argv[1] if len(sys.argv) > 1 else "logs/75m_loss_curve.png"

steps, losses = [], []
val_steps, val_losses = [], []
last_step = None
step_re = re.compile(r"^step\s+(\d+)/(\d+) \| loops (\d+) \| lm ([0-9.]+)")
val_re = re.compile(r"^\s*validation loss: ([0-9.]+)")
with open(LOG) as handle:
    for line in handle:
        m = step_re.match(line)
        if m:
            last_step = int(m.group(1))
            steps.append(last_step)
            losses.append(float(m.group(4)))
            continue
        m = val_re.match(line)
        if m and last_step is not None:
            val_steps.append(last_step)
            val_losses.append(float(m.group(1)))

if not steps:
    raise SystemExit(f"No training steps found in {LOG} yet.")

# Moving average of the (noisy, batch-size-1) training loss.
window = 50
smooth = []
for i in range(len(losses)):
    lo = max(0, i - window // 2)
    hi = min(len(losses), i + window // 2 + 1)
    smooth.append(sum(losses[lo:hi]) / (hi - lo))

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.plot(steps, losses, color="#8ecae6", linewidth=0.7, alpha=0.65, label="train loss (raw, per batch)")
ax.plot(steps, smooth, color="#023047", linewidth=2.2,
        label=f"train loss (smoothed, {window}-print window)")
ax.plot(val_steps, val_losses, "o-", color="#e63946", linewidth=2, markersize=7,
        label="validation loss (eval mode)")
for xs, ys in zip(val_steps, val_losses):
    ax.annotate(f"{ys:.3f}", (xs, ys), textcoords="offset points", xytext=(8, -14),
                color="#e63946", fontsize=10, fontweight="bold")
ax.axhline(math.log(8192), linestyle="--", color="#6c757d", linewidth=1.2,
           label="random init = ln(8192) ≈ 9.01")
ax.axvline(steps[-1], linestyle=":", color="#ffb703", linewidth=1.5, label="now")

pct = 100 * steps[-1] / TOTAL
ax.set_title(
    "75M hybrid pretraining - loss from start to now\n"
    f"step {steps[-1]:,} / {TOTAL:,} ({pct:.1f}%)  -  {datetime.datetime.now():%a %d %b %H:%M}",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("training step")
ax.set_ylabel("loss (nats)")
ax.set_xlim(0, TOTAL)
ax.set_ylim(min(5.0, min(val_losses) - 0.4), 9.6)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved: {OUT}  (step {steps[-1]:,}/{TOTAL:,}, "
      f"train {smooth[-1]:.2f}, validation {val_losses[-1]:.3f})" if val_losses
      else f"saved: {OUT}  (step {steps[-1]:,}/{TOTAL:,}, train {smooth[-1]:.2f})")
