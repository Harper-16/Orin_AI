#!/usr/bin/env python3
"""Plot the 75M hybrid chat fine-tuning (Phase 2) loss from the pipeline log.

Usage: python plot_sft.py [output.png]   (default: logs/75m_sft_curve.png)
"""
import re
import datetime
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "logs/75m_pipeline.log"
TOTAL = 6000
PRETRAIN_FINAL_VAL = 4.739  # where Phase 2 started from
OUT = sys.argv[1] if len(sys.argv) > 1 else "logs/75m_sft_curve.png"

steps, losses = [], []
step_re = re.compile(r"^step\s+(\d+)/(\d+) \| chat loss ([0-9.]+)")
with open(LOG) as handle:
    for line in handle:
        m = step_re.match(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(3)))

if not steps:
    raise SystemExit("No fine-tuning steps found in the log yet (Phase 2 has not started).")

# Moving average of the (noisy, batch-size-1) chat loss.
window = 50
smooth = []
for i in range(len(losses)):
    lo = max(0, i - window // 2)
    hi = min(len(losses), i + window // 2 + 1)
    smooth.append(sum(losses[lo:hi]) / (hi - lo))

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.plot(steps, losses, color="#ffb703", linewidth=0.7, alpha=0.55, label="chat loss (raw, per batch)")
ax.plot(steps, smooth, color="#9d0208", linewidth=2.2,
        label=f"chat loss (smoothed, {window}-print window)")
ax.axhline(PRETRAIN_FINAL_VAL, linestyle="--", color="#6c757d", linewidth=1.2,
           label=f"pretrain val loss at hand-off ({PRETRAIN_FINAL_VAL})")
ax.axvline(steps[-1], linestyle=":", color="#023047", linewidth=1.5, label="now")

pct = 100 * steps[-1] / TOTAL
ax.set_title(
    "75M hybrid - chat fine-tuning (Phase 2: all 9 conversation sets)\n"
    f"step {steps[-1]:,} / {TOTAL:,} ({pct:.1f}%)  -  {datetime.datetime.now():%a %d %b %H:%M}",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("fine-tuning step")
ax.set_ylabel("loss (nats)")
ax.set_xlim(0, TOTAL)
lo = min(min(losses), min(smooth))
hi = max(max(losses), max(smooth))
ax.set_ylim(max(0.0, lo - 0.4), hi + 0.4)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved: {OUT}  (step {steps[-1]:,}/{TOTAL:,}, raw {losses[-1]:.2f}, smoothed {smooth[-1]:.2f})")
