#!/bin/bash
# Control the 75M hybrid training run.
#
#   ./train75.sh start    start or resume training; a watchdog restarts it
#                         automatically if it crashes
#   ./train75.sh stop     stop training (safe at any moment)
#   ./train75.sh status   show whether it is running and how far it got
#
# Stopping is always safe: pretraining checkpoints are written every 1,000
# steps to checkpoints/hybrid/hybrid-75m/latest.pt, so at most ~1,000 steps
# of work (the last ~40 minutes) are lost. "start" resumes from that file and
# skips any phase that already finished.
set -uo pipefail
cd "$(dirname "$0")"

LOG=logs/75m_pipeline.log

running() {
  pgrep -f 'run_75m_pipeline\.sh|hybrid/train_hybrid\.py|hybrid/finetune_hybrid\.py' > /dev/null
}

case "${1:-status}" in
  start)
    if running; then
      echo "Already running - use './train75.sh status' to watch it."
      exit 0
    fi
    mkdir -p logs
    setsid nohup bash -c '
      while true; do
        bash run_75m_pipeline.sh >> logs/75m_pipeline.log 2>&1
        if [ -f checkpoints/hybrid/hybrid-75m-chat/final.pt ]; then
          echo "[watchdog] all phases complete - $(date)" >> logs/75m_pipeline.log
          break
        fi
        echo "[watchdog] run ended; restarting (resumes from last checkpoint) in 15s - $(date)" >> logs/75m_pipeline.log
        sleep 15
      done
    ' > /dev/null 2>&1 < /dev/null &
    echo "Started with auto-restart watchdog (watchdog PID $!)."
    echo "Progress: tail -f logs/75m_pipeline.log"
    ;;
  stop)
    pkill -f 'run_75m_pipeline\.sh' 2>/dev/null
    pkill -f 'hybrid/train_hybrid\.py' 2>/dev/null
    pkill -f 'hybrid/finetune_hybrid\.py' 2>/dev/null
    sleep 1
    if running; then
      echo "Processes still shutting down; repeat './train75.sh stop' in a few seconds."
    else
      echo "Stopped. Progress is kept in checkpoints/hybrid/hybrid-75m/latest.pt (every 1,000 steps)."
      echo "Start again whenever you like:  ./train75.sh start"
    fi
    ;;
  graph)
    if [ "${2:-}" = "sft" ]; then
      python plot_sft.py "${3:-logs/75m_sft_curve.png}"
    else
      python plot_loss.py "${2:-logs/75m_loss_curve.png}"
    fi
    ;;
  status)
    if running; then echo "STATE: RUNNING"; else echo "STATE: STOPPED (./train75.sh start resumes)"; fi
    if [ -f "$LOG" ]; then
      echo "First start:  $(head -1 "$LOG" | grep -o 'Tue.*+03' || echo unknown)"
      echo "Last line:    $(grep -E '^step|validation|Phase|DONE|watchdog' "$LOG" | tail -1)"
    fi
    if [ -f checkpoints/hybrid/hybrid-75m/latest.pt ]; then
      echo "Last checkpoint saved: $(date -r checkpoints/hybrid/hybrid-75m/latest.pt '+%a %d %b %H:%M')"
    fi
    if [ -f checkpoints/hybrid/hybrid-75m/final.pt ]; then
      echo "Phase 1 (pretrain): FINISHED"
    fi
    if [ -f checkpoints/hybrid/hybrid-75m-chat/final.pt ]; then
      echo "Phase 2 (chat SFT): FINISHED - everything is done!"
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|status|graph [sft]}"
    exit 1
    ;;
esac
