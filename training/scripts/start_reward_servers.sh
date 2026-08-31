#!/bin/bash
# Launch the reward servers that the GRPO reward functions talk to over HTTP.
#
#   port 8001  whisper      CER (Whisper large-v3)                 [required]
#   port 8002  animescore   anime-likeness reward                  [required for the AS axis]
#   port 8003  likability   likability classifier                  [optional axis]
#   port 8006  vad          VAD arousal regressor                  [optional axis]
#   port 8007  utmos        UTMOS22-strong naturalness predictor    [optional axis]
#
# Every server exposes POST /score and GET /healthz (note: /healthz, not /health).
#
# Usage:
#   bash training/scripts/start_reward_servers.sh                  # whisper + animescore
#   AXES="animescore utmos likability vad" bash .../start_reward_servers.sh
#
# Env:
#   PYBIN            python to use                (default: current `python`)
#   REWARD_GPU       CUDA device for the servers  (default: 1)
#   ANIMESCORE_CKPT  AnimeScore RankNet weights   (default: the server's own default)
#   LOG_DIR          where to write logs          (default: ./logs)
set -euo pipefail

PYBIN=${PYBIN:-$(command -v python)}
REWARD_GPU=${REWARD_GPU:-1}
LOG_DIR=${LOG_DIR:-logs}
AXES=${AXES:-"animescore"}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # training/
mkdir -p "$LOG_DIR"

launch() {  # name port extra-args...
  local name=$1 port=$2; shift 2
  echo "starting $name on :$port (GPU $REWARD_GPU)"
  CUDA_VISIBLE_DEVICES=$REWARD_GPU nohup "$PYBIN" "$HERE/reward_servers/${name}_server.py" \
    --port "$port" "$@" > "$LOG_DIR/${name}_server.log" 2>&1 &
  echo "  pid $!"
}

launch whisper 8001 --model large-v3
for axis in $AXES; do
  case $axis in
    animescore) launch animescore 8002 ${ANIMESCORE_CKPT:+--ckpt "$ANIMESCORE_CKPT"} ;;
    likability) launch likability 8003 ;;
    vad)        launch vad        8006 ;;
    utmos)      launch utmos      8007 ;;
    *) echo "unknown axis: $axis" >&2; exit 1 ;;
  esac
done

echo
echo "waiting for models to load..."
for port in 8001 8002 8003 8006 8007; do
  for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://localhost:$port/healthz" || true)
    [ "$code" = "200" ] && break
    sleep 5
  done
  [ "${code:-000}" = "200" ] && echo "  :$port ok" || echo "  :$port not up (skipped or still loading)"
done
