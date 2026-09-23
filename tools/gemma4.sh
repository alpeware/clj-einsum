#!/usr/bin/env bash
JH="${JAVA_HOME:-/usr/lib64/openjdk-25}"
JSIG=""
if [ -f "$JH/lib/server/libjsig.so" ]; then
  JSIG="$JH/lib/server/libjsig.so"
elif [ -f "$JH/lib/libjsig.so" ]; then
  JSIG="$JH/lib/libjsig.so"
elif [ -f "/usr/lib64/openjdk-25/lib/server/libjsig.so" ]; then
  JSIG="/usr/lib64/openjdk-25/lib/server/libjsig.so"
fi

if [ -n "$JSIG" ]; then
  export LD_PRELOAD="$JSIG${LD_PRELOAD:+:$LD_PRELOAD}"
fi

MODE="inference"
SCRIPT_NAME="$(basename "$0")"

if [ "$SCRIPT_NAME" = "gemma4_agent.sh" ] || [ "$SCRIPT_NAME" = "gemma4-agent.sh" ]; then
  MODE="agent"
elif [ "$1" = "agent" ] || [ "$1" = "--agent" ]; then
  MODE="agent"
  shift
elif [ "$1" = "relational" ] || [ "$1" = "--relational" ]; then
  MODE="relational"
  shift
elif [ "$1" = "cat-q" ] || [ "$1" = "--cat-q" ] || [ "$1" = "ternary" ] || [ "$1" = "--ternary" ]; then
  MODE="cat-q"
  shift
elif [ "$1" = "train-catq" ] || [ "$1" = "--train-catq" ] || [ "$1" = "train-ternary" ] || [ "$1" = "--train-ternary" ]; then
  MODE="train-catq"
  shift
fi

if [ "$MODE" = "agent" ]; then
  exec clojure -M:tools -m tools.gemma4-agent "$@"
elif [ "$MODE" = "relational" ]; then
  exec clojure -M:tools -m tools.eval-gemma4-webnlg-relational "$@"
elif [ "$MODE" = "cat-q" ]; then
  exec clojure -M:tools -m experiments.gate1-compression.cat-q-ternary.run "$@"
elif [ "$MODE" = "train-catq" ]; then
  exec clojure -M:tools -m tools.train-catq "$@"
else
  exec clojure -M:tools -m tools.gemma4-inference "$@"
fi
