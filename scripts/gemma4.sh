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
fi

if [ "$MODE" = "agent" ]; then
  exec clojure -M:gemma4-agent "$@"
elif [ "$MODE" = "relational" ]; then
  exec clojure -M:gemma4-relational "$@"
else
  exec clojure -M:gemma4 "$@"
fi
