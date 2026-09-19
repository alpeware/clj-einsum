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

exec clojure -M:benchmark "$@"
