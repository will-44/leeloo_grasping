#!/bin/bash
# Usage : run_phase.sh <phase> <load:0|1> [commande bench dans le conteneur | "sleep N"]
# Lance (si load=1) la charge stack (RM + cuRobo MPC + sonde base 10 Hz) dans
# grasp_bench, tegrastats sur l'hôte, puis la commande ; arrête tout à la fin.
set -u
PHASE=$1; LOAD=$2; shift 2; CMD="$*"
R=/data/grasp_bench/results; mkdir -p $R
C="docker exec grasp_bench"

sudo tegrastats --interval 500 --logfile $R/${PHASE}_tegra.log &
sleep 1

if [ "$LOAD" = 1 ]; then
  for w in rm curobo base; do
    $C bash -c "cd /bench && nohup python3 -I bench/stack_load.py $w --out /bench/results/${PHASE}_$w.csv > /bench/results/${PHASE}_$w.log 2>&1 &"
  done
  # attendre que cuRobo tourne (cuda graph + warmup)
  for i in $(seq 1 180); do
    n=$(wc -l < $R/${PHASE}_curobo.csv 2>/dev/null || echo 0)
    [ "$n" -gt 20 ] && break; sleep 1
  done
  echo "[$PHASE] charge stack active (curobo rows=$n) $(date +%T)"
  sleep 10
fi

echo "[$PHASE] start: $CMD  $(date +%T)"
if [[ "$CMD" == sleep* ]]; then $CMD; else $C bash -c "cd /bench && $CMD" > $R/${PHASE}.log 2>&1; fi
echo "[$PHASE] done $(date +%T)"

if [ "$LOAD" = 1 ]; then
  $C pkill -TERM -f bench/stack_load.py; sleep 3
fi
sudo pkill tegrastats
