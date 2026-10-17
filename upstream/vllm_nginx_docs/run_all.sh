#!/bin/sh
# Runs every test against each config and saves the raw output in results/.
cd "$(dirname "$0")"
mkdir -p results
errs() { docker logs nginx 2>&1 | grep -oE "(recv\(\) failed \([^)]*\)|upstream prematurely closed|no live upstreams|upstream server temporarily disabled|connect\(\) failed \([^)]*\))" | sort | uniq -c; }

for c in docs proposed; do
  ./lab.sh down; ./lab.sh up $c
  { ./lab.sh client ttfb 20 64; ./lab.sh client ttfb 10 512; } > results/ttfb_$c.txt
done

for c in docs proposed; do
  ./lab.sh down; ./lab.sh up $c
  ( sleep 10; docker exec vllm1 sh -c 'touch /tmp/down; pkill -f server.py'; sleep 10; docker exec vllm1 rm /tmp/down ) &
  ./lab.sh client spread 75 10 > results/one_down_$c.txt; wait
  errs > results/one_down_${c}_nginx_errors.txt
done

for c in docs proposed; do
  ./lab.sh down; ./lab.sh up $c
  ( sleep 10; for s in vllm0 vllm1; do docker exec $s sh -c 'touch /tmp/down; pkill -f server.py'; done
    sleep 5; for s in vllm0 vllm1; do docker exec $s rm /tmp/down; done ) &
  ./lab.sh client spread 45 10 > results/both_down_$c.txt; wait
  errs > results/both_down_${c}_nginx_errors.txt
done

for c in docs_keepalive proposed; do
  ./lab.sh down; BUSY_MS=50 ./lab.sh up $c
  ./lab.sh client bursts 270 32 > results/keepalive_race_$c.txt
  errs > results/keepalive_race_${c}_nginx_errors.txt
done
./lab.sh down
{ docker run --rm nginx:1.27 nginx -v 2>&1; docker version --format 'docker {{.Server.Version}}'; date -u +%FT%TZ; } > results/environment.txt
echo done
