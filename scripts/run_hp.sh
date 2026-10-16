#!/usr/bin/env bash
# Hyperparameter search driver: train one config variant for each seed, then benchmark best.pt
# (selected on validation hard_neg_acc) and save its validation metrics.
# Usage (from qclip/): bash scripts/run_hp.sh <tag> <method> <variant> [--emulate] <seed>... [-- --set k=v ...]
#   method: a configs/baseline/vqc_<method>.yaml (nodiscard, discard, discard_hs, discard_uj)
#   everything after -- is passed to train.py (e.g. --set qlr=3e-3 --set text.layers=3)
# Checkpoints: checkpoints/svo-probes/vqc/hp_<tag>_<method>_<variant>_s<seed>/
# Results:     results/hpsearch/<tag>/<method>_<variant>/s<seed>/{val,benchmark_best,emulate_best}.json
set -uo pipefail
PY=${PY:-/cs/research/pplv/comp_bridge/.venv/bin/python}
export PYTHONPATH=.

(( $# >= 4 )) || { echo "usage: $0 <tag> <method> <variant> [--emulate] <seed>... [-- train args]" >&2; exit 1; }
tag=$1 method=$2 variant=$3; shift 3
seeds=() emulate=0 train_extra=()
while (( $# )); do
    case $1 in
        --emulate) emulate=1; shift ;;
        --) shift; train_extra=("$@"); break ;;
        *) seeds+=("$1"); shift ;;
    esac
done
cfg=configs/baseline/vqc_${method}.yaml
[[ -f $cfg ]] || { echo "no $cfg" >&2; exit 1; }

for seed in "${seeds[@]}"; do
    run=hp_${tag}_${method}_${variant}_s${seed}
    out=results/hpsearch/$tag/${method}_${variant}/s$seed
    ckpt=checkpoints/svo-probes/vqc/$run/best.pt
    echo "=== $run | $(date -Is) | $(hostname) | ${train_extra[*]}"
    if [[ -e $out ]]; then echo "skip: $out exists"; continue; fi
    mkdir -p "$out"
    "$PY" scripts/train.py --config "$cfg" --seed "$seed" --run_name "$run" "${train_extra[@]}" \
        || { echo "FAILED train $run"; continue; }
    [[ -f $ckpt ]] || { echo "FAILED no best.pt for $run"; continue; }
    "$PY" - "$ckpt" "$out/val.json" "${train_extra[*]}" <<'PYEOF'
import json, sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
json.dump({'checkpoint': sys.argv[1], 'epoch': ck['epoch'], 'train_loss': ck['train_loss'],
           'val_metrics': {k: float(v) for k, v in ck['val_metrics'].items()}, 'train_args': sys.argv[3]},
          open(sys.argv[2], 'x'), indent=1)
PYEOF
    "$PY" scripts/benchmark.py -cp "$ckpt" -o "$out/benchmark_best.json" || echo "FAILED benchmark $run"
    if (( emulate )); then "$PY" scripts/emulate.py -cp "$ckpt" -o "$out/emulate_best.json" || echo "FAILED emulate $run"; fi
done
echo "=== all done $(date -Is)"
