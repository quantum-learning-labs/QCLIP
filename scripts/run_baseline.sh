#!/usr/bin/env bash
# Discard vs non-discard baseline: for each seed, train both methods, then benchmark and
# emulate each best.pt (the checkpoint selected on validation hard_neg_acc).
# Usage (from qclip/): bash scripts/run_baseline.sh <tag> [--methods nodiscard,discard] [--epochs N] [--limit N] <seed>...
# Checkpoints: checkpoints/svo-probes/vqc/<tag>_<method>_s<seed>/
# Results:     results/discard_baseline/<tag>/<tag>_<method>_s<seed>/{benchmark,emulate}_best.json
set -uo pipefail
PY=${PY:-/cs/research/pplv/comp_bridge/.venv/bin/python}
export PYTHONPATH=.

tag=$1; shift
train_extra=() emul_extra=() seeds=() methods=(nodiscard discard)
while (( $# )); do
    case $1 in
        --epochs) train_extra+=(--epochs "$2"); shift 2 ;;
        --limit)  emul_extra+=(--limit "$2"); shift 2 ;;
        --methods) IFS=, read -r -a methods <<<"$2"; shift 2 ;;
        *)        seeds+=("$1"); shift ;;
    esac
done
(( ${#seeds[@]} )) || { echo "usage: $0 <tag> [--methods m1,m2] [--epochs N] [--limit N] <seed>..." >&2; exit 1; }

for seed in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
        run=${tag}_${method}_s${seed}
        out=results/discard_baseline/$tag/$run
        ckpt=checkpoints/svo-probes/vqc/$run/best.pt
        echo "=== $run | $(date -Is) | $(hostname)"
        if [[ -e $out ]]; then echo "skip: $out exists"; continue; fi
        mkdir -p "$out"
        "$PY" scripts/train.py --config "configs/baseline/vqc_${method}.yaml" --seed "$seed" --run_name "$run" "${train_extra[@]}" \
            || { echo "FAILED train $run"; continue; }
        [[ -f $ckpt ]] || { echo "FAILED no best.pt for $run"; continue; }
        "$PY" scripts/benchmark.py -cp "$ckpt" -o "$out/benchmark_best.json" || echo "FAILED benchmark $run"
        "$PY" scripts/emulate.py -cp "$ckpt" -o "$out/emulate_best.json" "${emul_extra[@]}" || echo "FAILED emulate $run"
    done
done
echo "=== all done $(date -Is)"
