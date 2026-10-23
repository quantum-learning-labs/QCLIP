#!/usr/bin/env bash
# Run tools/tn/check_change.sh for a fixed commit in its own worktree (comp_bridge/qclip_wt/<tag>), so edits to the
# main checkout never leak into a running check. data/, checkpoints/, results/ and cache/ link to the main checkout.
#   bash tools/tn/wt_check.sh <tag> <commit> [steps] [extra profile args]      (run from qclip/)
#   launch: run_gpu.sh -C qclip <node> tn_s2_chk_<tag> bash tools/tn/wt_check.sh <tag> <commit>
set -eu
tag=$1; commit=$2; shift 2
main=$(pwd)
wt=$main/../qclip_wt/$tag
if [ ! -e "$wt" ]; then
  git worktree add --detach "$wt" "$commit"
  for d in data checkpoints results cache; do ln -s "$main/$d" "$wt/$d"; done
fi
cd "$wt"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$commit")" ] || { echo "worktree $wt is not at $commit"; exit 1; }
bash tools/tn/check_change.sh "$tag" "$@"
