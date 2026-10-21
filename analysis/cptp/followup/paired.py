# Paired per-seed comparison: distilled student vs the warm start alone (same seed, same teacher) and vs the teacher.
import json, glob, os, statistics as st, sys
T = os.path.expanduser('~/comp_bridge_agent/qclip/tools')
def load(d):
    out = {}
    for s in glob.glob(f'{d}/s*/benchmark_best.json'):
        r = json.load(open(s))['results']; out[int(s.split('/')[-2][1:])] = (r['std_hard_neg_acc'], r['swap_hard_neg_acc_norm'])
    return out
kd = load(f'{T}/kd/res/discard_kd_ws'); ws = load(f'{T}/res4/do1/discard_tl3_pw0_initnd')
seeds = sorted(set(kd) & set(ws))
d = [kd[s][0] - ws[s][0] for s in seeds]
f = lambda x: f'{st.mean(x):.3f} ± {(st.stdev(x) if len(x) > 1 else 0):.3f}'
print('seeds', seeds)
print('KD + warm start  std', f([kd[s][0] for s in seeds]), ' swap', f([kd[s][1] for s in seeds]), ' best', max(kd[s][0] for s in seeds))
print('warm start only  std', f([ws[s][0] for s in seeds]), ' swap', f([ws[s][1] for s in seeds]))
print('paired diff      std', f(d), f' wins {sum(x > 0 for x in d)}/{len(d)}', ' t =', f'{st.mean(d) / (st.stdev(d) / len(d) ** 0.5):.2f}' if len(d) > 1 else '')
