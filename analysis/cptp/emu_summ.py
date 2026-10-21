# Summarise shot-sweep / noise emulation JSONs (stdlib). usage: emu_summ.py <dir with *.json>
import json, glob, os, sys, re, statistics as st, collections
D = sys.argv[1]
G = collections.defaultdict(lambda: collections.defaultdict(list)); acc_frac = collections.defaultdict(lambda: collections.defaultdict(list))
noise = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(os.path.join(D, '*.json'))):
    name = os.path.basename(f)[:-5]
    if name.startswith('test'): continue
    j = json.load(open(f)); bs = j['results'].get('by_shots', {})
    m = re.match(r'(noise_)?(nodiscard|discardWS|discard)_(L\w+)_s(\d+)', name); isnoise, method, L, seed = m.groups()
    for s, r in bs.items():
        tgt = noise if isnoise else G
        tgt[(method, L)][int(s)].append(r['accuracy'])
        if not isnoise: acc_frac[(method, L)][int(s)].append(r['accept_frac_mean'])
ms = lambda x: f"{st.mean(x):.3f}±{st.stdev(x) if len(x) > 1 else 0:.3f}"
print('== accuracy vs shots per circuit (600 std pairs, 3 shot-noise runs each; mean±sd over seeds)')
for k in sorted(G):
    print(k, ' '.join(f"{s}:{ms(v)}" for s, v in sorted(G[k].items())), '| accepted fraction', f"{st.mean(acc_frac[k][max(acc_frac[k])]):.3f}")
print('\n== same, against EFFECTIVE shots (accepted shots for post-selection)')
for k in sorted(G):
    a = st.mean(acc_frac[k][max(acc_frac[k])])
    print(k, ' '.join(f"{int(s * a)}:{st.mean(v):.3f}" for s, v in sorted(G[k].items())))
print('\n== hardware noise model (FakeMiami), noiseless vs noisy, seeds with both')
for k in sorted(noise):
    for s, v in sorted(noise[k].items()):
        base = G[k].get(s, [])
        print(k, f"shots {s}: noisy {ms(v)} | noiseless {ms(base[:len(v)]) if base else 'n/a'}")
