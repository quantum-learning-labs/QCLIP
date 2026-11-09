# Check that frames(typical=k) gives the same typical decisions as compiling all pairs and filtering after (the old load()).
import sys, time
from collections import Counter
from pathlib import Path
from g_count import frames
for c in sys.argv[1:]:
    t0 = time.time(); _, _, full = frames(Path(c), 0); t_full = time.time() - t0
    mode = Counter(x[1][0][0].num_qubits for x in full).most_common(1)[0][0]
    old = [x for x in full if x[1][0][0].num_qubits == mode and x[1][1][0].num_qubits == mode][:40]
    t0 = time.time(); _, _, new = frames(Path(c), 0, typical=60); t_new = time.time() - t0
    new = [x for x in new if x[1][0][0].num_qubits == mode and x[1][1][0].num_qubits == mode][:40]
    same = [a[0] for a in old] == [b[0] for b in new] and all(a[1][k][j] == b[1][k][j] for a, b in zip(old, new) for k in (0, 1) for j in (0, 1)) \
        and all(a[1][k][2] == b[1][k][2] for a, b in zip(old, new) for k in (0, 1))
    print(f'CHECK {Path(c).parent.name}: width {mode}, {len(old)} vs {len(new)} decisions, identical circuits {same}; compile {t_full:.0f}s -> {t_new:.0f}s', flush=True)
