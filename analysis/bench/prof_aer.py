# Where does the time go in the debias / emulation runs? For N circuits of a bench set, per backend profile:
# transpile, twirl, schedule+relax (honest), Aer wall time vs Aer's own time_taken, and the qubits Aer actually simulated.
#   PYTHONPATH=.:analysis/bench python analysis/bench/prof_aer.py <set dir> <ckpt> [N]
import copy, json, sys, time
from pathlib import Path
import numpy as np
from qiskit import qpy
import qiskit_ibm_runtime.fake_provider as fake_provider
from modules.utils.general import setup_exp, resolve_eval_config
from modules.compilation.quantum.emul import BackendManager
from backends import NOISY
from debias import variants

d, ckpt = Path(sys.argv[1]), Path(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 16
QL = sys.argv[4] if len(sys.argv) > 4 else 'none'   # 'meta': restrict the coupling map to meta['qlimit'] qubits as emulate.py does
meta = json.load(open(d / 'summary.json'))['meta']
circs = qpy.load(open(d / 'logical.qpy', 'rb'))[:N]
config, _, _ = setup_exp(resolve_eval_config(None, ckpt), seed=0)
config['emulate'].update(set=meta['eval_set'], measurement_method=meta['meas'])
print('logical qubits', sorted({c.num_qubits for c in circs}), 'clbits', sorted({c.num_clbits for c in circs}), flush=True)
for name in ['ibm_miami', 'honest_miami']:
    profile, cls = NOISY[name]
    cfg = copy.deepcopy(config)
    cfg['emulate'].update(noise=True, noise_profile=profile, backend='density_matrix',
                          aer_options={'batched_shots_gpu': False, 'shot_branching_enable': False})
    t = time.time(); nb = BackendManager(cfg, meta['qlimit'] if QL == 'meta' else None, hardware_profile=getattr(fake_provider, cls)()); t_bm = time.time() - t
    t = time.time(); tc = nb.transpile(circs, batch_size=256, optimization_level=1); t_tr = time.time() - t   # honest: incl. schedule+relax
    act = [len({q for ins in c.data if ins.operation.name not in ('barrier',) for q in ins.qubits}) for c in tc]
    act_nodelay = [len({q for ins in c.data if ins.operation.name not in ('barrier', 'delay') and not ins.operation.name.startswith('quantum_channel') for q in ins.qubits}) for c in tc]
    ops = tc[0].count_ops()
    print(f'\n== {name}: BackendManager {t_bm:.1f}s, transpile(+sched) {t_tr:.1f}s for {N}; width {tc[0].num_qubits}; '
          f'qubits touched incl. delays/channels {sorted(set(act))}, by gates only {sorted(set(act_nodelay))}; ops of circuit 0: {dict(ops)}', flush=True)
    for k, chunk, sh in [('one circuit, 1 shot', tc[:1], 1), ('one circuit, 2048 shots', tc[:1], 2048), (f'{N} circuits, 2048 shots', tc, 2048)]:
        for rep in range(1):
            t = time.time(); res = nb.backend.run(chunk, shots=sh).result(); w = time.time() - t
            r0 = res.results[0]; md = r0.metadata if hasattr(r0, 'metadata') else {}
            tt = [getattr(r, 'time_taken', None) for r in res.results]
            print(f'  {k} run {rep}: wall {w:.2f}s, Aer time_taken sum {sum(x for x in tt if x):.2f}s, result.time_taken {getattr(res, "time_taken", None)}, '
                  f'device {md.get("device")}, method {md.get("method")}, active_input_qubits {md.get("active_input_qubits")}, num_qubits {md.get("num_qubits")}, '
                  f'noise {md.get("noise")}, measure_sampling {md.get("measure_sampling")}, parallel_shots {md.get("parallel_shots")}, parallel_state_update {md.get("parallel_state_update")}', flush=True)
    t = time.time(); var = [v for c in tc[:4] for v, _ in variants(c, 4, np.random.default_rng(0), target=nb.hardware_profile.target)]; t_tw = time.time() - t
    print(f'  twirl+readout flips, 4 circuits x 4 variants: {t_tw:.2f}s', flush=True)
