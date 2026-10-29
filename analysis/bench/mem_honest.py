# peak memory of one honest-profile Aer run() over N circuits (qlimit map), CPU
import copy, json, resource, sys, time
from pathlib import Path
from qiskit import qpy
import qiskit_ibm_runtime.fake_provider as fake_provider
from modules.utils.general import setup_exp, resolve_eval_config
from modules.compilation.quantum.emul import BackendManager
d, ckpt, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
meta = json.load(open(d / 'summary.json'))['meta']; circs = qpy.load(open(d / 'logical.qpy', 'rb'))[:N]
config, _, _ = setup_exp(resolve_eval_config(None, ckpt), seed=0)
config['emulate'].update(set=meta['eval_set'], measurement_method=meta['meas'], noise=True, noise_profile='honest', backend='density_matrix',
                         aer_options={'batched_shots_gpu': False, 'shot_branching_enable': False, 'max_parallel_threads': 8})
nb = BackendManager(config, meta['qlimit'], hardware_profile=fake_provider.FakeMiami())
tc = nb.transpile(circs, batch_size=256, optimization_level=1)
print('after transpile maxrss GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, flush=True)
t = time.time(); res = nb.backend.run(tc, shots=8192).result()
print(f'run {N}: {time.time() - t:.1f}s, maxrss GB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}', flush=True)
st = open('/proc/self/status').read(); print([l for l in st.splitlines() if l.startswith(('VmPeak', 'VmHWM'))], flush=True)
