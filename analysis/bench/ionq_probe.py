# One tiny Bell circuit on IonQ's cloud simulator with the forte-1 noise model (free; never a QPU backend).
import os, time, json, urllib.request
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
key = open(os.path.expanduser('~/.ionq_key')).read().strip()
prov = IonQProvider(token=key)
print('backends', [b.name for b in prov.backends()])
sim = prov.get_backend('simulator')
print('num_qubits', sim.num_qubits, 'max_shots', getattr(sim.options, 'shots', None))
qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
t = time.time(); job = sim.run([qc, qc], shots=4096, noise_model='forte-1'); res = job.result()
print('elapsed', round(time.time() - t, 1), 'counts', res.get_counts())
for path in ['characterizations/backends/qpu.forte-1/current', 'backends']:
    req = urllib.request.Request('https://api.ionq.co/v0.4/' + path, headers={'Authorization': 'apiKey ' + key})
    try: print(path, json.dumps(json.load(urllib.request.urlopen(req, timeout=30)))[:1500])
    except Exception as e: print(path, 'ERR', e)
