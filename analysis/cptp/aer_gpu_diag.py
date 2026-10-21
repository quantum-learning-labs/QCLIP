import faulthandler, sys, time; faulthandler.enable()
import qiskit_aer, qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
print('qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__, 'devices', AerSimulator().available_devices(), flush=True)
qc = QuantumCircuit(12); qc.h(0); [qc.cx(i, i + 1) for i in range(11)]; qc.measure_all()
for kw in (dict(method='statevector', device='GPU', cuStateVec_enable=False), dict(method='statevector', device='GPU', cuStateVec_enable=True),
           dict(method='density_matrix', device='GPU'), dict(method='tensor_network', device='GPU')):
    try:
        t = time.time(); r = AerSimulator(**kw).run(qc, shots=1000).result()
        print('OK  ', kw, 'success', r.success, 'counts', dict(list(r.get_counts().items())[:2]), f'{time.time()-t:.2f}s', flush=True)
    except Exception as e:
        print('FAIL', kw, type(e).__name__, str(e)[:300], flush=True)
