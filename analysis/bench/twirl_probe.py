import inspect
from qiskit.circuit import pauli_twirl_2q_gates as p
print(inspect.signature(p)); print((p.__doc__ or '')[:1200])
