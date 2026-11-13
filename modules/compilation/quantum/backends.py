import copy, torch
from tqdm import tqdm
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, coherent_unitary_error, RelaxationNoisePass
import qiskit_ibm_runtime.fake_provider as fake_provider
from qiskit.circuit.library import RZZGate
from qiskit.circuit import Delay
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay

# "honest" hardware noise (emulate.noise_profile: honest): the device calibration made worse where Aer's
# from_backend model is known to be optimistic. Each knob is an assumption, see the CPTP report, section on E6.
HONEST_DEFAULTS = {
    "scale_2q": 3.0,     # 2q depolarising error x3: layered/crosstalk error exceeds isolated 2q error (McKay 2023, Sarovar 2020)
    "scale_1q": 2.0,     # 1q error x2
    "scale_ro": 1.5,     # readout assignment error x1.5 (drift between calibrations)
    "scale_t": 0.5,      # T1, T2 halved (TLS fluctuations, Klimov 2018)
    "zz_rad": 0.02,      # coherent RZZ over-rotation after every 2q gate (coherent errors add up, Harper 2020)
    "idle": True,        # ALAP scheduling + thermal relaxation on idle delays (absent from from_backend)
}

# emulate.provider -> its default emulate.profile. Every profile is a local calibration snapshot of a real device,
# simulated with Aer: no account, no token, never hardware
DEFAULT_PROFILE = {'ibm': 'miami', 'iqm': 'garnet'}

def fake_backend(name, provider='ibm'):
    # device calibration snapshot by name, e.g. ('kingston', 'ibm') or ('garnet', 'iqm')
    if provider == 'ibm':
        classes = {c.lower(): c for c in dir(fake_provider) if c.startswith('Fake')}
        return getattr(fake_provider, classes['fake' + name.lower().replace('fake_', '').replace('fake', '')])()
    if provider == 'iqm':
        return iqm_fake_backend(name)
    raise ValueError(f"emulate.provider {provider!r}: choose from {list(DEFAULT_PROFILE)}")

def iqm_fake_backend(name):
    # IQM's own snapshots (qiskit extra of iqm-client): native gates r (prx) and cz, noise model built from the error profile
    import importlib
    name = name.lower().replace('iqm', '').replace('fake', '').strip('_')
    devices = ['adonis', 'aphrodite', 'apollo', 'garnet']   # deneb routes through a resonator (move gates): not supported
    if name not in devices:
        raise ValueError(f"IQM profile {name!r}: choose from {devices}")
    module = importlib.import_module(f'iqm.qiskit_iqm.fake_backends.fake_{name}')
    return getattr(module, f'IQMFake{name.capitalize()}')()

def noise_model(backend, provider):
    return backend.noise_model if provider == 'iqm' else NoiseModel.from_backend(backend)

def honest_backend(backend, knobs):
    # a copy of the fake backend whose target carries the scaled error rates and coherence times
    honest = copy.deepcopy(backend)
    target = honest.target
    for name in target.operation_names:
        props = target[name]
        if not props:
            continue
        if name == 'measure':
            scale, max_error = knobs['scale_ro'], 0.5
        else:
            scale = knobs['scale_2q'] if target.operation_from_name(name).num_qubits == 2 else knobs['scale_1q']
            max_error = 0.75
        for p in props.values():
            if p is not None and p.error is not None:
                p.error = min(p.error * scale, max_error)
    for qp in target.qubit_properties:
        if qp is None:
            continue
        if qp.t1: qp.t1 *= knobs['scale_t']
        if qp.t2: qp.t2 *= knobs['scale_t']
        if qp.t1 and qp.t2: qp.t2 = min(qp.t2, 1.99 * qp.t1)   # physical bound T2 <= 2 T1 (some snapshot qubits violate it)
    return honest

class BackendManager:
    def __init__(self, config, qlimit=None, hardware_profile=None):
        self.config = config
        self.qlimit = qlimit
        self._backend = None
        self.coupling_map = None
        self.basis_gates = None
        # emulate.provider (ibm, iqm) and emulate.profile: device calibration snapshot (ibm default FakeMiami = the
        # 120-qubit Nighthawk, iqm default Garnet)
        self.provider = config['emulate'].get('provider', 'ibm')
        if self.provider not in DEFAULT_PROFILE:
            raise ValueError(f"emulate.provider {self.provider!r}: choose from {list(DEFAULT_PROFILE)}")
        self.hardware_profile = hardware_profile or fake_backend(config['emulate'].get('profile', DEFAULT_PROFILE[self.provider]),
                                                                 self.provider)
        self._setup_backend()

    def _setup_backend(self):
        devices = AerSimulator().available_devices()
        qdev = 'GPU' if 'GPU' in devices and torch.cuda.is_available() else 'CPU'  # aer-gpu builds list GPU even on CPU-only nodes
        method = self.config['emulate'].get('backend', 'statevector')
        
        backend_options = {
            "device": qdev,
            "method": method,
            "precision": "double",
            "max_parallel_experiments": 0,
            "max_parallel_threads": 0,
            "max_parallel_shots": 0,
            "fusion_enable": True,
            "fusion_threshold": 2,
            "fusion_max_qubit": 6,
        }

        edges = self.hardware_profile.coupling_map.get_edges()
        if self.qlimit is not None:
            filtered_edges = [edge for edge in edges if edge[0] < self.qlimit and edge[1] < self.qlimit]
            self.coupling_map = CouplingMap(filtered_edges)
            if not self.coupling_map.is_connected():
                print(f" [Warning] the first {self.qlimit} qubits of {self.hardware_profile.name} are not connected")
        else:
            self.coupling_map = self.hardware_profile.coupling_map
        # IBM fake backends list their basis gates; IQM's (BackendV2 only) carry them in the target
        self.basis_gates = getattr(self.hardware_profile, 'basis_gates', None) or list(self.hardware_profile.target.operation_names)

        self.noise_profile = self.config['emulate'].get('noise_profile', 'backend')
        self.honest = {**HONEST_DEFAULTS, **(self.config['emulate'].get('honest') or {})}
        if self.config['emulate'].get('noise') and self.noise_profile == 'honest':
            if self.provider != 'ibm':
                raise ValueError("emulate.noise_profile honest is calibrated on IBM snapshots (target error rates, T1/T2, gate "
                                 "durations): use noise_profile backend with other providers")
            self.hardware_profile = honest_backend(self.hardware_profile, self.honest)
        if self.config['emulate'].get('noise'):
            noise = noise_model(self.hardware_profile, self.provider)
            if self.noise_profile == 'honest' and self.honest['zz_rad']:
                # coherent ZZ over-rotation composed onto every 2q gate's local error
                zz = coherent_unitary_error(RZZGate(self.honest['zz_rad']).to_matrix())
                for gate, errs in noise._local_quantum_errors.items():
                    for qubits, err in list(errs.items()):
                        if len(qubits) == 2:
                            errs[qubits] = err.compose(zz)
            backend_options["noise_model"] = noise
            backend_options["seed_simulator"] = 42
            backend_options['shot_branching_enable'] = True
            backend_options["statevector_parallel_threshold"] = 0
            backend_options["batched_shots_gpu_max_qubits"] = 30
            if method == 'statevector':
                backend_options["batched_shots_gpu"] = True
                backend_options["max_parallel_shots"] = 0
        else:
            if method == 'statevector':
                backend_options["statevector_sample_measure_opt"] = 10
                backend_options["batched_shots_gpu"] = True
                backend_options["max_parallel_shots"] = 256
        if method == 'matrix_product_state':
            backend_options["matrix_product_state_max_bond_dimension"] = 128
            backend_options["matrix_product_state_truncation_threshold"] = 1e-8
            backend_options["matrix_product_state_parallel_swap"] = True

        print(f" Simulation: {self.provider} profile {self.hardware_profile.name} ({self.qlimit} qubits), Device {qdev}, Method {method}, Noise {'Enabled' if self.config['emulate'].get('noise') else 'Disabled'}"
              + (f" | noise profile {self.noise_profile} {self.honest if self.noise_profile == 'honest' else ''}" if self.config['emulate'].get('noise') else ''))
        backend_options.update(self.config['emulate'].get('aer_options') or {})   # e.g. --set emulate.aer_options.batched_shots_gpu=false
        custatevec = backend_options.pop('cuStateVec_enable', True)
        if self.config['emulate'].get('aer_options'):
            print(f" Aer option overrides: {self.config['emulate']['aer_options']}")
        if method == 'tensor_network':
            self._backend = AerSimulator(method=method, device=qdev)
        else:
            self._backend = AerSimulator(method=method, device=qdev, cuStateVec_enable=custatevec)
        self._backend.set_options(**backend_options)

    @property
    def backend(self):
        return self._backend

    def transpile(self, circuits, batch_size=128, optimization_level=1):
        transpiled_circs = []
        chunks = [circuits[i:i + batch_size] for i in range(0, len(circuits), batch_size)]
        idle = self.config['emulate'].get('noise') and self.noise_profile == 'honest' and self.honest['idle']
        if idle:
            # ALAP-schedule with the device's gate durations, then thermal relaxation on every idle delay. Two pass managers:
            # the relaxation pass rewrites nodes the scheduler's start-time map refers to
            target = self.hardware_profile.target
            t1s = [q.t1 if q and q.t1 else float('inf') for q in target.qubit_properties]
            t2s = [q.t2 if q and q.t1 and q.t2 else float('inf') for q in target.qubit_properties]
            schedule = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)])
            relax = PassManager([RelaxationNoisePass(t1s, t2s, dt=self.hardware_profile.dt, op_types=[Delay])])

        for chunk in tqdm(chunks, desc="Transpiling", unit="batch"):
            transpiled_chunk = transpile(
                chunk, 
                coupling_map=self.coupling_map,
                basis_gates=self.basis_gates,
                optimization_level=optimization_level,
                seed_transpiler=self.config.get('seed'),
                num_processes=4,  # Kept strict to manage RAM limits
            )
            if idle:
                transpiled_chunk = [relax.run(schedule.run(c)) for c in transpiled_chunk]
            transpiled_circs.extend(transpiled_chunk)
        return transpiled_circs
