import math
import numpy as np
from collections import defaultdict
from random import randint
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from modules.compilation.einsum import einsum2interleaved


def amplitude_encoding(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    size = 1 << (len(values) - 1).bit_length()
    encoded = np.zeros(size, dtype=float)
    encoded[:len(values)] = values
    norm = np.linalg.norm(encoded)
    if norm == 0:
        raise ValueError("Cannot amplitude-encode an all-zero vector")
    return encoded / norm

def tn2qiskit(einsum_expr, gate_arr, param_dict={}, out_q=None, meas_output=True):
    if isinstance(einsum_expr, str):
        lhs, rhs = einsum_expr.split('->')
        input_indices = [list(sub.replace('$', '')) for sub in lhs.split(',')]
        output_indices = list(rhs.replace('$', ''))
    else:
        input_indices, output_indices = einsum_expr

    nq = sum(1 for gate in gate_arr if gate['op_type'] == '0')
    qreg = QuantumRegister(nq, f"qc{randint(1,10000)}")
    creg = ClassicalRegister(nq, f"c{randint(1,10000)}")
    qc = QuantumCircuit(qreg, creg)

    wire2q = defaultdict(list)
    qcounter = 0
    name2param = {}
    qiskit_param_dict = {}

    for idx_arr, gate in zip(input_indices, gate_arr):
        if gate['op_type'] == '0':
            wire_name = idx_arr[0]
            wire2q[wire_name].append(qcounter)
            qcounter += 1
        elif gate['op_type'] == '0_dag':
            # post_selection
            q_target = wire2q[idx_arr[0]].pop()
            qc.measure(q_target, q_target)
        else: 
            num_in = len(idx_arr) // 2 
            in_wires = idx_arr[:num_in]
            out_wires = idx_arr[num_in:]
            q_targets = [wire2q[w].pop(0) for w in in_wires]

            gate_func = getattr(qc, gate['op_type'].lower())
            if gate['name'] is None:
                gate_func(*q_targets)
            else:
                if gate['name'] in name2param:
                    param = name2param[gate['name']]
                else:
                    param = Parameter(gate['name'])
                    name2param[gate['name']] = param
                gate_func(param, *q_targets)

                if gate['name'] not in param_dict:
                    # unknown symbol: angle 0, as VQCModel's unk parameter (tanh(0) * 2pi)
                    qiskit_param_dict[param] = 0.0
                else:
                    qiskit_param_dict[param] = param_dict[gate['name']]

            for w, q in zip(out_wires, q_targets):
                wire2q[w].append(q) 

    # bell-test
    for wire_name, q_list in wire2q.items():
        if len(q_list) == 2:
            q_left, q_right = q_list
            qc.cx(q_left, q_right)
            qc.h(q_left)
            qc.measure(q_left, q_left)
            qc.measure(q_right, q_right)

    output_qubits = []
    sentence_wires = output_indices[:out_q] if out_q is not None else output_indices

    for wire_name in sentence_wires:
        q_list = wire2q[wire_name]
        if len(q_list) == 1:
            q_out = q_list[0]
            if meas_output:
                qc.measure(q_out, q_out)
            output_qubits.append(q_out)
    
    return qc, output_qubits, qiskit_param_dict

def _to_float_params(params):
    return {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in params.items()}


def compile_text_circuits(recipes, text_model, out_q, qlimit=None):
    txt_params = text_model._get_params()
    circuits = []
    dropped = 0

    for recipe in tqdm(recipes, desc="Compiling text circuits", unit="circuit"):
        einsum, wires = recipe
        qc, output_qubits, params = tn2qiskit(
            einsum2interleaved(einsum), wires, txt_params,
            out_q=out_q, meas_output=False,
        )
        qc.assign_parameters(_to_float_params(params), inplace=True)
        qc.metadata = {**(qc.metadata or {}), "output_qubits": output_qubits}
        if qlimit is not None and qc.num_qubits > qlimit:
            dropped += 1
            continue
        circuits.append(qc)

    print(f" Compiled {len(circuits)} text circuits | Dropped {dropped} wider than {qlimit} qubits")
    return circuits


def compile_image_circuits(images, image_model, out_q, qlimit=None):
    circuits = []
    dropped = 0

    for image in tqdm(images, desc="Compiling image circuits", unit="circuit"):
        img_vars = image_model.encode_features(image)
        in_idx, out_idx = einsum2interleaved(image_model.einsum_expr.replace('b', ''))
        qc, output_qubits, params = tn2qiskit(
            [in_idx, out_idx], image_model.gate_arr, img_vars,
            out_q=out_q, meas_output=False,
        )
        qc.assign_parameters(_to_float_params(params), inplace=True)
        qc.metadata = {**(qc.metadata or {}), "output_qubits": output_qubits}

        if qlimit is not None and qc.num_qubits > qlimit:
            dropped += 1
            continue
        circuits.append(qc)

    print(f" Compiled {len(circuits)} image circuits | Dropped {dropped} wider than {qlimit} qubits")
    return circuits
