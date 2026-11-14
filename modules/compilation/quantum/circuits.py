import math
import numpy as np
from collections import defaultdict
from random import randint
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from modules.compilation.einsum import einsum2interleaved
from modules.models.vision.clip import amplitude_encoding

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

def strip_clbits(qc):
    if qc.num_clbits == 0:
        return qc
    clean_qc = QuantumCircuit(qc.num_qubits)
    for instr in qc.data:
        if instr.operation.name != 'measure':
            qargs = [qc.find_bit(q).index for q in instr.qubits]
            clean_qc.append(instr.operation, qargs)
    return clean_qc

def compute_uncompute(qc_txt, qc_img, out_q, discard):
    qc_txt = strip_clbits(qc_txt)
    qc_img = strip_clbits(qc_img)

    n_txt = qc_txt.num_qubits
    n_out = len(out_q)

    if discard:
        qc = QuantumCircuit(n_txt, n_out)
        qc.compose(qc_txt, inplace=True)
        qc.compose(qc_img.inverse(), qubits=out_q, inplace=True)
        qc.measure(out_q, range(n_out))
    else:
        qc = QuantumCircuit(n_txt, n_txt)
        qc.compose(qc_txt, inplace=True)
        qc.compose(qc_img.inverse(), qubits=out_q, inplace=True)
        qc.measure(range(n_txt), range(n_txt))
    return qc

def destructive_swap(qc_txt, qc_img, out_q, discard):
    qc_txt = strip_clbits(qc_txt)
    qc_img = strip_clbits(qc_img)

    n_out = len(out_q)
    n_txt = qc_txt.num_qubits
    n_total = n_txt + n_out

    if discard:
        qc = QuantumCircuit(n_total, 2 * n_out)
        qc.compose(qc_txt, range(n_txt), inplace=True)
        qc.compose(qc_img, range(n_txt, n_total), inplace=True)

        for i, q_out in enumerate(out_q):
            img_q = n_txt + i 
            qc.cx(q_out, img_q)
            qc.h(q_out)
            qc.measure(q_out, i)
            qc.measure(img_q, n_out + i)
    else:
        qc = QuantumCircuit(n_total, n_total)
        qc.compose(qc_txt, range(n_txt), inplace=True)
        qc.compose(qc_img, range(n_txt, n_total), inplace=True)

        for i, q_out in enumerate(out_q):
            img_q = n_txt + i 
            qc.cx(q_out, img_q)
            qc.h(q_out)
        qc.measure(range(n_total), range(n_total))
    return qc

def compile_circuits(dataset, text_model, image_model, config, qlimit, limit=None):
    # one (text, image) overlap circuit per side of every pair: text choice (one image, two captions) or image choice
    discard = config['text'].get('discard', False)
    sent_q = config['embedding_qubits'] if discard else None   # discard: only the sentence wires are output qubits
    meas_method = config['emulate'].get('measurement_method', 'compute_uncompute')
    frame = {'compute_uncompute': compute_uncompute, 'destructive_swap': destructive_swap}[meas_method]
    data_size = min(limit, len(dataset)) if limit else len(dataset)
    print(f" Compiling {data_size} circuit pairs from dataset with {len(dataset)} samples.")

    txt_params = text_model._get_params()
    img_params = image_model._get_params()

    raw_pos_circs, raw_neg_circs = [], []
    too_wide = 0
    unknown_symbols = set()

    def to_f(params):
        return {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in params.items()}

    def text_circ(recipe):
        einsum, wires = recipe
        unknown_symbols.update(g['name'] for g in wires if g['name'] is not None and g['name'] not in txt_params)
        qc, out_q, prm = tn2qiskit(einsum2interleaved(einsum), wires, txt_params, out_q=sent_q, meas_output=False)
        qc.assign_parameters(to_f(prm), inplace=True)
        return qc, out_q

    def image_circ(image):
        if config['vision']['method'] == 'amp':
            img_vec = image if isinstance(image, np.ndarray) else image.detach().cpu().numpy()
            normed_img_vec = amplitude_encoding(img_vec)
            qc = QuantumCircuit(int(math.ceil(math.log2(len(normed_img_vec)))), 0)
            qc.initialize(normed_img_vec)
            return qc
        img_vars = img_params | image_model.encode_features(image)
        in_idx, out_idx = einsum2interleaved(image_model.einsum_expr.replace('b', ''))
        qc, _, img_p = tn2qiskit([in_idx, out_idx], image_model.gate_arr, img_vars, out_q=sent_q, meas_output=False)
        if img_p: qc.assign_parameters(to_f(img_p), inplace=True)
        return qc

    for idx in tqdm(range(data_size), desc="Compiling Qiskit DAGs", unit="pair"):
        sample = dataset[idx]
        if "image" in sample:      # text choice (swap): one image, two captions
            qc_pos_txt, pos_out_q = text_circ(sample["pos_caption"])
            qc_neg_txt, neg_out_q = text_circ(sample["neg_caption"])
            qc_img_pos = qc_img_neg = image_circ(sample["image"])
        else:                      # image choice (std): one caption, two images
            qc_pos_txt, pos_out_q = text_circ(sample["caption"])
            qc_neg_txt, neg_out_q = qc_pos_txt, pos_out_q
            qc_img_pos, qc_img_neg = image_circ(sample["pos_image"]), image_circ(sample["neg_image"])

        if qc_pos_txt.num_qubits > qlimit or qc_neg_txt.num_qubits > qlimit:
            too_wide += 1
            continue

        qc_pos = frame(qc_pos_txt, qc_img_pos, pos_out_q, discard)
        qc_neg = frame(qc_neg_txt, qc_img_neg, neg_out_q, discard)

        qc_pos.metadata = {"output_qubits": pos_out_q}
        qc_neg.metadata = {"output_qubits": neg_out_q}
        raw_pos_circs.append(qc_pos)
        raw_neg_circs.append(qc_neg)

    print(f" Compiled {len(raw_pos_circs)} circuit pairs | Dropped {too_wide} pairs wider than {qlimit} qubits | Unknown text symbols (angle 0): {len(unknown_symbols)}")
    return raw_pos_circs, raw_neg_circs
