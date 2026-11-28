# SugarCrepe subsets with a PSG graph (data/psg/splits/sc_<type>_psg.pkl) and per-run eval configs (only these tests)
import pickle, yaml, torch, copy, os
from pathlib import Path
G = torch.load('data/psg/graphs.pt'); ids = set(G['coco_id'].tolist())
T = ['swap_object', 'swap_atribute', 'replace_object', 'replace_attribute', 'replace_relation']
for t in T:
    df = pickle.load(open(f'data/sugarcrepe/processed/{t}.pkl', 'rb'))
    out = df[[int(Path(str(x)).stem) in ids for x in df['filename']]].reset_index(drop=True)
    pickle.dump(out, open(f'data/psg/splits/sc_{t}_psg.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL); print(t, len(out))
base = yaml.safe_load(open('configs/tn/dec_t3.yaml'))['dataset']['test']
C = 'checkpoints/mscoco/tn/'
for r in ['tn_s7_r0_s0', 'tn_s7_r0_s1', 'tn_s7_g1_s0', 'tn_s7_g1_s1', 'tn_s7_g2_s0', 'tn_s7_g2_s1', 'tn_s7_g3_s0_hpc']:
    c = yaml.safe_load(open(C + r + '/config.yaml'))
    img = 'data/psg/img_graph.pt' if c['vision']['method'] == 'graph' else 'data/mscoco/processed/image_embeddings.pt'
    test = {}
    for t in T:
        e = copy.deepcopy(base['sugarcrepe_' + t]); e['text_path'] = f'data/psg/splits/sc_{t}_psg.pkl'; e['img_path'] = img; test['scpsg_' + t] = e
    c['dataset']['test'] = test
    yaml.safe_dump(c, open(os.path.expanduser(f'~/tlim_scratch/s7/sc/{r}.yaml'), 'w'), sort_keys=False)
