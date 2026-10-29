"""decouple=false: the step-3 compiler gives the same einsums/symbols as the step-2 compiler (f768b85) on 20k train rows."""
import subprocess, types, sys
from modules.compilation.classical.tensor_network import TNCompiler
from modules.compilation.classical.decompositions import MPS
from modules.utils.general import load_pkl
old_src = subprocess.check_output(['git', 'show', 'f768b85:modules/compilation/classical/tensor_network.py'], text=True)
old = types.ModuleType('old_tn'); exec(old_src, old.__dict__)
df = load_pkl('data/mscoco/processed/tn/train.pkl').iloc[:20000]
ob = {'n': 128, 's': 128, 'p': 128, 'out': 512}
a = old.TNCompiler(dict(ob), MPS(8, 1)).compile_dataset(df.copy())
b = TNCompiler(dict(ob, sem=128), MPS(8, 1)).compile_dataset(df.copy())
cols = [c for c in a.columns if c.endswith('_einsum') or c.endswith('_symbols')]
same = all(a[c].tolist() == b[c].tolist() for c in cols)
print('columns', cols, 'identical:', same, 'ids', old.TNCompiler(ob, None).id, TNCompiler(dict(ob, sem=128), None).id)
sys.exit(0 if same else 1)
