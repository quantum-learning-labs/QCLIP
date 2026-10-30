"""Does opt_einsum path finding ('auto') consume Python's global random? (step-4 resume check)"""
import random, opt_einsum as oe, numpy as np
for n in (4, 8, 12, 16, 24):
    eq = ','.join(f"{oe.get_symbol(i)}{oe.get_symbol(i+1)}" for i in range(n)) + '->' + oe.get_symbol(0) + oe.get_symbol(n)
    shapes = [(8, 8)] * n
    random.seed(0); s0 = random.getstate()
    oe.contract_path(eq, *shapes, shapes=True)
    a = random.getstate() != s0
    random.seed(0); s0 = random.getstate()
    oe.contract_expression(eq, *shapes, optimize='auto')
    b = random.getstate() != s0
    print(f"{n} tensors: contract_path consumed {a}, contract_expression consumed {b}")
