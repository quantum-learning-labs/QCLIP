# Anytime profile of contraction-path search: how the best path found improves with search time, and the provably optimal path where it is computable.
# For the first typical decision of each checkpoint, on the post-selection (ps) and discard (rho) networks:
#   trace  : one cotengra HyperOptimizer (greedy trials + subtree reconfiguration, one process, same settings as tw_bounds.py --opts ctg --ctg_reconf)
#            run in chunks; at cumulative wall times 1, 3, 10, 30, ... s up to --budget the best tree so far is scored with opt_einsum
#            (log10 opt_cost and log2 largest intermediate: the same numbers tw_bounds.py writes and the page uses)
#   optimal: cotengra's exact dynamic programming (optimize_optimal, minimize='flops'), capped at --opt_s seconds and --mem_gb of memory
#            (a cap gives a 'timeout' / 'memory' row instead of a killed job); only for networks with <= --opt_max tensors
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/path_anytime.py -o out.csv -cp ckpt... [--budget 7200] [--nets ps,rho] [--opt_s 0]
import argparse, csv, resource, signal, time
from pathlib import Path
import numpy as np, opt_einsum as oe, torch
import cotengra as ctg
import cotengra.pathfinders.path_basic as pb
from inf_engines import TN, load

CHECK = [1, 3, 10, 30, 100, 300, 600, 1000, 1800, 3600, 7200, 14400]


def _alarm(*_): raise TimeoutError


def score(eq, arrs, path):
    _, info = oe.contract_path(eq, *arrs, optimize=path)
    return float(np.log2(max(float(info.largest_intermediate), 1.0))), float(np.log10(max(float(info.opt_cost), 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--nets', default='ps,rho')
    ap.add_argument('--budget', type=float, default=1800, help='trace length (s) per network; 0 skips the trace')
    ap.add_argument('--methods', default='greedy')
    ap.add_argument('--opt_s', type=float, default=0, help='time cap (s) of the exact optimiser; 0 skips it')
    ap.add_argument('--opt_max', type=int, default=400, help='largest network (tensors) the exact optimiser is tried on')
    ap.add_argument('--mem_gb', type=float, default=48, help='address-space cap during the exact optimiser')
    a = ap.parse_args()
    out = Path(a.out); new = not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
    w = csv.writer(open(out, 'a', newline='', buffering=1))
    if new: w.writerow(['ckpt', 'text_n', 'layers', 'nq', 'net', 'n_tensors', 'kind', 'checkpoint_s', 'elapsed_s', 'trials', 'width', 'log10_flops', 'status'])
    signal.signal(signal.SIGALRM, _alarm)
    for c in a.checkpoints:
        ckpt = Path(c); config, nq, D = load(ckpt, a); L = config['text'].get('layers'); qt, _, _, oq = D[0]
        for net in a.nets.split(','):
            T = TN(qt, oq, 'cpu', torch.complex64, rho=(net == 'rho'), opt='greedy'); arrs = [x.numpy() for x in T.arrs]
            head = [ckpt.parent.name, config['text'].get('n'), L, nq, net, len(arrs)]
            wd, lf = score(T.eq, arrs, 'greedy'); w.writerow(head + ['greedy', 0, T.path_s, 1, wd, lf, 'ok'])
            print(f'{ckpt.parent.name} {net}: {len(arrs)} tensors, greedy 10^{lf:.2f} ({T.path_s:.1f}s)', flush=True)
            if a.budget > 0:
                opt = ctg.HyperOptimizer(methods=a.methods.split(','), max_time=1, max_repeats=10 ** 9, minimize='flops', reconf_opts={}, parallel=False, progbar=False)
                t0 = time.time()
                for cp in sorted(set([x for x in CHECK if x < a.budget] + [a.budget])):
                    left = cp - (time.time() - t0)
                    if left > 0:
                        opt.max_time = left; signal.alarm(int(left * 1.5) + 60)
                        try: oe.contract_path(T.eq, *arrs, optimize=opt)
                        except TimeoutError: print('    hard stop', flush=True)
                        finally: signal.alarm(0)
                    if 'tree' not in opt.best: continue
                    wd, lf = score(T.eq, arrs, opt.tree.get_path()); el = time.time() - t0
                    w.writerow(head + ['trace', cp, el, len(opt.scores), wd, lf, 'ok'])
                    print(f'    {cp:>6}s (elapsed {el:.0f}s, {len(opt.scores)} trials): width {wd:.0f} 10^{lf:.2f}', flush=True)
            if a.opt_s > 0 and len(arrs) <= a.opt_max:
                ins = [tuple(t) for t in T.eq.split('->')[0].split(',')]; outp = tuple(T.eq.split('->')[1])
                size = {ix: d for t, x in zip(ins, arrs) for ix, d in zip(t, x.shape)}
                soft, hard = resource.getrlimit(resource.RLIMIT_AS); resource.setrlimit(resource.RLIMIT_AS, (int(a.mem_gb * 2 ** 30), hard))
                t1 = time.time(); status, wd, lf = 'ok', float('nan'), float('nan'); signal.alarm(int(a.opt_s))
                try:
                    path = pb.optimize_optimal(ins, outp, size, minimize='flops')
                    signal.alarm(0); resource.setrlimit(resource.RLIMIT_AS, (soft, hard)); wd, lf = score(T.eq, arrs, path)
                except TimeoutError: status = 'timeout'
                except MemoryError: status = 'memory'
                finally: signal.alarm(0); resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
                w.writerow(head + ['optimal', a.opt_s, time.time() - t1, 1, wd, lf, status])
                print(f'    optimal: {status} ({time.time() - t1:.0f}s) 10^{lf:.2f}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
