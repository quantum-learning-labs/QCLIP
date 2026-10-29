"""Step-2 profiling harness: the scripts/train.py setup (same config, seed and call order, so the same RNG stream), then
N timed training steps of epoch 0. Writes a JSON with startup phase times, the per-step losses (comparable with the first
N rows of a run's steps.csv: tools/tn/regress.py), cuda-synchronised seconds per step phase, samples/s, peak memory,
and the top hotspots (cProfile by cumulative time, torch.profiler by CUDA/CPU time) over a few extra steps.

  PYTHONUNBUFFERED=1 PYTHONPATH=. python tools/tn/profile_train.py -cfg configs/tn/ref_clip.yaml --seed 0 --steps 300 \
      -o results/tn/step2/prof_<tag>.json
"""
import argparse, cProfile, io, json, pstats, socket, time
from pathlib import Path
import pandas as pd
import torch

from modules.utils.factory import build_experiment, build_optimizer
from modules.utils.general import setup_exp, git_state
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import ContrastiveTrainer

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-cfg', '--config', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=300, help='timed training steps (losses recorded)')
    ap.add_argument('--hotspot_steps', type=int, default=10, help='extra steps under cProfile, then under torch.profiler')
    ap.add_argument('--train_frac', type=float, default=None)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--val_eval', action='store_true', help='also time one validation pass (as train.py runs each epoch), cProfiled')
    ap.add_argument('-o', '--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    config, DEV, SEED = setup_exp(args.config, seed=args.seed, overrides=args.set)
    if args.train_frac: config['train_frac'] = args.train_frac
    startup = {}
    t0 = time.time()
    ansatz, image_model, text_model, loss_fn = build_experiment(config, DEV)
    data_engine = DataEngine(config, ansatz, DEV)
    sync(); startup['build'] = time.time() - t0; t0 = time.time()
    compiled_train = data_engine.compile_text('train')
    compiled_val = data_engine.compile_text('val')
    startup['compile'] = time.time() - t0; t0 = time.time()
    data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.image_init(image_model)
    startup['text_init'] = time.time() - t0; t0 = time.time()
    text_model.to(DEV); sync()
    startup['to_device'] = time.time() - t0; t0 = time.time()
    train_loader = data_engine.get_loader(compiled_train, split='train')
    loss_fn.to(DEV)
    optimizer, scheduler = build_optimizer(config, text_model, image_model, loss_fn)
    trainer = ContrastiveTrainer(image_model, text_model, optimizer, loss_fn, DEV)
    startup['loader_optimizer'] = time.time() - t0
    print(f"startup: {startup}", flush=True)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    trainer.timing, trainer.max_steps = {}, args.steps
    sync(); t0 = time.time()
    trainer.train_epoch(train_loader, data_engine.eval_mapper)
    sync(); train_s = time.time() - t0
    losses = [l for l, _ in trainer.step_log]
    n = len(losses)
    bs = config['batch_size']
    res = {"config": args.config, "seed": SEED, "overrides": args.set, "git_commit": git_state(), "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, "batch_size": bs,
           "params_text": sum(p.numel() for p in text_model.parameters()), "startup_s": startup,
           "steps": n, "train_s": train_s, "samples_per_s": n * bs / train_s,
           "step_s_mean_excl_first": sum(dt for _, dt in trainer.step_log[1:]) / max(n - 1, 1),
           "phase_s_per_step": {k: v / n for k, v in trainer.timing.items()},
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
           "losses": [repr(l) for l in losses]}
    print(f"{n} steps in {train_s:.1f}s = {res['samples_per_s']:.1f} samples/s | per step: "
          + " ".join(f"{k} {v*1e3:.1f}ms" for k, v in res['phase_s_per_step'].items()), flush=True)

    if args.hotspot_steps:
        # hotspots after the trace (they continue training, so they don't touch the recorded losses)
        trainer.timing, trainer.max_steps, trainer.step_log = None, args.hotspot_steps, []
        pr = cProfile.Profile(); pr.enable()
        trainer.train_epoch(train_loader, data_engine.eval_mapper); sync()
        pr.disable()
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(30)
        res["cprofile_top"] = s.getvalue()
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        with profile(activities=acts) as prof:
            trainer.train_epoch(train_loader, data_engine.eval_mapper); sync()
        ka = prof.key_averages()
        res["torch_top_cuda"] = ka.table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=25)
        res["torch_top_cpu"] = ka.table(sort_by="self_cpu_time_total", row_limit=25)
        print(res["cprofile_top"]); print(res["torch_top_cuda"]); print(res["torch_top_cpu"])

    if args.val_eval:
        from modules.models.fusion.engine import MMEvaluator
        val_loader = data_engine.get_loader(compiled_val, split='val')
        evaluator = MMEvaluator(image_model, text_model, DEV)
        tasks = config['dataset']['val'].get('diagnostics', ['global_retrieval'])
        for k in range(2):   # the first pass also searches any missing plans; report both
            pr = cProfile.Profile(); sync(); t0 = time.time(); pr.enable()
            m = evaluator.eval_set(dataloader=val_loader, tasks=tasks, eval_mapper=data_engine.eval_mapper); sync()
            pr.disable(); res[f"val_eval_s_{k}"] = time.time() - t0
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(25)
        res["val_cprofile_top"] = s.getvalue(); res["val_metrics"] = m
        print(f"val eval: {res['val_eval_s_0']:.1f}s, {res['val_eval_s_1']:.1f}s | {m}", flush=True)

    with open(out, 'x') as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")
