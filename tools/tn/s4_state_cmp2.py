"""Per-row agreement of RowAdam row_step between two save_state checkpoints (correlation, mean |diff|)."""
import sys, torch
a, b = (torch.load(p, map_location='cpu', weights_only=False)['state']['optimizer']['state'] for p in sys.argv[1:3])
for k in sorted(a):
    if 'row_step' not in a[k]: continue
    x, y = a[k]['row_step'].double(), b[k]['row_step'].double()
    c = torch.corrcoef(torch.stack([x, y]))[0, 1].item() if x.std() > 0 else float('nan')
    print(f"bank {k}: rows {len(x)}, corr {c:.4f}, mean |diff| {(x - y).abs().mean():.2f}, mean steps {x.mean():.1f}, rows differing {(x != y).float().mean():.3f}")
