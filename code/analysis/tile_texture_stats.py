"""Does texture (sharpness / noise relative to GT) depend on which SwinIR tile a region came from?
eta^2 = between-tile variance / total variance of per-cell log-ratios (pred/GT), with cells grouped by
the SwinIR tile that dominates them. Bicubic (untiled) on the same grouping = content-only baseline."""
import sys, os, numpy as np, cv2, torch
sys.path.insert(0, 'code')
from uhdd.metrics.consistency import cell_stats
from uhdd.tiling import grid_positions
torch.set_num_threads(4)
D = os.environ.get('NATIVE', 'dataset/dpdd_native'); R = os.environ.get('RESULTS', 'dataset/results/local')
CELL = 256
def t(p): x = cv2.imread(p, -1); return torch.from_numpy(x[:, :, ::-1].astype(np.float32) / 65535).permute(2, 0, 1)[None].contiguous()
def tile_id(n, pos, T, s=4):  # index of tile whose non-overlap core contains coordinate n (native px)
    cores = [((a + (pos[i - 1] + T if i else a)) / 2 * s if i else 0) for i, a in enumerate(pos)]
    return int(np.searchsorted(np.array(cores[1:]), n, side='right'))
xs, ys = grid_positions(1680, 512, 32), grid_positions(1120, 512, 32)
def eta2(v, g):
    v, g = np.asarray(v), np.asarray(g)
    tot = ((v - v.mean()) ** 2).sum()
    btw = sum(((v[g == k].mean() - v.mean()) ** 2) * (g == k).sum() for k in np.unique(g))
    return btw / max(tot, 1e-12)
methods = {'bicubic (untiled)': 'drb_x4+bicubic_x4', 'SwinIR-real (512 tiles)': 'drb_x4+swinir_x4_real'}
names = sorted(f for f in os.listdir(f'{R}/drb_x4+swinir_x4_real') if f.endswith('.png'))
res = {k: {'sharp': [], 'noise': [], 'slope': []} for k in methods}
for n in names:
    g = t(f'{D}/x1/targets/{n}'); sg = cell_stats(g, CELL)
    H, W = g.shape[-2:]
    nh, nw = H // CELL, W // CELL
    grp = np.array([tile_id(cy * CELL + CELL // 2, ys, 512) * 10 + tile_id(cx * CELL + CELL // 2, xs, 512) for cy in range(nh) for cx in range(nw)])
    for k, d in methods.items():
        sp = cell_stats(t(f'{R}/{d}/{n}'), CELL)
        for s in ('sharp', 'noise'):
            res[k][s].append(eta2(np.log((sp[s] + 1e-6) / (sg[s] + 1e-6)).numpy(), grp))
        res[k]['slope'].append(eta2((sp['slope'] - sg['slope']).numpy(), grp))
print(f'eta^2 of per-cell texture deviation explained by SwinIR tile membership ({len(names)} images, {CELL}-px cells; higher = tile-dependent texture):')
for k in methods:
    print(f"  {k:24s} sharpness {np.mean(res[k]['sharp']):.3f}  noise {np.mean(res[k]['noise']):.3f}  spectral slope {np.mean(res[k]['slope']):.3f}")
d = {s: np.array(res['SwinIR-real (512 tiles)'][s]) - np.array(res['bicubic (untiled)'][s]) for s in ('sharp', 'noise', 'slope')}
print('  SwinIR - bicubic per image:', {s: f'{v.mean():+.3f} (images higher: {(v > 0).sum()}/{len(v)})' for s, v in d.items()})
