"""Blur-aware composite: input where DP says in-focus, upsampled deblur elsewhere (soft weight)."""
import sys, os, json, numpy as np, cv2, torch
sys.path.insert(0, 'code')
from uhdd import metrics
torch.set_num_threads(8)
D = os.environ.get('NATIVE', 'dataset/dpdd_native'); R = os.environ.get('RESULTS', 'dataset/results/local')
T0, T1 = 0.4, 1.2           # DP |disparity| (px @1680): w=1 below T0, 0 above T1
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
def t(x): return torch.from_numpy(x[:, :, ::-1].astype(np.float32) / 65535).permute(2, 0, 1)[None].contiguous()
def blurmap(n, hw):
    d = (rd(f'dataset/dpdd_1680/test/dp_maps/{n}_disp.png').astype(np.float32) - 32768) / 1000
    c = rd(f'dataset/dpdd_1680/test/dp_maps/{n}_conf.png').astype(np.float32) / 255
    k = cv2.boxFilter(np.abs(d) * c, -1, (31, 31)) / (cv2.boxFilter(c, -1, (31, 31)) + 1e-6)
    return cv2.resize(k, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
up = sys.argv[1]            # e.g. drb_x4+bicubic_x4 or drb_x4+swinir_x4_real
names = sorted(f for f in os.listdir(f'{R}/{up}') if f.endswith('.png'))
rows = []
for n in names:
    gt, inp, sr = rd(f'{D}/x1/targets/{n}'), rd(f'{D}/inputs/{n}'), rd(f'{R}/{up}/{n}')
    m = rd(f'{D}/x1/masks/{n}') > 127
    w = np.clip((T1 - blurmap(n[:-4], gt.shape[:2])) / (T1 - T0), 0, 1)[..., None]
    comp = (w * inp.astype(np.float32) + (1 - w) * sr.astype(np.float32))
    c = 64
    g, p = t(gt)[..., c:-c, c:-c], t(comp)[..., c:-c, c:-c]
    mm = torch.from_numpy(m)[None, None][..., c:-c, c:-c]
    r = metrics.compute(['psnr', 'ssim', 'lpips', 'hb'], p, g, mm, {'scale': 1, 'fr_tile': 1024})
    r.update(name=n[:-4], w_mean=float(w.mean()))
    rows.append(r); print(n, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
out = f'{R}/composite_{up}.json'
json.dump(rows, open(out, 'w'))
print('mean', {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ('psnr', 'ssim', 'lpips', 'hb_nmse_db', 'w_mean')})
