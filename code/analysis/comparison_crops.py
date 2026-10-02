import sys, os, numpy as np, cv2
sys.path.insert(0, 'code')
from uhdd.tiling import grid_positions
D, R, V = 'dataset/dpdd_native', 'dataset/results/local', 'dataset/results/viz'
C = 384
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
def to8(x): return (np.clip(x.astype(np.float32) / 65535, 0, 1) ** 1.0 * 255).astype(np.uint8)
def blurmap(n, hw):
    d = (rd(f'dataset/dpdd_1680/test/dp_maps/{n}_disp.png').astype(np.float32) - 32768) / 1000
    c = rd(f'dataset/dpdd_1680/test/dp_maps/{n}_conf.png').astype(np.float32) / 255
    k = cv2.boxFilter(np.abs(d) * c, -1, (31, 31)) / (cv2.boxFilter(c, -1, (31, 31)) + 1e-6)
    return cv2.resize(k, (hw[1], hw[0]))
def texture(g):  # local gradient energy of GT luma, 1/8 res for speed
    s = cv2.resize(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32), None, fx=1/8, fy=1/8, interpolation=cv2.INTER_AREA)
    e = np.hypot(cv2.Sobel(s, cv2.CV_32F, 1, 0), cv2.Sobel(s, cv2.CV_32F, 0, 1))
    return cv2.resize(cv2.blur(e, (9, 9)), (g.shape[1], g.shape[0]))
names = sorted(os.listdir(f'{R}/drb_x4+swinir_x4_real'))
names = [n for n in names if n.endswith('.png')]
xs = grid_positions(1680, 512, 32); ys = grid_positions(1120, 512, 32)
seam_x = [int((b + a + 512) / 2 * 4) for a, b in zip(xs[:-1], xs[1:])]
for n in names:
    st = n[:-4]
    gt = rd(f'{D}/x1/targets/{n}'); H, W = gt.shape[:2]
    cols = [('GT f/22', gt), ('input f/4', rd(f'{D}/inputs/{n}')), ('DRBNet native t1024', rd(f'{R}/drb_x1_t1024/{n}')),
            ('DRBNet x4 + bicubic', rd(f'{R}/drb_x4+bicubic_x4/{n}')), ('DRBNet x4 + SwinIR-real', rd(f'{R}/drb_x4+swinir_x4_real/{n}'))]
    tex = texture(gt); bm = blurmap(st, (H, W))
    m = C // 2 + 64
    valid = np.zeros((H, W), bool); valid[m:H - m, m:W - m] = True
    picks = []
    # seam: most textured point on a vertical SwinIR seam line
    sx = seam_x[1]; col = np.where(valid[:, sx], tex[:, sx], -1); picks.append(('seam x=%d' % sx, int(col.argmax()), sx))
    for lab, sel in (('in-focus', bm < 0.4), ('defocused', bm > 3.5)):
        sc = np.where(valid & sel, tex, -1)
        if sc.max() > 0:
            y, x = np.unravel_index(sc.argmax(), sc.shape); picks.append((lab, int(y), int(x)))
    rows = []
    for lab, y, x in picks:
        tiles = []
        for t, im in cols:
            c = to8(im[y - C // 2:y + C // 2, x - C // 2:x + C // 2]).copy()
            cv2.putText(c, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(c)
        r = np.concatenate(tiles, 1)
        cv2.putText(r, f'{st} {lab} (y={y}, x={x})', (6, C - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        rows.append(r)
    cv2.imwrite(f'{V}/{st}.jpg', np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(st, [p[0] for p in picks])
