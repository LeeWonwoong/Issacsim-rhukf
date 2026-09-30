"""Isaac 스텝 로그 → 보상과 무관한 공통 운용비용/에피 = (c_fa·FP + c_d·FN − B·탐지 사건)/에피 (낮을수록 좋음).
라벨 규약: 행의 prev_action(직전 행동) × atk_flag. 사건 = atk_flag 연속 구간, 탐지 = 구간 안에 hover 가 한 번이라도.
사용: python isaac_cost.py 이름=런폴더 ...   (가중: vfinal (1.5,1.5,4) · (5,5,10))"""
import glob, sys, numpy as np
W = {'vf': (1.5, 1.5, 4.0), 'x5': (5.0, 5.0, 10.0)}
def run_cost(d, ep_lo=1, ep_hi=200):
    tot = {k: 0.0 for k in W}; n = 0
    for f in sorted(glob.glob(d + '/steps/ep*.npz')):
        ep = int(f[-8:-4])
        if not (ep_lo <= ep <= ep_hi): continue
        z = np.load(f, allow_pickle=True); c = [str(x) for x in z['cols']]; R = z['rows']
        a = R[:, c.index('prev_action')].astype(int); atk = R[:, c.index('atk_flag')].astype(int)
        fp = int(((a == 1) & (atk == 0)).sum()); fn = int(((a == 0) & (atk == 1)).sum())
        det = 0; i = 0
        while i < len(atk):
            if atk[i]:
                j = i
                while j < len(atk) and atk[j]: j += 1
                det += int((a[i:j] == 1).any()); i = j
            else: i += 1
        for k, (cfa, cd, B) in W.items(): tot[k] += cfa * fp + cd * fn - B * det
        n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}, n
if __name__ == '__main__':
    for arg in sys.argv[1:]:
        nm, d = arg.split('=', 1); r, n = run_cost(d); print(f'{nm:14s} n_ep {n}  비용 vf {r["vf"]:.2f}  x5 {r["x5"]:.2f}')
