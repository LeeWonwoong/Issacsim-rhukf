# -*- coding: utf-8 -*-
"""surrogate 그리드 결과 정리: TOP-N + 축별 한계효과(marginal) + 세트별 요약."""
import json, glob, numpy as np, sys
rows=[]
for f in glob.glob('/tmp/surr_grid_shard*.jsonl'):
    for line in open(f):
        try:
            r=json.loads(line)
            if 'error' not in r and r.get('m_F1') is not None: rows.append(r)
        except Exception: pass
print(f"완료 {len(rows)}/192 config\n")
if not rows: sys.exit()
rows.sort(key=lambda r:(-r['m_F1'], -r['m_reward']))
print("── TOP 10 (F1 우선·reward 타이브레이크) ──")
print(f"{'set':>4s} {'α':>4s} {'R':>4s} {'Q':>5s} {'N':>2s} {'pΔ':>5s} | {'F1':>6s} {'P':>6s} {'Rc':>6s} {'delay':>5s} {'fpr':>6s} {'rwd':>6s} {'loss':>5s}")
for r in rows[:10]:
    st='A' if r['ui']==4 else 'B'
    print(f"{st:>4s} {r['alpha']:4.1f} {r['R_axis']:4.1f} {r['Q_axis']:>5s} {r['N_axis']:2d} {r['PD_axis']:5.2f} | "
          f"{r['m_F1']:6.3f} {r['m_P']:6.3f} {r['m_R']:6.3f} {r['m_delay']:5.2f} {r['m_fpr']:6.3f} {r['m_reward']:6.1f} {r['m_loss']:5.2f}")
print("\n── 축별 한계효과 (해당 값 config들의 F1 평균 ± std) ──")
for ax,vals in [('set(ui)',[4,1]),('alpha',[0.1,0.5,0.9]),('R_axis',[1.0,2.0]),('Q_axis',['1e-2','1e-3']),('N_axis',[5,6]),('PD_axis',[0.01,0.03,0.05,0.1])]:
    parts=[]
    for v in vals:
        sel=[r['m_F1'] for r in rows if (r['ui']==v if ax=='set(ui)' else r[ax]==v)]
        lab = ('A' if v==4 else 'B') if ax=='set(ui)' else v
        parts.append(f"{lab}: {np.mean(sel):.3f}±{np.std(sel):.3f}(n{len(sel)})" if sel else f"{lab}: -")
    print(f"  {ax:<8s} " + " | ".join(parts))
print("\n── 하위 5 (뭘 피할지) ──")
for r in rows[-5:]:
    st='A' if r['ui']==4 else 'B'
    print(f"  {st} α{r['alpha']} R{r['R_axis']} Q{r['Q_axis']} N{r['N_axis']} pΔ{r['PD_axis']}: F1 {r['m_F1']:.3f} rwd {r['m_reward']:.1f}")
