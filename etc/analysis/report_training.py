#!/usr/bin/env python3
"""report_training.py — 학습 metrics CSV 요약 (reward/loss/F1/지연/FP 학습곡선 + 최종구간).
  python3 report_training.py <outdir> [--agent adam|rhukf]
공격/무공격 에피소드 분리, 이동평균, 최종 50ep 성능, 크래시율, TD 첨도.
"""
import csv, os, sys
import numpy as np

def load(outdir, agent):
    p = os.path.join(outdir, f'metrics_{agent}.csv')
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            rows.append({k: r[k] for k in r})
        except Exception:
            pass
    return rows

def col(rows, k, cast=float):
    out = []
    for r in rows:
        try: out.append(cast(r[k]))
        except Exception: out.append(float('nan'))
    return np.array(out)

def ma(x, w=20):
    if len(x) < 1: return x
    w = min(w, len(x))
    return np.convolve(x, np.ones(w)/w, mode='valid')

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'results_adam_fdi_0819'
    agent = 'adam'
    if '--agent' in sys.argv: agent = sys.argv[sys.argv.index('--agent')+1]
    rows = load(outdir, agent)
    n = len(rows)
    if n == 0:
        print("no rows"); return
    ep = col(rows,'episode'); rew = col(rows,'reward'); loss = col(rows,'loss')
    f1 = col(rows,'f1'); prec = col(rows,'precision'); rec = col(rows,'recall')
    fpr = col(rows,'fp_rate'); dly = col(rows,'det_delay'); crash = col(rows,'crashed')
    bias = col(rows,'bias_scale'); exk = col(rows,'td_exkurt'); eps = col(rows,'epsilon')
    atk = bias > 1e-6      # 공격 에피소드
    ben = ~atk

    print(f"\n{'='*72}\n  {outdir} [{agent}] — 학습 요약 (총 {n} 에피소드)\n{'='*72}")
    print(f"  공격 에피={atk.sum()} ({100*atk.mean():.0f}%)  무공격={ben.sum()} ({100*ben.mean():.0f}%)")
    print(f"  크래시={int(np.nansum(crash))} / {n}  (crashed=1 비율 {100*np.nanmean(crash):.1f}%)")
    print(f"  최종 ε={eps[-1]:.3f}")

    def block(name, mask):
        m = mask & ~np.isnan(rew)
        if m.sum()==0: print(f"\n  [{name}] 없음"); return
        last = np.where(m)[0][-min(50,m.sum()):]
        print(f"\n  ── [{name}] 전체 {m.sum()} ep / 최종 {len(last)} ep ──")
        print(f"     reward  전체 {np.nanmean(rew[m]):8.1f}  →  최종 {np.nanmean(rew[last]):8.1f}")
        print(f"     loss    전체 {np.nanmean(loss[m]):8.3f}  →  최종 {np.nanmean(loss[last]):8.3f}")
        if name.startswith('공격'):
            vf = f1[m]; vf=vf[~np.isnan(vf)]; lf=f1[last]; lf=lf[~np.isnan(lf)]
            print(f"     F1      전체 {vf.mean():8.3f}  →  최종 {lf.mean():8.3f}")
            print(f"     precision 최종 {np.nanmean(prec[last]):.3f}  recall 최종 {np.nanmean(rec[last]):.3f}")
            dd = dly[last]; dd = dd[dd>=0]
            print(f"     det_delay 최종 {dd.mean():.2f} 스텝 (검출된 {len(dd)}/{len(last)} ep)" if len(dd) else "     det_delay: 검출 없음")
        else:
            print(f"     fp_rate 전체 {np.nanmean(fpr[m]):8.3f}  →  최종 {np.nanmean(fpr[last]):8.3f}")

    block('공격(탐지 성능)', atk)
    block('무공격(오탐 FP)', ben)

    print(f"\n  ── 학습곡선 (20-ep 이동평균, 균등 8점 샘플) ──")
    mr = ma(rew,20); ml = ma(loss,20)
    idx = np.linspace(0, len(mr)-1, min(8,len(mr))).astype(int) if len(mr)>0 else []
    print(f"     {'ep':>5s} " + ' '.join(f'{int(ep[min(i+19,n-1)]):>7d}' for i in idx))
    print(f"     {'rew':>5s} " + ' '.join(f'{mr[i]:7.0f}' for i in idx))
    print(f"     {'loss':>5s} " + ' '.join(f'{ml[i]:7.2f}' for i in idx))
    xk = exk[~np.isnan(exk)]
    print(f"\n  TD 초과첨도(공정비교): 전체평균 {xk.mean():.3f}  최근10ep {xk[-10:].mean():.3f}" if len(xk) else "")
    print()

if __name__ == '__main__':
    main()
