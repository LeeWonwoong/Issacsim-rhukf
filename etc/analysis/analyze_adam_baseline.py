#!/usr/bin/env python3
"""Adam baseline(results_adam_final) 지표 정리 — RHUKF 비교 기준점.
학습곡선 / delay 분포 / FAR(무공격·공격) / s별 delay / crash / loss.
사용: python3 analyze_adam_baseline.py [metrics.csv] [out.png]"""
import sys, csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os
from matplotlib import font_manager
for _fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
    if os.path.exists(_fp):
        try: font_manager.fontManager.addfont(_fp); plt.rcParams['font.family']=font_manager.FontProperties(fname=_fp).get_name()
        except Exception: pass
        break
plt.rcParams['axes.unicode_minus']=False

CSV = sys.argv[1] if len(sys.argv)>1 else 'results_adam_final/metrics_adam.csv'
OUT = sys.argv[2] if len(sys.argv)>2 else 'results_adam_final/baseline_summary.png'
TAIL = 60

rows=[r for r in csv.DictReader(open(CSV)) if r['episode'].isdigit()]
ep =np.array([int(r['episode']) for r in rows])
f1 =np.array([float(r['f1']) for r in rows]); rec=np.array([float(r['recall']) for r in rows])
prec=np.array([float(r['precision']) for r in rows]); rew=np.array([float(r['reward']) for r in rows])
loss=np.array([float(r['loss']) for r in rows]); dd=np.array([int(r['det_delay']) for r in rows])
far=np.array([float(r['fp_rate']) for r in rows]); cr=np.array([int(r['crashed']) for r in rows])
tp=np.array([int(r['tp']) for r in rows]); fn=np.array([int(r['fn']) for r in rows])
bs=np.array([float(r.get('bias_scale',0.0) or 0.0) for r in rows])

def sm(x,w=9):
    if len(x)<w: return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

# ── 수렴창(마지막 TAIL) 집계 ──
sl=slice(-TAIL,None)
atk_ep = (tp[sl]+fn[sl])>0            # 공격 에피소드
d_conv = dd[sl][dd[sl]>=0]
far_no = far[sl][~atk_ep]; far_at = far[sl][atk_ep]
print(f"===== Adam baseline (results_adam_final, 전체 {len(rows)}ep, 수렴창 마지막 {TAIL}ep) =====")
print(f"[delay] median={np.median(d_conv):.1f} mean={np.mean(d_conv):.1f} | d<=3={np.mean(d_conv<=3)*100:.0f}% d<=7={np.mean(d_conv<=7)*100:.0f}%")
h,_=np.histogram(d_conv,bins=[0,4,8,12,16,100])
print(f"[delay 분포] [0-3,4-7,8-11,12-15,16+] = {h.tolist()}  (둘째봉=12-15 항목)")
print(f"[FAR] 무공격={np.mean(far_no):.3f}(n={far_no.size}) | 공격={np.mean(far_at):.3f}(n={far_at.size}) | 전체 median={np.median(far[sl]):.3f}")
print(f"[정탐] F1={np.mean(f1[sl]):.3f} recall={np.mean(rec[sl]):.3f} precision={np.mean(prec[sl]):.3f}")
print(f"[생존] crash_rate={np.mean(cr[sl]):.2f} | reward mean={np.mean(rew[sl]):.1f}")

# ── s별 delay (공격 에피소드만) ──
print("\n[s별 delay] (수렴창 공격 에피소드, bias_scale 반올림 그룹)")
mask = (bs[sl]>0.5) & (dd[sl]>=0)
bsl=bs[sl][mask]; ddl=dd[sl][mask]
for lo,hi,tag in [(1.335,1.355,'1.34'),(1.355,1.385,'1.37'),(1.385,1.405,'1.40')]:
    m=(bsl>=lo)&(bsl<hi)
    if m.sum(): print(f"   s≈{tag}: median delay={np.median(ddl[m]):.1f} d<=7={np.mean(ddl[m]<=7)*100:.0f}% (n={m.sum()})")
    else: print(f"   s≈{tag}: (표본 없음)")

# ── 그림: 6패널 ──
fig,ax=plt.subplots(2,3,figsize=(15,8))
x=ep
ax[0,0].plot(x[:len(sm(f1))]+4,sm(f1),label='F1'); ax[0,0].plot(x[:len(sm(rec))]+4,sm(rec),label='recall')
ax[0,0].plot(x[:len(sm(prec))]+4,sm(prec),label='precision'); ax[0,0].set_title('학습곡선 F1/recall/precision (9ep 평활)')
ax[0,0].set_xlabel('episode'); ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)
ax[0,1].plot(x[:len(sm(rew))]+4,sm(rew),color='tab:green'); ax[0,1].set_title('reward 추이'); ax[0,1].set_xlabel('episode'); ax[0,1].grid(alpha=.3)
ax[0,2].plot(x[:len(sm(loss))]+4,sm(loss),color='tab:red'); ax[0,2].set_title('loss 곡선'); ax[0,2].set_xlabel('episode'); ax[0,2].grid(alpha=.3)
# delay vs ep (탐지된 것만)
dv=dd.astype(float); dv[dd<0]=np.nan
ax[1,0].scatter(x,dv,s=8,alpha=.4); ax[1,0].axhline(3,color='r',ls='--',lw=1,label='목표 d≤3'); ax[1,0].axhline(7,color='orange',ls=':',lw=1,label='데드라인상단')
ax[1,0].set_title('det_delay vs episode'); ax[1,0].set_xlabel('episode'); ax[1,0].set_ylabel('delay'); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=.3)
# delay 분포 (수렴창)
ax[1,1].bar(['0-3','4-7','8-11','12-15','16+'],h,color=['tab:green','tab:blue','gray','tab:red','black'])
ax[1,1].set_title(f'delay 분포 (수렴창 {TAIL}ep) — 둘째봉 확인'); ax[1,1].set_ylabel('에피소드 수')
for i,v in enumerate(h): ax[1,1].text(i,v+0.3,str(v),ha='center',fontsize=9)
# FAR 무공격 vs 공격
ax[1,2].bar(['무공격','공격'],[np.mean(far_no),np.mean(far_at)],color=['tab:green','tab:red'])
ax[1,2].axhline(0.02,color='k',ls='--',lw=1,label='목표 0.02'); ax[1,2].set_title('FAR 무공격 vs 공격 (수렴창)')
ax[1,2].set_ylabel('FAR'); ax[1,2].legend(fontsize=8)
for i,v in enumerate([np.mean(far_no),np.mean(far_at)]): ax[1,2].text(i,v+0.003,f'{v:.3f}',ha='center',fontsize=9)
fig.suptitle('Adam baseline (results_adam_final) — RHUKF 비교 기준점',fontsize=13)
fig.tight_layout(rect=(0,0,1,0.97)); os.makedirs(os.path.dirname(OUT) or '.',exist_ok=True); fig.savefig(OUT,dpi=130)
print(f"\n[그림] → {OUT}")
