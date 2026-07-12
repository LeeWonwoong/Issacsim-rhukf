#!/usr/bin/env python3
"""Adam baseline seed 반복 집계 — RHUKF 공식 비교 기준점.
seed별 + 종합(평균±std). delay median 분산으로 baseline 신뢰도 판정.
delay 7(구,no-clip) vs 14(clip) 이 클립 효과인지 seed 변동인지 분리.
사용: python3 aggregate_adam_seeds.py [seed_dir_glob] [out.png]"""
import sys, csv, glob, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
    if os.path.exists(_fp):
        try: font_manager.fontManager.addfont(_fp); plt.rcParams['font.family']=font_manager.FontProperties(fname=_fp).get_name()
        except Exception: pass
        break
plt.rcParams['axes.unicode_minus']=False

DIRS = sorted(glob.glob(sys.argv[1] if len(sys.argv)>1 else 'results_adam_seed[0-9]'))
OUT  = sys.argv[2] if len(sys.argv)>2 else 'results_adam_seeds_summary.png'
TAIL = 60
BINS = [0,4,8,12,16,100]; BINLBL=['0-3','4-7','8-11','12-15','16+']

def load(d):
    p=os.path.join(d,'metrics_adam.csv')
    if not os.path.exists(p): return None
    rows=[r for r in csv.DictReader(open(p)) if r['episode'].isdigit()]
    if len(rows)<TAIL: return None
    sl=rows[-TAIL:]
    dd=np.array([int(r['det_delay']) for r in sl]); dcv=dd[dd>=0]
    tp=np.array([int(r['tp']) for r in sl]); fn=np.array([int(r['fn']) for r in sl])
    far=np.array([float(r['fp_rate']) for r in sl]); atk=(tp+fn)>0
    bs=np.array([float(r.get('bias_scale',0) or 0) for r in sl])
    sdelay={}
    for lo,hi,tag in [(1.335,1.355,'1.34'),(1.355,1.385,'1.37'),(1.385,1.405,'1.40')]:
        m=(bs>=lo)&(bs<hi)&(dd>=0)
        sdelay[tag]=(np.median(dd[m]) if m.sum() else np.nan, int(m.sum()))
    return dict(
        delay_med=np.median(dcv), delay_mean=np.mean(dcv),
        d3=np.mean(dcv<=3)*100, d7=np.mean(dcv<=7)*100,
        hist=np.histogram(dcv,bins=BINS)[0],
        far_no=np.mean(far[~atk]) if (~atk).any() else np.nan,
        far_at=np.mean(far[atk]) if atk.any() else np.nan,
        f1=np.mean([float(r['f1']) for r in sl]),
        crash=np.mean([int(r['crashed']) for r in sl]),
        sdelay=sdelay, ep=len(rows),
        f1curve=np.array([float(r['f1']) for r in rows]))

res={}
for d in DIRS:
    r=load(d)
    if r: res[os.path.basename(d)]=r
if not res:
    print("완료된 seed 결과 없음 (metrics_adam.csv 부족)"); sys.exit(0)

print(f"===== Adam baseline seed 반복 집계 ({len(res)} seeds, 각 수렴창 {TAIL}ep) =====\n")
print(f"{'seed':<18}{'delay med':>9}{'mean':>6}{'d<=3':>6}{'d<=7':>6}{'  hist['+','.join(BINLBL)+']':>28}{'무FAR':>7}{'공FAR':>7}{'F1':>6}{'crash':>6}")
for k,r in res.items():
    print(f"{k:<18}{r['delay_med']:>9.1f}{r['delay_mean']:>6.1f}{r['d3']:>5.0f}%{r['d7']:>5.0f}%{str(r['hist'].tolist()):>28}{r['far_no']:>7.3f}{r['far_at']:>7.3f}{r['f1']:>6.2f}{r['crash']:>6.2f}")

meds=np.array([r['delay_med'] for r in res.values()])
print(f"\n★ delay median: 평균 {meds.mean():.1f} ± {meds.std():.1f}  (min {meds.min():.0f}, max {meds.max():.0f})")
print(f"   → 분산 {'작음 → baseline 신뢰, 단일값 비교가능' if meds.std()<=2.5 else '큼 → RHUKF도 다seed 필수, 통계검정 필요'}")
for m,lbl in [('far_no','무공격FAR'),('far_at','공격FAR'),('f1','F1'),('crash','crash')]:
    a=np.array([r[m] for r in res.values()]); print(f"   {lbl}: {a.mean():.3f} ± {a.std():.3f}")

print("\n[s별 delay median] (약한공격 1.34 느림·강한공격 1.40 빠름이 seed 무관 일관?)")
for tag in ['1.34','1.37','1.40']:
    vals=[r['sdelay'][tag][0] for r in res.values() if not np.isnan(r['sdelay'][tag][0])]
    if vals: print(f"   s={tag}: seed별 {[f'{v:.0f}' for v in vals]}  평균 {np.mean(vals):.1f}±{np.std(vals):.1f}")

havg=np.mean([r['hist'] for r in res.values()],axis=0)
print(f"\n[delay 분포 평균] {BINLBL} = {[f'{v:.1f}' for v in havg]}  (둘째봉=12-15)")
print(f"\n[판정] delay 7(구 no-clip) vs {meds.mean():.0f}(clip, seed평균): ", end='')
print("seed평균이 14 근처면 클립효과 확정 / 7 포함 넓게 퍼지면 seed변동 지배")

# ── 그림 ──
fig,ax=plt.subplots(2,2,figsize=(13,9))
# delay median seed별
ax[0,0].bar([k.replace('results_adam_','') for k in res],meds,color='tab:blue')
ax[0,0].axhline(meds.mean(),color='r',ls='--',label=f'평균 {meds.mean():.1f}±{meds.std():.1f}')
ax[0,0].axhline(7,color='green',ls=':',label='구 baseline(no-clip) 7')
ax[0,0].set_title('delay median — seed별'); ax[0,0].set_ylabel('delay'); ax[0,0].legend(fontsize=8)
# delay 분포 평균
ax[0,1].bar(BINLBL,havg,color=['tab:green','tab:blue','gray','tab:red','black'])
ax[0,1].set_title('delay 분포 (seed 평균)'); ax[0,1].set_ylabel('ep 수')
for i,v in enumerate(havg): ax[0,1].text(i,v+0.2,f'{v:.0f}',ha='center',fontsize=9)
# s별 delay
x=np.arange(3); w=0.8/max(len(res),1)
for i,(k,r) in enumerate(res.items()):
    ys=[r['sdelay'][t][0] for t in ['1.34','1.37','1.40']]
    ax[1,0].bar(x+i*w,ys,w,label=k.replace('results_adam_',''))
ax[1,0].set_xticks(x+w*len(res)/2); ax[1,0].set_xticklabels(['s=1.34','s=1.37','s=1.40'])
ax[1,0].axhline(7,color='orange',ls=':',lw=1); ax[1,0].set_title('s별 delay median (약공격 느림 일관?)'); ax[1,0].set_ylabel('delay'); ax[1,0].legend(fontsize=7)
# F1 학습곡선 안정성
for k,r in res.items():
    c=r['f1curve'];
    if len(c)>=9: c=np.convolve(c,np.ones(9)/9,mode='valid')
    ax[1,1].plot(c,label=k.replace('results_adam_',''),alpha=.8)
ax[1,1].set_title('F1 학습곡선 (seed별 안정성)'); ax[1,1].set_xlabel('episode'); ax[1,1].set_ylabel('F1'); ax[1,1].legend(fontsize=7); ax[1,1].grid(alpha=.3)
fig.suptitle(f'Adam baseline seed 반복 ({len(res)} seeds) — RHUKF 공식 비교 기준점',fontsize=13)
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig(OUT,dpi=130)
print(f"\n[그림] → {OUT}")
