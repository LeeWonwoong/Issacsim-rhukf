# capture_drift: 평시 추종오차(패턴×풍속) · 놓친 지속공격의 오차 성장(클래스×풍속) · 임무실패 임계 (e_max,T_max) 후보의 오탐/탐지 표
import numpy as np,glob,json,collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R='/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/'
E=[]
for f in sorted(glob.glob(R+'capture_drift/capture/ep*.npz')):
    z=np.load(f,allow_pickle=True); cols=[str(c) for c in z['cols']]; Rw=z['rows']; C={k:Rw[:,i] for i,k in enumerate(cols)}
    ev=json.loads(str(z['events'])) if 'events' in z.files else []
    E.append(dict(kind=str(z['kind']),ws=float(z['wind_speed']),pat=str(z['pattern']),g=C['gt_err'],st=C['step'].astype(int),ev=ev,reason=str(z['reason'])))
print('에피', len(E), dict(collections.Counter(e['kind'] for e in E)), 'reason', dict(collections.Counter(e['reason'] for e in E)))
tiers=[(0,3,'ws<3'),(3,6,'3–6'),(6,8,'6–8'),(8,11,'≥8')]
cl=[e for e in E if e['kind']=='none']
print('평시 에피', len(cl))
for lo,hi,lab in tiers:
    g=[e['g'] for e in cl if lo<=e['ws']<hi]
    if g: gg=np.concatenate(g); print(f'  평시 {lab:5s} n={len(g):2d} 중앙 {np.median(gg):.2f} 95% {np.percentile(gg,95):.2f} 99% {np.percentile(gg,99):.2f} 최대 {gg.max():.2f} | 에피 최대 중앙 {np.median([x.max() for x in g]):.2f}')
for p in ['waypoint','circle','figure8','scurve','aggressive']:
    g=[e['g'] for e in cl if e['pat']==p]
    if g: print(f'  평시 {p:10s} n={len(g)} 에피 최대 중앙 {np.median([x.max() for x in g]):.2f} 최대 {max(x.max() for x in g):.2f}')
# 성장 곡선
cls=['weak_persist','trans_persist','strong_persist']; col={'weak_persist':'#3a9a63','trans_persist':'#c98b2a','strong_persist':'#d1483a'}
fig,ax=plt.subplots(1,4,figsize=(17,4),sharey=True)
for j,(lo,hi,lab) in enumerate(tiers):
    cg=[e['g'] for e in cl if lo<=e['ws']<hi]
    if cg: cgg=np.concatenate(cg); ax[j].axhspan(0,np.percentile(cgg,99),color='gray',alpha=.15,label='평시 0–99%'); ax[j].axhline(cgg.max(),color='gray',ls=':',label='평시 최대')
    for c in cls:
        segs=[]
        for e in E:
            if e['kind']!=c or not (lo<=e['ws']<hi) or not e['ev']: continue
            s0=e['ev'][0]['start']-int(e['st'][0]); x=np.full(70,np.nan); seg=e['g'][max(s0-10,0):s0+60]; x[10-min(s0,10):10-min(s0,10)+len(seg)]=seg; segs.append(x)
        if len(segs)<2: continue
        S=np.array(segs); t=np.arange(-10,60); ax[j].plot(t,np.nanmedian(S,0),color=col[c],lw=1.6,label=f'{c} (n={len(segs)})'); ax[j].fill_between(t,np.nanpercentile(S,25,0),np.nanpercentile(S,75,0),color=col[c],alpha=.15)
    ax[j].axvline(0,color='k',lw=.8); ax[j].set_title(lab); ax[j].set_xlabel('온셋 기준 스텝'); ax[j].grid(alpha=.3); ax[j].legend(fontsize=7.5)
ax[0].set_ylabel('추종오차 gt_err [m] (중앙값·IQR)'); fig.suptitle('놓친 지속공격(track 고정)의 추종오차 성장 — 클래스 × 풍속 (Isaac capture_drift)'); fig.tight_layout(); fig.savefig(R+'figs/fig_drift_capture.png',dpi=120)
# 임계 후보 표: e_max × T_max=10 → 평시 오탐 실패율 / 클래스별 실패율
def fails(g,emax,T):
    x=(g>emax).astype(int); run=0
    for v in x:
        run=run+1 if v else 0
        if run>=T: return True
    return False
print('e_max\\T=10 | 평시 실패율 | weak | trans | strong   (에피 비율)')
for emax in (3,4,5,6,8):
    row=[np.mean([fails(e['g'],emax,10) for e in E if e['kind']==k]) if any(e['kind']==k for e in E) else np.nan for k in ['none']+cls]
    print(f'  {emax:4.0f}      | '+' | '.join(f'{v:.2f}' for v in row))
print('e_max\\T=5')
for emax in (3,4,5,6,8):
    row=[np.mean([fails(e['g'],emax,5) for e in E if e['kind']==k]) for k in ['none']+cls]; print(f'  {emax:4.0f}      | '+' | '.join(f'{v:.2f}' for v in row))
