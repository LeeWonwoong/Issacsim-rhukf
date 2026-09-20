#!/usr/bin/env python3
# n6_probe16: 5ep 마다 16ep greedy 프로브 F1 — SWIRL vs Adam 10시드(중앙값·IQR) + 짝 차이(Δ, 우위 시드 수)
import json,glob,numpy as np
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R='/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/'
def probe(pat):
    out={}
    for d in sorted(glob.glob(R+pat)):
        H=json.load(open(d+'/hist.json'))['hist']; s=d.split('_s')[-1]
        out[s]=np.array([(h['ep'],h['probe_f1']) for h in H if h['ep']%5==0 and h['ep']>0])
    return out
S=probe('n6_probe16/a_s_buffer50000_s*'); A=probe('n6_probe16/a_a_buffer50000_s*'); seeds=sorted(set(S)&set(A))
ep=S[seeds[0]][:,0]; FS=np.array([S[s][:,1] for s in seeds]); FA=np.array([A[s][:,1] for s in seeds])
fig,ax=plt.subplots(1,2,figsize=(12,4.2))
for Y,c,l in ((FS,'#2f62e6','SWIRL'),(FA,'#d1483a','Adam')):
    ax[0].plot(ep,np.median(Y,0),color=c,label=f'{l} (n={len(seeds)})'); ax[0].fill_between(ep,np.percentile(Y,25,0),np.percentile(Y,75,0),color=c,alpha=.15)
ax[0].set_title('16ep greedy 프로브 F1 (5ep 마다, 중앙값·IQR)'); ax[0].set_xlabel('학습 에피소드'); ax[0].set_ylim(0,1); ax[0].legend(); ax[0].grid(alpha=.3)
D=FS-FA; ax[1].plot(ep,D.mean(0),color='k'); ax[1].fill_between(ep,D.mean(0)-D.std(0)/np.sqrt(len(seeds)),D.mean(0)+D.std(0)/np.sqrt(len(seeds)),color='k',alpha=.15); ax[1].axhline(0,color='gray',ls=':')
for i,e in enumerate(ep):
    if e in (25,50,100,150,195):
        p=wilcoxon(D[:,i]).pvalue if np.any(D[:,i]) else 1; ax[1].annotate(f'{int((D[:,i]>0).sum())}/{len(seeds)}\np={p:.3f}',(e,D[:,i].mean()),textcoords='offset points',xytext=(0,12),ha='center',fontsize=8)
ax[1].set_title('짝 차이 SWIRL − Adam (평균 ± SE, 우위 시드 수·Wilcoxon p)'); ax[1].set_xlabel('학습 에피소드'); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(R+'figs/fig_probe16.png',dpi=130)
for e in (25,50,75,100,150,195):
    i=int(np.argmin(abs(ep-e))); print(f'ep{int(ep[i])}: SWIRL {FS[:,i].mean():.3f} Adam {FA[:,i].mean():.3f} Δ {D[:,i].mean():+.3f} ({(D[:,i]>0).sum()}/{len(seeds)}, p={wilcoxon(D[:,i]).pvalue:.3f})')
