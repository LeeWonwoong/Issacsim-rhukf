#!/usr/bin/env python3
# 공통 정의 loss 패널: FVU = E[TD²]/Var(target) (in-sample, 학습 배치). SWIRL loss=mean(residual²)=E[TD²]; Adam loss=Huber(β1)=0.5·TD² (|TD|<1 영역, c0.6 에서 |TD|~0.1) → MSE=2·loss.
import json,glob,numpy as np,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R='/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/'
def sm(x,w=10): return np.convolve(x,np.ones(w)/w,'valid')
S={'swirl':('final_v4/a_sF_buffer50000_s*',1.0,'#2f62e6'),'adam':('final_v4/a_aF_buffer50000_s*',2.0,'#d1483a')}
fig,ax=plt.subplots(1,3,figsize=(15,4.2)); out=[]
for L,(pat,mult,col) in S.items():
    F=[];Lo=[];Tv=[]
    for d in sorted(glob.glob(R+pat)):
        H=json.load(open(d+'/hist.json'))['hist']; lo=np.array([h['loss'] for h in H]); tv=np.array([h['tvar'] for h in H])
        F.append(sm(mult*lo/np.maximum(tv,1e-9))); Lo.append(sm(lo)); Tv.append(sm(tv))
    F=np.array(F); Lo=np.array(Lo); Tv=np.array(Tv); x=np.arange(9,200)
    for a,Y,t in ((ax[0],F,'FVU'),(ax[1],Lo,'loss'),(ax[2],Tv,'tvar')):
        med=np.median(Y,0); a.plot(x,med,color=col,label=f'{L} (n={len(Y)})'); a.fill_between(x,np.percentile(Y,25,0),np.percentile(Y,75,0),color=col,alpha=.15)
    out.append((L,len(F),np.median(F[:,-100:]),np.median(F[:,:20]),np.median(Lo[:,-100:]),np.median(Tv[:,-100:])))
ax[0].set_title('공통 정의: FVU = E[TD²] / Var(TD 타깃)  (10ep 이동평균, 중앙값·IQR)'); ax[0].axhline(1,color='gray',ls=':',lw=1); ax[0].set_ylim(0,1.2)
ax[1].set_title('각자 정의 loss (SWIRL: 잔차² 평균 @c5 / Adam: Huber @c0.6)'); ax[1].set_yscale('log')
ax[2].set_title('TD 타깃 분산 Var(target) (각자 보상 스케일)'); ax[2].set_yscale('log')
for a in ax: a.set_xlabel('에피소드'); a.grid(alpha=.3); a.legend(fontsize=9)
fig.tight_layout(); fig.savefig(R+'figs/fig_loss_fvu.png',dpi=130)
for r in out: print(f'{r[0]:5s} n={r[1]} FVU ep0-20 {r[3]:.3f} → ep100-199 {r[2]:.3f} | loss(자기단위) {r[4]:.4f} | tvar {r[5]:.3f}')
