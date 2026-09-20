#!/usr/bin/env python3
# Isaac 본선 학습곡선(metrics_*.csv) SWIRL vs Adam, 시드별 짝 + surrogate 같은 시드 점선
import csv,glob,json,numpy as np,os,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R='/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/'
def load(d):
    f=glob.glob(d+'/metrics_*.csv')
    if not f: return None
    rows=list(csv.DictReader(open(f[0]))); 
    return dict(ep=np.array([int(r['episode']) for r in rows]), f1=np.array([float(r['f1']) for r in rows]), fpr=np.array([float(r['fp_rate']) for r in rows]),
                atk=np.array([(float(r['tp'])+float(r['fn']))>0 for r in rows]), rew=np.array([float(r['reward']) for r in rows]), st=np.array([float(r['steps']) for r in rows]))
def sm(x,w=20):
    m=~np.isnan(x); k=np.ones(w); num=np.convolve(np.where(m,x,0),k,'valid'); den=np.convolve(m.astype(float),k,'valid'); return np.where(den>0,num/np.maximum(den,1),np.nan)
fig,ax=plt.subplots(1,2,figsize=(12,4.2)); col={'swirl':'#2f62e6','adam':'#d1483a'}; c={'swirl':5.0,'adam':0.6}
summ=[]
for L in ('swirl','adam'):
    for d in sorted(glob.glob(R+f'isaac_v2/{L}_s*')):
        M=load(d); 
        if M is None or len(M['ep'])<50: continue
        s=d.split('_s')[-1]; f1=np.where(M['atk'],M['f1'],np.nan); cost=M['rew']/c[L]-0.5*(M['st']-3)
        ax[0].plot(M['ep'][19:],sm(f1),color=col[L],alpha=.9,label=f'{L} s{s} (Isaac)'); ax[1].plot(M['ep'][19:],sm(cost),color=col[L],alpha=.9)
        late=M['ep']>100; summ.append((L,s,np.nanmean(f1[late]),M['fpr'][late].mean(),cost[late].mean(),len(M['ep'])))
        # surrogate 같은 시드
        sd=R+(f'final_v4/a_sF_buffer50000_s{s}' if L=='swirl' else f'final_v4/a_aF_buffer50000_s{s}')
        if os.path.exists(sd+'/hist.json'):
            H=json.load(open(sd+'/hist.json'))['hist']; sf=np.array([h['f1'] if h['has_atk'] else np.nan for h in H]); sc=np.array([h['reward_cost'] for h in H])/c[L]
            ax[0].plot(np.arange(19,200),sm(sf),color=col[L],ls=':',alpha=.6,label=f'{L} s{s} (surrogate)'); ax[1].plot(np.arange(19,200),sm(sc),color=col[L],ls=':',alpha=.6)
ax[0].set_title('공격 에피소드 F1 (20ep 이동평균) — Isaac 실선 / surrogate 점선'); ax[1].set_title('보상 cost/c (20ep 이동평균)'); 
for a in ax: a.set_xlabel('에피소드'); a.grid(alpha=.3)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(R+'figs/fig_isaac_curves.png',dpi=130)
for r in summ: print(f"{r[0]:5s} s{r[1]} 후반(ep>100) F1 {r[2]:.3f} FPR {r[3]:.4f} cost/c {r[4]:+.2f} (기록 {r[5]}ep)")
