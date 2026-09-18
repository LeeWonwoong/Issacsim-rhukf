"""CUSUM/임계 baseline (오프라인, 기존 sweep NIS). RL이 이겨야 할 대조군.
채널별(gyro/vel/combined) CUSUM: benign(clean+wind)으로 μ0,σ0 → z-score → S_k=max(0,S+z-κ).
h를 FPR 타깃으로 보정 → 탐지지연·미탐률 측정. gyro=쉬움 / vel=애매 대비.
사용: python cusum_baseline.py [results_axis_single|results_burst_single]"""
import os, csv, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)
D=sys.argv[1] if len(sys.argv)>1 else 'results_axis_single'
KAPPA=0.5   # CUSUM slack (half-sigma)

# 에피소드별: (step, gyro압축, vel압축, attack_active)
eps=defaultdict(list)
for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
    if r['policy']!='track': continue
    try:
        d=round(float(r['bias'])/4.36,2); ws=int(round(float(r['wind_speed'])))
        aa=int(float(r['attack_active'])) if r.get('attack_active','') not in ('','nan') else (1 if 180<=int(r['step'])<380 else 0)
        eps[(d,ws,r['episode']+r['cell_idx'])].append((int(r['step']),comp(float(r['nis_g_raw'])),comp(float(r['nis_v_raw'])),aa))
    except: pass

# benign 통계 (attack_active=0 전 스텝)
bg=defaultdict(list)  # ch -> vals
for k,rows in eps.items():
    for st,g,v,aa in rows:
        if aa==0: bg['g'].append(g); bg['v'].append(v)
mu={c:np.mean(bg[c]) for c in 'gv'}; sd={c:np.std(bg[c])+1e-9 for c in 'gv'}

def cusum_run(rows, ch, h):
    """반환: (탐지지연 or None, FP스텝수, 총benign스텝, alarm궤적)"""
    S=0.0; first_alarm=None; onset=None; fp=0; nb=0; alarms=[]
    idx={'g':1,'v':2}[ch]
    for st,g,v,aa in rows:
        x=[g,v][idx-1]; z=(x-mu[ch])/sd[ch]
        S=max(0.0,S+z-KAPPA); al=S>h; alarms.append((st,al))
        if aa==1 and onset is None: onset=st
        if al and first_alarm is None and onset is not None and st>=onset: first_alarm=st
        if aa==0:
            nb+=1
            if al: fp+=1
        if al: S=0.0   # reset after alarm (이벤트 카운트)
    delay=(first_alarm-onset) if (first_alarm is not None and onset is not None) else None
    return delay,fp,nb,alarms

def combined_run(rows,h):  # gyro OR vel (둘 중 하나라도 alarm)
    Sg=Sv=0.0; first=None; onset=None; fp=0; nb=0
    for st,g,v,aa in rows:
        zg=(g-mu['g'])/sd['g']; zv=(v-mu['v'])/sd['v']
        Sg=max(0,Sg+zg-KAPPA); Sv=max(0,Sv+zv-KAPPA); al=(Sg>h)or(Sv>h)
        if aa==1 and onset is None: onset=st
        if al and first is None and onset is not None and st>=onset: first=st
        if aa==0:
            nb+=1
            if al: fp+=1
        if al: Sg=Sv=0.0
    return ((first-onset) if (first and onset) else None),fp,nb

L=[]; P=lambda s='':(L.append(str(s)),print(s))
P('='*80); P(f' CUSUM baseline ({D}) — RL이 이겨야 할 대조군'); P('='*80)
P(f' benign μ/σ: gyro {mu["g"]:.2f}/{sd["g"]:.2f}  vel {mu["v"]:.2f}/{sd["v"]:.2f}  (κ={KAPPA})')

# h 스윕 → FPR vs 탐지지연 (채널별)
hs=np.logspace(-0.3,2.4,44)   # 0.5 ~ 250
res={}
for ch in ['g','v','comb']:
    rows_fpr=[]; rows_delay=[]; rows_miss=[]
    for h in hs:
        delays=[]; fps=0; nbs=0; miss=0; natk=0
        for k,rows in eps.items():
            rows=sorted(rows)
            if any(aa for _,_,_,aa in rows): natk+=1
            if ch=='comb': dly,fp,nb=combined_run(rows,h)
            else: dly,fp,nb,_=cusum_run(rows,ch,h)
            fps+=fp; nbs+=nb
            if any(aa for _,_,_,aa in rows):
                if dly is None: miss+=1
                else: delays.append(dly)
        rows_fpr.append(fps/max(nbs,1)); rows_delay.append(np.median(delays) if delays else np.nan); rows_miss.append(miss/max(natk,1))
    res[ch]=(np.array(rows_fpr),np.array(rows_delay),np.array(rows_miss))

# FPR 1% 타깃에서 각 채널 성능
P(f'\n [FPR≈1% 타깃에서 CUSUM 성능]  (탐지지연=온셋후 스텝, 미탐률)')
P('  채널     | 임계 h | 실FPR | 탐지지연(스텝) | 미탐률')
for ch,lbl in [('g','gyro'),('v','vel'),('comb','gyro∨vel')]:
    fpr,delay,miss=res[ch]
    i=np.argmin(np.abs(fpr-0.01))
    P(f'  {lbl:8s} | {hs[i]:5.1f}  | {fpr[i]*100:4.1f}% | {delay[i]:5.1f}         | {miss[i]*100:4.0f}%')
P('\n → gyro CUSUM은 빠르고 미탐 적음(easy). vel은 지연↑·미탐↑ 예상(POMDP).')
P('   RL은 (a)vel 애매 (b)간헐 복귀 (c)돌풍 FP 에서 이 baseline을 이겨야 함.')
open(f'{D}_cusum.txt','w').write("\n".join(L))
print(f'\n[저장] {D}_cusum.txt')

# plot: FPR vs 탐지지연 (채널별 ROC-유사)
fig,axes=plt.subplots(1,2,figsize=(14,5))
col={'g':'#C44E52','v':'#4C72B0','comb':'#55A868'}; nm={'g':'gyro','v':'vel','comb':'gyro∨vel'}
for ch in ['g','v','comb']:
    fpr,delay,miss=res[ch]
    axes[0].plot(fpr*100,delay,'o-',ms=3,color=col[ch],label=nm[ch])
    axes[1].plot(fpr*100,miss*100,'o-',ms=3,color=col[ch],label=nm[ch])
axes[0].set_xlabel('FPR (%)'); axes[0].set_ylabel('탐지지연 (스텝)'); axes[0].set_xscale('log'); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title('FPR vs 탐지지연')
axes[1].set_xlabel('FPR (%)'); axes[1].set_ylabel('미탐률 (%)'); axes[1].set_xscale('log'); axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_title('FPR vs 미탐률')
fig.suptitle(f'CUSUM baseline ({D}) — gyro 쉬움 / vel 어려움 (RL 목표선)',fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig(f'{D}_cusum.png',dpi=110,bbox_inches='tight'); plt.close(fig)
print(f'[plot] {D}_cusum.png')
