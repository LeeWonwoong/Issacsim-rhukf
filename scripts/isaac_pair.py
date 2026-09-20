#!/usr/bin/env python3
# Isaac 본선 시드별 짝비교(SWIRL vs Adam): 후반(ep>100) 스텝 F1/recall/FPR(metrics csv) · 사건 단위 탐지율/지연/오경보에피(steps/) · 교차평가(eval_surrogate.log) + 3시드 평균 막대 그림
import csv,glob,json,re,os,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R='/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/'; EP0=100
CC={'swirl':5.0,'adam':0.6}
def steps_metrics(d):
    res=dict(n=0,det=0,dl=[],ncl=0,nfa=0,wb=[0,0],cost=[]); cc=CC[d.split('/')[-1].split('_')[0]]
    for f in sorted(glob.glob(d+'/steps/ep*.npz')):
        if int(f[-8:-4])<=EP0: continue
        z=np.load(f,allow_pickle=True); ev=json.loads(str(z['events'])) if z['events'].shape==() else []
        Rw=z['rows']; cols=[str(c) for c in z['cols']]; C={k:Rw[:,i] for i,k in enumerate(cols)}
        st=C['step'].astype(int); pa=C['prev_action'].astype(int); a=np.full(st.max()+2,-1); a[st[:-1]]=pa[1:]
        res['cost'].append(C['reward'].sum()/cc-0.5*len(st))   # 학습중 cost/c = 생존보상 뺀 운용비용(오경보·지연·탐지보너스)
        if not ev: res['ncl']+=1; res['nfa']+=int((a[a>=0]>0).any())
        for e in ev:
            lab=a[e['start']+1:min(e['end']+1,len(a))]; lab=lab[lab>=0]; det=int((lab>0).any()) if len(lab) else 0
            res['n']+=1; res['det']+=det
            if det: res['dl'].append(int(np.argmax(lab>0)))
            if e['cls']=='weak_burst': res['wb'][0]+=det; res['wb'][1]+=1
    return res
def csv_metrics(d):
    f=glob.glob(d+'/metrics_*.csv')[0]; rows=list(csv.DictReader(open(f)))
    ep=np.array([int(r['episode']) for r in rows]); f1=np.array([float(r['f1']) for r in rows]); rec=np.array([float(r['recall']) for r in rows]); fpr=np.array([float(r['fp_rate']) for r in rows]); atk=np.array([(float(r['tp'])+float(r['fn']))>0 for r in rows])
    m=(ep>EP0)&atk; return dict(f1=f1[m].mean(),rec=rec[m].mean(),fpr=fpr[ep>EP0].mean(),win=[f1[(ep>a)&(ep<=b)&atk].mean() for a,b in [(0,25),(25,50),(50,100),(100,150),(150,200)]],crash=int(sum(float(r['crashed']) for r in rows)),hard=sum(1 for l in open(d+'/train_stdout.log') if 'HARD_RESET' in l))
def xeval(d):
    try: t=open(d+'/eval_surrogate.log').read().strip().splitlines()[-1]
    except Exception: return None
    m=re.search(r'F1 ([\d.]+) .*?사건탐지 ([\d.]+)/\d+ .*?오경보에피 ([\d.]+)',t); return tuple(map(float,m.groups())) if m else None
T={}; 
for L in ('swirl','adam'):
    for d in sorted(glob.glob(R+f'isaac_v2/{L}_s*')):
        if not os.path.exists(d+'/RUN_DONE'): continue
        s=d.split('_s')[-1]; c=csv_metrics(d); e=steps_metrics(d); x=xeval(d)
        T[(L,s)]=dict(c, cost=np.mean(e['cost']), ev_det=e['det']/max(e['n'],1), delay=np.mean(e['dl']) if e['dl'] else np.nan, fa_ep=e['nfa']/max(e['ncl'],1), wb=e['wb'][0]/max(e['wb'][1],1), x_f1=x[0] if x else np.nan, x_ev=x[1] if x else np.nan, x_fa=x[2] if x else np.nan)
seeds=sorted({s for (L,s) in T if ('swirl',s) in T and ('adam',s) in T})
keys=[('f1','후반 F1'),('cost','cost/c'),('rec','후반 recall'),('fpr','후반 FPR'),('ev_det','사건탐지'),('delay','지연'),('fa_ep','오경보에피'),('wb','weakB 탐지'),('x_f1','교차 F1'),('x_fa','교차 FA에피')]
print('시드 | '+' | '.join(f'{n} S/A' for _,n in keys)+' | HARD S/A | 창평균 F1 S / A')
for s in seeds:
    S,A=T[('swirl',s)],T[('adam',s)]
    print(f"{s} | "+' | '.join(f"{S[k]:.3f}/{A[k]:.3f}" for k,_ in keys)+f" | {S['hard']}/{A['hard']} | {'/'.join(f'{v:.2f}' for v in S['win'])} vs {'/'.join(f'{v:.2f}' for v in A['win'])}")
if len(seeds)>=2:
    print('평균 | '+' | '.join(f"{np.mean([T[('swirl',s)][k] for s in seeds]):.3f}/{np.mean([T[('adam',s)][k] for s in seeds]):.3f} (Δ{np.mean([T[('swirl',s)][k]-T[('adam',s)][k] for s in seeds]):+.3f}, {sum(T[('swirl',s)][k]>T[('adam',s)][k] for s in seeds)}/{len(seeds)})" for k,_ in keys))
    fig,ax=plt.subplots(1,5,figsize=(16,3.8)); col={'swirl':'#2f62e6','adam':'#d1483a'}
    for a,(k,n) in zip(ax,[('f1','후반 F1 (ep>100, 공격 에피)'),('cost','학습중 cost/c (ep>100)'),('ev_det','사건 탐지율 (ep>100)'),('fa_ep','오경보 에피 비율'),('x_f1','교차평가 F1 (surrogate greedy)')]):
        for i,L in enumerate(('swirl','adam')):
            v=[T[(L,s)][k] for s in seeds]; a.bar(i,np.mean(v),color=col[L],alpha=.85,label=L); a.errorbar(i,np.mean(v),yerr=np.std(v),color='k',capsize=4); a.scatter([i]*len(v),v,color='k',s=12,zorder=3)
        a.set_xticks([0,1]); a.set_xticklabels(['SWIRL','Adam']); a.set_title(n,fontsize=10); a.grid(axis='y',alpha=.3)
    fig.suptitle(f'Isaac 본선 (speed 2.5, 시드 {", ".join(seeds)}; 점=시드)'); fig.tight_layout(); fig.savefig(R+'figs/fig_isaac_bars.png',dpi=130)
