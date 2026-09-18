"""Adam 옵티마이저 sweep 분석: lr{3e-4,5e-4,1e-3}×amsgrad{on,off}.
각 config eval_history.npz → 학습곡선(탐지지연·FP·생존·정탐률 vs 에피소드) + 최종성능 비교표.
목표: 지연↓ FP↓ 정탐↑. 사용: python adam_opt_analysis.py"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
CFGS=[('base_lr3e4_ams1_obs0_net24','baseline (3e-4·ams+·obs-·net24)'),
      ('var_lr1e3_ams1_obs0_net24','lr 1e-3'),
      ('var_lr3e4_ams0_obs0_net24','amsgrad off'),
      ('var_lr3e4_ams1_obs1_net24','obs /3.5 정규화'),
      ('var_lr3e4_ams1_obs0_net16','net [16,16]')]

def load(tag):
    p=f'results_opt_{tag}/eval_history.npz'
    if not os.path.exists(p): return None
    try: eh=np.load(p,allow_pickle=True)['eval_history']
    except: return None
    ep=[e['train_episode'] for e in eh]; sv=[e['survival_rate'] for e in eh]
    dd=[e['mean_det_delay'] for e in eh]; fa=[e['mean_false_alarm_rate'] for e in eh]
    # 정탐률(detection rate) = per_scenario 중 det_delay>=0 비율
    det=[]
    for e in eh:
        ps=e.get('per_scenario',[]);
        det.append(np.mean([1 if r.get('det_delay',-1)>=0 else 0 for r in ps]) if ps else np.nan)
    return dict(ep=np.array(ep),sv=np.array(sv),dd=np.array(dd),fa=np.array(fa),det=np.array(det))

data={t:load(t) for t,_ in CFGS}
have=[(t,l) for t,l in CFGS if data[t] is not None]
L=[]; P=lambda s='':(L.append(str(s)),print(s))
P('='*78); P(' Adam 옵티마이저 sweep — 최종성능 (마지막 eval, 목표: 지연↓ FP↓ 정탐/생존↑)'); P('='*78)
P('  config      | 탐지지연 | FP율  | 정탐률 | 생존율 | (마지막 eval ep)')
for t,l in CFGS:
    d=data[t]
    if d is None: P(f'  {l:11s} | (미완)'); continue
    i=-1
    P(f'  {l:11s} | {d["dd"][i]:6.1f}   | {d["fa"][i]:.3f} | {d["det"][i]:.2f}   | {d["sv"][i]:.2f}   | ep{int(d["ep"][i])}')
open('results_adam_opt_summary.txt','w').write("\n".join(L))
print('\n[저장] results_adam_opt_summary.txt')

if have:
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    metrics=[('dd','탐지지연 (스텝, ↓)'),('fa','오탐률 FP (↓)'),('det','정탐률 (↑)'),('sv','생존율 (↑)')]
    cmap=plt.cm.tab10
    for ax,(key,ttl) in zip(axes.flat,metrics):
        for k,(t,l) in enumerate(have):
            d=data[t]; ls='-' if 'ams1' in t else '--'
            ax.plot(d['ep'],d[key],ls,color=cmap(k%10),lw=1.6,marker='o',ms=3,label=l)
        ax.set_xlabel('학습 에피소드'); ax.set_title(ttl,fontsize=12); ax.grid(alpha=0.3); ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Adam 옵티마이저 sweep 학습곡선 (실선=amsgrad on, 점선=off)',fontsize=14,y=1.01)
    fig.tight_layout(); fig.savefig('adam_opt_curves.png',dpi=110,bbox_inches='tight'); plt.close(fig)
    print('[plot] adam_opt_curves.png')
