"""통합 표: 공격δ × 바람ws → 표류(gt_err) · NIS(gyro/vel) · 생존.
axis_single(δ0.3-0.8 전ws) + G2(δ0.1,0.2 ws0,8) 병합. 단일축 지속공격.
표(txt) + 히트맵 4패널(표류/gyro/vel/생존).
사용: python consolidated_table.py"""
import os, csv, warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: float(min(np.log1p(np.sqrt(max(x,0.0))),3.0))
AT_S,AT_E=180,380

def load(D, dmin=0.0, dmax=9.0):
    det=defaultdict(lambda:defaultdict(list))  # (δ,ws)->{gt,g,v}
    if os.path.exists(f'{D}/sweep_detail.csv'):
        for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
            if r['policy']!='track': continue
            try:
                d=round(float(r['bias'])/4.36,2); ws=int(round(float(r['wind_speed']))); st=int(r['step'])
                if not(dmin<=d<=dmax): continue
                if AT_S<=st<AT_E:
                    det[(d,ws)]['gt'].append(float(r['gt_err'])); det[(d,ws)]['g'].append(comp(float(r['nis_g_raw']))); det[(d,ws)]['v'].append(comp(float(r['nis_v_raw'])))
            except: pass
    surv=defaultdict(lambda:[0,0])
    if os.path.exists(f'{D}/sweep_summary.csv'):
        for r in csv.DictReader(open(f'{D}/sweep_summary.csv')):
            if r['policy']!='track': continue
            try:
                d=round(float(r['bias'])/4.36,2); ws=int(round(float(r['wind_speed'])))
                if not(dmin<=d<=dmax): continue
                surv[(d,ws)][0]+=1 if r['survived']=='1' else 0; surv[(d,ws)][1]+=1
            except: pass
    return det,surv

# axis(δ0.3-0.8) + G2(δ0.1,0.2)
dA,sA=load('results_axis_single',0.25,0.9)
dW,sW=load('results_qr_G2_qg2e-3',0.05,0.22)
det={**dA,**dW}; surv={**sA,**sW}

DS=[0.1,0.2,0.3,0.5,0.7,0.8]; WINDS=[0,4,8,10,12]
def cell(d,ws,key):
    v=det.get((d,ws),{}).get(key,[])
    return np.median(v) if v else np.nan
def sv(d,ws):
    s=surv.get((d,ws),[0,0]); return s[0]/s[1] if s[1] else np.nan

L=[]; P=lambda s='':(L.append(str(s)),print(s))
P('='*94); P(' 통합 표: 공격 δ × 바람 ws → 표류(gt_err,m) / gyro / vel / 생존  (단일축 지속공격, 공격구간)'); P('='*94)
P(' 평시 표류 기준 ≈ 0.68m. δ0.1-0.2=G2, δ0.3-0.8=axis.')
for ws in WINDS:
    P(f'\n ── 바람 ws{ws} ──')
    P('  δ    | 표류(m) | gyro | vel  | 생존')
    for d in DS:
        gt=cell(d,ws,'gt'); g=cell(d,ws,'g'); v=cell(d,ws,'v'); s=sv(d,ws)
        if np.isnan(gt) and np.isnan(s): P(f'  {d:.1f}  |   -     |  -   |  -   |  -'); continue
        gts=f'{gt:5.2f}' if not np.isnan(gt) else '  -  '
        gs=f'{g:.2f}' if not np.isnan(g) else ' -  '; vs=f'{v:.2f}' if not np.isnan(v) else ' -  '
        ss=f'{s:.2f}' if not np.isnan(s) else ' - '
        P(f'  {d:.1f}  |  {gts}  | {gs} | {vs} |  {ss}')
open('results_consolidated_table.txt','w').write("\n".join(L))
print('\n[저장] results_consolidated_table.txt')

# 히트맵 4패널
fig,axes=plt.subplots(1,4,figsize=(22,4.6))
def grid(key,fn):
    M=np.full((len(DS),len(WINDS)),np.nan)
    for i,d in enumerate(DS):
        for j,ws in enumerate(WINDS): M[i,j]=fn(d,ws,key)
    return M
panels=[('표류 gt_err (m)',lambda d,w,k:cell(d,w,'gt'),'magma',0,8,'%.1f'),
        ('gyro NIS',lambda d,w,k:cell(d,w,'g'),'viridis',0,3,'%.1f'),
        ('vel NIS',lambda d,w,k:cell(d,w,'v'),'viridis',0,3,'%.1f'),
        ('생존율',lambda d,w,k:sv(d,w),'RdYlGn',0,1,'%.1f')]
for ax,(ttl,fn,cmap,vmn,vmx,fmt) in zip(axes,panels):
    M=np.full((len(DS),len(WINDS)),np.nan)
    for i,d in enumerate(DS):
        for j,ws in enumerate(WINDS): M[i,j]=fn(d,ws,None)
    ax.imshow(M,aspect='auto',cmap=cmap,vmin=vmn,vmax=vmx,origin='lower')
    ax.set_xticks(range(len(WINDS))); ax.set_xticklabels(WINDS); ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS)
    ax.set_xlabel('바람 ws'); ax.set_ylabel('공격 δ'); ax.set_title(ttl,fontsize=13)
    for i in range(len(DS)):
        for j in range(len(WINDS)):
            if np.isfinite(M[i,j]):
                col='white' if (cmap in ('magma','viridis') and M[i,j]<vmx*0.55) else 'black'
                ax.text(j,i,fmt%M[i,j],ha='center',va='center',fontsize=7,color=col)
fig.suptitle('공격 δ × 바람 ws → 표류(원치않은 이동) · NIS(관측) · 생존  (단일축 지속공격)',fontsize=14,y=1.03)
fig.tight_layout(); fig.savefig('consolidated_heatmap.png',dpi=110,bbox_inches='tight'); plt.close(fig)
print('[plot] consolidated_heatmap.png')
