"""전 인자 분해 faceted plot: 궤적(4) × 축모드(2) × δ × ws.
생존 / 궤적이탈gt_err / gyro NIS / vel NIS 각각 2행(축)×4열(패턴) δ×ws 히트맵.
사용: python facet_plots.py"""
import os, csv
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: float(min(np.log1p(np.sqrt(max(x,0.0))),3.0))
DIV={'single':4.36,'balanced':3.0829}
PATS=['circle','figure8','waypoint','aggressive']
AXES=[('single','단일축(roll)'),('balanced','동시축(roll=pitch)')]
WINDS=[0,4,5,6,7,8,9,10,11,12]; DS=[0.3,0.4,0.5,0.6,0.7,0.8]
AT_S,AT_E=180,380

def load_sum(tag):
    fn=f'results_axis_{tag}/sweep_summary.csv'
    return list(csv.DictReader(open(fn))) if os.path.exists(fn) else []
def load_det(tag):
    fn=f'results_axis_{tag}/sweep_detail.csv'; ep=defaultdict(list)
    if not os.path.exists(fn): return ep
    for r in csv.DictReader(open(fn)):
        if r['policy']!='track': continue
        try:
            d=round(float(r['bias'])/DIV[tag],1); ws=int(round(float(r['wind_speed'])))
            if d<0.1: continue
            st=int(r['step'])
            if AT_S<=st<AT_E:
                ep[(r['pattern'],ws,d)].append((comp(float(r['nis_g_raw'])),comp(float(r['nis_v_raw'])),float(r['gt_err'])))
        except: pass
    return ep

# 데이터: 생존(summary), NIS/gt_err(detail)
SURV={t:defaultdict(lambda:[0,0]) for t,_ in AXES}
for tag,_ in AXES:
    for r in load_sum(tag):
        if r['policy']!='track': continue
        d=round(float(r['bias'])/DIV[tag],1); ws=int(round(float(r['wind_speed'])))
        SURV[tag][(r['pattern'],d,ws)][0]+=1 if r['survived']=='1' else 0
        SURV[tag][(r['pattern'],d,ws)][1]+=1
DET={t:load_det(t) for t,_ in AXES}

def grid_surv(tag,pat):
    M=np.full((len(DS),len(WINDS)),np.nan)
    for i,d in enumerate(DS):
        for j,ws in enumerate(WINDS):
            s=SURV[tag].get((pat,d,ws),[0,0])
            if s[1]: M[i,j]=s[0]/s[1]
    return M
def grid_det(tag,pat,idx):
    M=np.full((len(DS),len(WINDS)),np.nan)
    for i,d in enumerate(DS):
        for j,ws in enumerate(WINDS):
            v=DET[tag].get((pat,ws,d),[])
            if v: M[i,j]=np.median([x[idx] for x in v])
    return M

def facet(gridfn,cmap,vmin,vmax,fmt,title,fname,cbar_label):
    fig,axes=plt.subplots(2,4,figsize=(21,9))
    for r,(tag,albl) in enumerate(AXES):
        for c,pat in enumerate(PATS):
            ax=axes[r,c]; M=gridfn(tag,pat)
            im=ax.imshow(M,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,origin='lower')
            ax.set_xticks(range(len(WINDS))); ax.set_xticklabels(WINDS,fontsize=8)
            ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS,fontsize=8)
            if r==1: ax.set_xlabel('바람 ws',fontsize=9)
            if c==0: ax.set_ylabel(f'{albl}\n\nδ',fontsize=10)
            ax.set_title(pat,fontsize=12,pad=4)
            for i in range(len(DS)):
                for j in range(len(WINDS)):
                    if np.isfinite(M[i,j]):
                        val=M[i,j]; txt=fmt(val)
                        lum=0.4 if (cmap=='viridis' and val<1.5) else 0.5
                        col='white' if (cmap=='viridis' and val<vmax*0.55) else 'black'
                        ax.text(j,i,txt,ha='center',va='center',fontsize=6.5,color=col)
    fig.suptitle(title,fontsize=15,y=1.0)
    cax=fig.add_axes([1.005,0.15,0.012,0.7]); fig.colorbar(im,cax=cax,label=cbar_label)
    fig.tight_layout(); fig.savefig(fname,dpi=108,bbox_inches='tight'); plt.close(fig)
    print(f'[plot] {fname}')

# 1. 생존
facet(grid_surv,'RdYlGn',0,1,lambda v:f'{v:.1f}',
      '생존율 — 궤적(열)×축모드(행)×δ×바람 (지속20초 track, ws0=공격만)',
      'facet_survival.png','track 생존율')
# 2. 궤적이탈 gt_err
facet(lambda t,p:grid_det(t,p,2),'magma',0,8,lambda v:f'{v:.1f}',
      '궤적이탈 gt_err(m) — 궤적×축모드×δ×바람 (공격구간 중앙, 평시 0.68m)',
      'facet_gterr.png','gt_err (m)')
# 3. gyro NIS
facet(lambda t,p:grid_det(t,p,0),'viridis',0,3,lambda v:f'{v:.1f}',
      'gyro 공격 NIS — 궤적×축모드×δ×바람 (압축[0,3], 3.0=포화=easy)',
      'facet_nis_gyro.png','gyro NIS')
# 4. vel NIS
facet(lambda t,p:grid_det(t,p,1),'viridis',0,3,lambda v:f'{v:.1f}',
      'vel 공격 NIS — 궤적×축모드×δ×바람 (압축[0,3], 바람과 겹침=POMDP)',
      'facet_nis_vel.png','vel NIS')
