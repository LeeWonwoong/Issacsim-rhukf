"""최종 grid 분석: 생존/gyro/vel 히트맵(δ×ws) + 시간창 시계열 + 종합표.
사용: python final_analysis.py [dir(기본 results_final)]"""
import sys, os, csv
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)
D=sys.argv[1] if len(sys.argv)>1 else 'results_final'
# B(balanced 45°): value=δ×3.0829 → δ=bias/3.0829
bias2d=lambda b: round(b/3.0829,1) if b>0.01 else 0
PATS=['circle','figure8','waypoint','aggressive']
WINDS=[0,4,5,6,7,8,9,10,11,12]; DS=[0.3,0.4,0.5,0.6,0.7,0.8]
WS_S,WS_E,AT_S,AT_E=100,300,180,380

eps=defaultdict(list)
for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
    try:
        if r['policy']!='track': continue
        ws=int(round(float(r['wind_speed']))); d=bias2d(float(r['bias']))
        if d==0: continue
        eps[(r['pattern'],ws,d,r['episode']+r['cell_idx'])].append((int(r['step']),float(r['nis_g_raw']),float(r['nis_v_raw'])))
    except: pass
# 집계
S=defaultdict(lambda: defaultdict(list))   # (pat,ws,d) -> {surv,phase NIS}
for (pat,ws,d,ek),rows in eps.items():
    rows.sort(); a=np.array(rows); mx=int(a[-1,0])
    S[(pat,ws,d)]['surv'].append(1 if mx>=AT_E-20 else 0)
    for st,g,v in a:
        st=int(st)
        ph='cl' if 10<=st<WS_S else 'wi' if WS_S<=st<AT_S else 'ov' if AT_S<=st<WS_E else 'at' if WS_E<=st<AT_E else None
        if ph: S[(pat,ws,d)][ph+'g'].append(comp(g)); S[(pat,ws,d)][ph+'v'].append(comp(v))
def grid(pat,key,fn=np.median):
    M=np.full((len(DS),len(WINDS)),np.nan)
    for i,d in enumerate(DS):
        for j,ws in enumerate(WINDS):
            v=S.get((pat,ws,d),{}).get(key,[])
            if v: M[i,j]=fn(v)
    return M

# ── 히트맵 1: 생존율 (δ×ws, 4패턴) ──
fig,axes=plt.subplots(1,4,figsize=(22,4.5))
for ax,pat in zip(axes,PATS):
    M=grid(pat,'surv',np.mean)
    im=ax.imshow(M,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1,origin='lower')
    ax.set_xticks(range(len(WINDS))); ax.set_xticklabels(WINDS); ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS)
    ax.set_xlabel('ws'); ax.set_ylabel('δ'); ax.set_title(f'{pat} 생존율',fontsize=12)
    for i in range(len(DS)):
        for j in range(len(WINDS)):
            if np.isfinite(M[i,j]): ax.text(j,i,f'{M[i,j]:.1f}',ha='center',va='center',fontsize=7)
fig.suptitle('생존율 (공격끝까지) — δ×ws, ws0=공격only',fontsize=14,y=1.03); fig.tight_layout()
fig.savefig(f'{D}_survival.png',dpi=110,bbox_inches='tight'); plt.close(fig)

# ── 히트맵 2: gyro/vel NIS (공격구간, ws따라 aliasing) ──
fig,axes=plt.subplots(2,4,figsize=(22,9))
for ax,pat in zip(axes[0],PATS):
    M=grid(pat,'atg')
    im=ax.imshow(M,aspect='auto',cmap='viridis',vmin=0,vmax=3,origin='lower')
    ax.set_xticks(range(len(WINDS))); ax.set_xticklabels(WINDS); ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS)
    ax.set_title(f'{pat} 공격 gyro',fontsize=11); ax.set_xlabel('ws'); ax.set_ylabel('δ')
for ax,pat in zip(axes[1],PATS):
    M=grid(pat,'atv')
    im=ax.imshow(M,aspect='auto',cmap='viridis',vmin=0,vmax=3,origin='lower')
    ax.set_xticks(range(len(WINDS))); ax.set_xticklabels(WINDS); ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS)
    ax.set_title(f'{pat} 공격 vel',fontsize=11); ax.set_xlabel('ws'); ax.set_ylabel('δ')
fig.suptitle('공격구간 gyro(위)/vel(아래) NIS (압축) — δ×ws',fontsize=14,y=1.01); fig.tight_layout()
fig.savefig(f'{D}_nis_heatmap.png',dpi=110,bbox_inches='tight'); plt.close(fig)

# ── 시계열 (패턴별 대표 ws8 δ0.6) ──
def rep(pat,ws,d):
    best=None;bl=0
    for (p,w,dd,ek),rows in eps.items():
        if p==pat and w==ws and abs(dd-d)<0.01 and len(rows)>bl: bl=len(rows);best=rows
    return np.array(sorted(best)) if best else None
fig,axes=plt.subplots(len(PATS),1,figsize=(14,3.2*len(PATS)))
for ax,pat in zip(axes,PATS):
    a=rep(pat,8,0.6)
    if a is None: a=rep(pat,6,0.6)
    if a is None or len(a)<20: ax.text(0.5,0.5,f'{pat}:데이터없음',ha='center',transform=ax.transAxes); continue
    st=a[:,0]; g=comp(a[:,1]); v=comp(a[:,2])
    ax.axvspan(WS_S,WS_E,color='skyblue',alpha=0.25,label='바람'); ax.axvspan(AT_S,AT_E,color='red',alpha=0.15,label='공격')
    ax.plot(st,g,color='#C44E52',lw=1.5,label='gyro NIS'); ax.plot(st,v,color='#4C72B0',lw=1.4,label='vel NIS')
    ax.set_title(f'{pat} (ws8, δ0.6): clean→바람(하늘)→겹침→공격(빨강)→복귀',fontsize=11); ax.set_ylabel('NIS'); ax.set_ylim(0,3.1); ax.legend(loc='upper left',fontsize=8); ax.grid(alpha=0.25)
fig.suptitle('시간창 시계열 (arm0.02, Q2e-3)',fontsize=14,y=1.005); fig.tight_layout()
fig.savefig(f'{D}_timeseries.png',dpi=110,bbox_inches='tight'); plt.close(fig)

# ── 종합표 ──
L=[]; P=lambda s:(L.append(s),print(s))
P('='*100); P(' 최종 grid: 공격only(ws0) + 공격+바람 — 생존 & gyro/vel NIS (arm0.02, Q2e-3)'); P('='*100)
for pat in PATS:
    P(f'\n【{pat}】 (생존율 | 공격 gyro | 공격 vel, 압축)')
    P('  δ＼ws | '+' '.join(f'{w:>5d}' for w in WINDS))
    for d in DS:
        sv=' '.join((lambda s:f'{np.mean(s):.1f}' if s else '  - ')(S.get((pat,w,d),{}).get('surv',[])) for w in WINDS)
        P(f'  {d:.1f}생존| '+' '.join(f'{x:>5s}' for x in sv.split()))
open(f'{D}_summary.txt','w').write("\n".join(str(x) for x in L))
print(f'\n[plot] {D}_survival.png / {D}_nis_heatmap.png / {D}_timeseries.png / {D}_summary.txt')
