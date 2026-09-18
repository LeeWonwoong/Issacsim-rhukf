"""시간창 grid: clean/바람/겹침/공격/복귀 시계열 plot(음영) + 영역별 NIS + 생존."""
import csv, os, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)
D='results_timewin'; PATS=['circle','figure8','waypoint','aggressive']
WS_S,WS_E,AT_S,AT_E=100,300,180,380  # 바람/공격 창(스텝)
DLAB={1.744:0.4,2.616:0.6,3.27:0.75,3.488:0.8}
eps=defaultdict(list)
for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
    try:
        eps[(r['pattern'],int(round(float(r['wind_speed']))),float(r['bias']),r['policy'],r['episode']+r['cell_idx'])].append(
            (int(r['step']),float(r['nis_g_raw']),float(r['nis_v_raw']),float(r['gt_err'])))
    except: pass

# ── 영역별 NIS 요약 (track만) ──
def regime(a,lo,hi): 
    m=(a[:,0]>=lo)&(a[:,0]<hi); return a[m]
L=[]
def P(s): L.append(s); print(s)
P("="*100); P(" 시간창 grid: 영역별 gyro/vel NIS (clean/바람만/겹침/공격만) — track"); P("="*100)
P(f"{'pat':>9s} {'ws':>3s} {'δ':>5s} | {'clean_g':>7s} {'바람_g':>6s} {'겹침_g':>6s} {'공격_g':>6s} | {'바람_v':>6s} {'겹침_v':>6s} | {'생존':>4s}")
agg=defaultdict(lambda:defaultdict(list))
for (pat,ws,bias,pol,ek),rows in eps.items():
    if pol!='track': continue
    rows.sort(); a=np.array(rows)
    if len(a)<50: continue
    cg=comp(a[:,1]); cv=comp(a[:,2])
    def rg(lo,hi,ch): 
        m=(a[:,0]>=lo)&(a[:,0]<hi); return np.median((cg if ch=='g' else cv)[m]) if m.sum()>3 else float('nan')
    clean_g=rg(10,WS_S,'g'); wind_g=rg(WS_S,AT_S,'g'); ov_g=rg(AT_S,WS_E,'g'); atk_g=rg(WS_E,AT_E,'g')
    wind_v=rg(WS_S,AT_S,'v'); ov_v=rg(AT_S,WS_E,'v')
    surv=1 if a[-1,0]>=AT_E-20 else 0  # 공격끝까지 살아있으면 생존
    d=DLAB.get(bias,bias)
    agg[(pat,ws,d)]['clean_g'].append(clean_g);agg[(pat,ws,d)]['wind_g'].append(wind_g);agg[(pat,ws,d)]['ov_g'].append(ov_g);agg[(pat,ws,d)]['atk_g'].append(atk_g)
    agg[(pat,ws,d)]['wind_v'].append(wind_v);agg[(pat,ws,d)]['ov_v'].append(ov_v);agg[(pat,ws,d)]['surv'].append(surv)
for k in sorted(agg):
    v=agg[k]; nm=lambda x:np.nanmedian(v[x])
    P(f"{k[0]:>9s} {k[1]:3d} {k[2]:5.2f} | {nm('clean_g'):7.2f} {nm('wind_g'):6.2f} {nm('ov_g'):6.2f} {nm('atk_g'):6.2f} | {nm('wind_v'):6.2f} {nm('ov_v'):6.2f} | {np.mean(v['surv']):4.2f}")
open(f'{D}_summary.txt','w').write("\n".join(L))

# ── 대표 시계열 plot (패턴별, ws8 δ0.6, 음영) ──
def rep(pat,ws,bias):
    best=None;bl=0
    for (p,w,b,pol,ek),rows in eps.items():
        if p==pat and w==ws and abs(b-bias)<0.01 and pol=='track' and len(rows)>bl: bl=len(rows);best=rows
    return np.array(sorted(best)) if best else None
fig,axes=plt.subplots(len(PATS),1,figsize=(14,3.2*len(PATS)))
for ax,pat in zip(axes,PATS):
    a=rep(pat,8,2.616)  # ws8 δ0.6
    if a is None: 
        for wtry in [10,6,12,4]:
            a=rep(pat,wtry,2.616)
            if a is not None: break
    if a is None: ax.text(0.5,0.5,f'{pat}: 데이터 없음',ha='center',transform=ax.transAxes); continue
    st=a[:,0]; g=comp(a[:,1]); v=comp(a[:,2])
    ax.axvspan(WS_S,WS_E,color='skyblue',alpha=0.25,label='바람')
    ax.axvspan(AT_S,AT_E,color='red',alpha=0.15,label='공격')
    ax.plot(st,g,color='#C44E52',lw=1.6,label='gyro NIS')
    ax.plot(st,v,color='#4C72B0',lw=1.4,label='vel NIS')
    ax.set_title(f'{pat} (ws8, δ0.6): clean→바람(하늘)→겹침→공격(빨강)→복귀',fontsize=11)
    ax.set_ylabel('NIS(압축)'); ax.set_xlabel('step'); ax.legend(loc='upper left',fontsize=8); ax.grid(alpha=0.25); ax.set_ylim(0,3.1)
fig.suptitle('시간창 시계열: 바람(하늘색)·공격(빨강) 구간별 gyro/vel NIS',fontsize=14,y=1.005)
fig.tight_layout(); fig.savefig(f'{D}_timeseries.png',dpi=110,bbox_inches='tight')
print(f"[plot] {D}_timeseries.png / {D}_summary.txt")
