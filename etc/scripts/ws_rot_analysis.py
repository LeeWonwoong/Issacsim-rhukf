"""ws 스윕(arm0.04): benign gyro/vel가 ws따라 오르나 + 공격 겹침 + windowed 분리."""
import csv, os, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)
def winmax(v,w=4): return np.array([v[max(0,i-w+1):i+1].max() for i in range(len(v))])
def dpr(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return abs(a.mean()-b.mean())/np.sqrt(0.5*(a.var()+b.var())+1e-9) if a.size>5 and b.size>5 else float('nan')
D='results_ws_rot'; WINDS=[0,6,9,12]
rows=defaultdict(lambda:defaultdict(list))
for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
    try:
        ws=int(round(float(r['wind_speed']))); bias=float(r['bias']); pol=r['policy']
        rows[(ws,bias,pol)][r['episode']+r['cell_idx']].append((int(r['step']),int(r['attack_active']),float(r['nis_g_raw']),float(r['nis_v_raw'])))
    except: pass
surv=defaultdict(list)
try:
    for r in csv.DictReader(open(f'{D}/sweep_summary.csv')):
        surv[(int(round(float(r['wind_speed']))),float(r['bias']),r['policy'])].append(int(r['survived']))
except: pass
def chan(ws,bias,pol,atk_only):
    g=[];gw=[];v=[]
    for ek,rr in rows.get((ws,bias,pol),{}).items():
        rr.sort(); a=np.array(rr)
        m=(a[:,1]==1) if atk_only else np.ones(len(a),bool)
        cg=comp(a[:,2]); g+=list(cg[m]); gw+=list(winmax(cg)[m]); v+=list(comp(a[:,3])[m])
    return np.array(g),np.array(gw),np.array(v)
L=[]
def P(s): L.append(s); print(s)
P("="*92); P(" ws 스윕(arm0.04): 바람 회전이 gyro/vel를 ws따라 얼마 올리나 + 공격 겹침"); P("="*92)
P(f"{'ws':>3s} | {'benign gyro p50/p90':>19s} {'benign vel p50/p90':>18s} | {'공격gyro p50':>10s} {'공격vel':>7s} | {'d′gyro단/윈':>11s} {'d′vel':>6s} | {'ben생존':>6s}")
BG={};AG={};BV={};AV={}
for ws in WINDS:
    bg,bgw,bv=chan(ws,0.0,'track',False)
    ag,agw,av=chan(ws,2.616,'track',True)
    BG[ws]=bg;AG[ws]=ag;BV[ws]=bv;AV[ws]=av
    dg=dpr(ag,bg); dgw=dpr(agw,bgw); dv=dpr(av,bv)
    bs=np.mean(surv.get((ws,0.0,'track'),[np.nan]))
    def pp(x,q): return np.percentile(x,q) if x.size else float('nan')
    P(f"{ws:3d} | {pp(bg,50):8.2f}/{pp(bg,90):8.2f}   {pp(bv,50):7.2f}/{pp(bv,90):8.2f}  | {pp(ag,50):10.2f} {pp(av,50):7.2f} | {dg:5.1f}/{dgw:5.1f} {dv:6.2f} | {bs:6.2f}")
P("\n판정: benign gyro가 ws따라 오르면 회전 gust 작동. d′gyro 작아지면 겹침(POMDP). 여전히 크면 gyro trivial(물리).")
open(f'{D}_summary.txt','w').write("\n".join(L))
# Plot: gyro/vel vs ws (benign vs 공격)
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
bgp=[np.percentile(BG[w],90) if BG[w].size else np.nan for w in WINDS]
agp=[np.percentile(AG[w],50) if AG[w].size else np.nan for w in WINDS]
ax[0].plot(WINDS,bgp,'o-',label='benign gyro p90',color='#4C72B0',lw=2)
ax[0].plot(WINDS,agp,'s--',label='공격(δ0.6) gyro p50',color='#C44E52',lw=2)
ax[0].set_title('gyro NIS vs 바람 (회전 arm0.04)',fontsize=12);ax[0].set_xlabel('ws');ax[0].set_ylabel('gyro NIS 압축');ax[0].legend();ax[0].grid(alpha=0.3)
bvp=[np.percentile(BV[w],90) if BV[w].size else np.nan for w in WINDS]
avp=[np.percentile(AV[w],50) if AV[w].size else np.nan for w in WINDS]
ax[1].plot(WINDS,bvp,'o-',label='benign vel p90',color='#4C72B0',lw=2)
ax[1].plot(WINDS,avp,'s--',label='공격 vel p50',color='#C44E52',lw=2)
ax[1].set_title('vel NIS vs 바람',fontsize=12);ax[1].set_xlabel('ws');ax[1].set_ylabel('vel NIS 압축');ax[1].legend();ax[1].grid(alpha=0.3)
fig.suptitle('바람↑ → benign gyro/vel 얼마 오르나 vs 공격 (겹치면 POMDP)',fontsize=13,y=1.02)
fig.tight_layout();fig.savefig(f'{D}_plot.png',dpi=110,bbox_inches='tight')
print(f"[plot] {D}_plot.png")
