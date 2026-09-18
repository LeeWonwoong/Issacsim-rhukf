"""time-step NIS 시계열 전 인자 분해: 궤적 × 축 × δ × 바람.
각 셀의 전 track 에피소드 스텝별 평균 압축NIS(gyro/vel). 바람창(하늘)·공격창(빨강) 음영.
Fig1 궤적(4행)×바람ws{0,8,12}(3열) [단일축 δ0.6]
Fig2 궤적(4행)×δ{0.4,0.6,0.8}(3열) [단일축 ws8]
Fig3 축모드(단일/동시)×δ{0.4,0.6,0.8} [aggressive ws8]
Fig4 δ 오버레이 (aggressive 단일 ws8, δ0.4~0.8 한 패널 gyro/vel)
사용: python facet_timeseries.py"""
import os, csv
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
compv=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)
DIV={'single':4.36,'balanced':3.0829}
PATS=['circle','figure8','waypoint','aggressive']
WS_S,WS_E,AT_S,AT_E=100,300,180,380
GC,VC='#C44E52','#4C72B0'

# 로드: (tag,pat,ws,d) -> {step: [g_raw...], [v_raw...]}
def load(tag):
    fn=f'results_axis_{tag}/sweep_detail.csv'
    data=defaultdict(lambda: defaultdict(lambda:[[],[]]))
    if not os.path.exists(fn): return data
    for r in csv.DictReader(open(fn)):
        if r['policy']!='track': continue
        try:
            d=round(float(r['bias'])/DIV[tag],1); ws=int(round(float(r['wind_speed']))); st=int(r['step'])
            if d<0.1: continue
            cell=data[(tag,r['pattern'],ws,d)]
            cell[st][0].append(float(r['nis_g_raw'])); cell[st][1].append(float(r['nis_v_raw']))
        except: pass
    return data
DATA={}
for t in ['single','balanced']: DATA.update(load(t))

def series(tag,pat,ws,d):
    cell=DATA.get((tag,pat,ws,d))
    if not cell: return None
    steps=sorted(cell.keys())
    g=[np.mean(cell[s][0]) for s in steps]; v=[np.mean(cell[s][1]) for s in steps]
    n=[len(cell[s][0]) for s in steps]
    return np.array(steps),compv(np.array(g)),compv(np.array(v)),np.array(n)

def shade(ax):
    ax.axvspan(WS_S,WS_E,color='skyblue',alpha=0.18); ax.axvspan(AT_S,AT_E,color='red',alpha=0.11)
def panel(ax,tag,pat,ws,d,title):
    s=series(tag,pat,ws,d)
    shade(ax)
    if s is not None:
        st,g,v,n=s
        ax.plot(st,g,color=GC,lw=1.5); ax.plot(st,v,color=VC,lw=1.4)
        # 생존 표시: 마지막 스텝 도달률
        surv=n[-1]/max(n.max(),1) if len(n) else 0
        ax.text(0.98,0.06,f'δ{d} ws{ws}',ha='right',va='bottom',transform=ax.transAxes,fontsize=7,color='#888')
    ax.set_ylim(0,3.15); ax.set_xlim(0,450); ax.grid(alpha=0.2)
    ax.set_title(title,fontsize=9,pad=3)
    ax.tick_params(labelsize=7)

def legend_fig(fig):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([0],[0],color=GC,lw=2,label='gyro NIS'),Line2D([0],[0],color=VC,lw=2,label='vel NIS'),
                        Patch(facecolor='skyblue',alpha=0.3,label='바람창'),Patch(facecolor='red',alpha=0.2,label='공격창')],
               loc='upper center',ncol=4,fontsize=9,bbox_to_anchor=(0.5,1.005))

# ── Fig1: 궤적(4행) × 바람(3열), 단일축 δ0.6 ──
WSS=[0,8,12]
fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,sharey=True)
for r,pat in enumerate(PATS):
    for c,ws in enumerate(WSS):
        panel(axes[r,c],'single',pat,ws,0.6,f'{pat} · ws{ws}')
        if c==0: axes[r,c].set_ylabel(f'{pat}\nNIS',fontsize=9)
        if r==3: axes[r,c].set_xlabel('step (10Hz)',fontsize=9)
legend_fig(fig)
fig.suptitle('time-step NIS: 궤적 × 바람 (단일축 δ0.6) — clean→바람(하늘)→겹침→공격(빨강)→복귀',fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig('facet_ts_pattern_wind.png',dpi=108,bbox_inches='tight'); plt.close(fig)
print('[plot] facet_ts_pattern_wind.png')

# ── Fig2: 궤적(4행) × δ(3열), 단일축 ws8 ──
DDS=[0.4,0.6,0.8]
fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,sharey=True)
for r,pat in enumerate(PATS):
    for c,d in enumerate(DDS):
        panel(axes[r,c],'single',pat,8,d,f'{pat} · δ{d}')
        if c==0: axes[r,c].set_ylabel(f'{pat}\nNIS',fontsize=9)
        if r==3: axes[r,c].set_xlabel('step (10Hz)',fontsize=9)
legend_fig(fig)
fig.suptitle('time-step NIS: 궤적 × 공격세기 δ (단일축 ws8) — δ↑에도 gyro는 3.0 포화',fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig('facet_ts_pattern_delta.png',dpi=108,bbox_inches='tight'); plt.close(fig)
print('[plot] facet_ts_pattern_delta.png')

# ── Fig3: 축모드(2행) × δ(3열), aggressive ws8 ──
fig,axes=plt.subplots(2,3,figsize=(15,7),sharex=True,sharey=True)
for r,(tag,albl) in enumerate([('single','단일축'),('balanced','동시축')]):
    for c,d in enumerate(DDS):
        panel(axes[r,c],tag,'aggressive',8,d,f'{albl} · δ{d}')
        if c==0: axes[r,c].set_ylabel(f'{albl}\nNIS',fontsize=9)
        if r==1: axes[r,c].set_xlabel('step (10Hz)',fontsize=9)
legend_fig(fig)
fig.suptitle('time-step NIS: 축모드 × δ (aggressive ws8) — 단일축 vs 동시축',fontsize=13,y=1.03)
fig.tight_layout(); fig.savefig('facet_ts_axis_delta.png',dpi=108,bbox_inches='tight'); plt.close(fig)
print('[plot] facet_ts_axis_delta.png')

# ── Fig4: δ 오버레이 (aggressive 단일 ws8) gyro/vel 각 패널 ──
fig,axes=plt.subplots(1,2,figsize=(15,4.5),sharex=True)
cmap=plt.cm.plasma
for ci,ch in enumerate(['gyro','vel']):
    ax=axes[ci]; shade(ax)
    for d in [0.3,0.4,0.5,0.6,0.7,0.8]:
        s=series('single','aggressive',8,d)
        if s is None: continue
        st,g,v,n=s
        ax.plot(st,g if ch=='gyro' else v,color=cmap(0.1+0.75*(d-0.3)/0.5),lw=1.3,label=f'δ{d}')
    ax.set_ylim(0,3.15); ax.set_xlim(0,450); ax.grid(alpha=0.2)
    ax.set_title(f'{ch} NIS — δ 오버레이',fontsize=11); ax.set_xlabel('step (10Hz)'); ax.set_ylabel('NIS')
    ax.legend(fontsize=8,ncol=2,loc='upper left')
fig.suptitle('time-step NIS: δ0.3→0.8 오버레이 (aggressive 단일축 ws8) — gyro 즉시포화 / vel 점증',fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig('facet_ts_delta_overlay.png',dpi=110,bbox_inches='tight'); plt.close(fig)
print('[plot] facet_ts_delta_overlay.png')
