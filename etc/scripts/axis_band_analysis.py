"""밴드 통합분석 (단일축 vs 동시축): track생존 + hover생존 + 밴드 + NIS + 시계열 plot.
밴드 = track추락(<0.5) ∧ hover생존(>0.5). hover셀=원점호버(패턴무관), track셀=패턴별.
사용: python axis_band_analysis.py   (results_axis_single + results_axis_balanced 둘 다)"""
import os, csv, sys
import numpy as np
PFX=sys.argv[1] if len(sys.argv)>1 else 'axis'   # results_{PFX}_single/balanced, {PFX}_*.png
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
compv=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)   # 벡터
comp=lambda x: float(min(np.log1p(np.sqrt(max(x,0.0))),3.0))
WINDS=[0,4,5,6,7,8,9,10,11,12]; DS=[0.3,0.4,0.5,0.6,0.7,0.8]
PATS=['circle','figure8','waypoint','aggressive']
DIV={'single':4.36,'balanced':3.0829}   # bias→δ (축모드별)
WS_S,WS_E,AT_S,AT_E=100,300,180,380      # 시간창 (clean<100, 바람100-300, 공격180-380)
SURV_STEP=AT_E-20                        # 공격끝-2s 도달 = 생존

def load(tag):
    D=f'results_{PFX}_{tag}'
    if not os.path.exists(f'{D}/sweep_detail.csv'): return None
    div=DIV[tag]
    # 에피소드별 (policy, pattern, ws, δ) → [(step,g,v)]
    eps=defaultdict(list)
    for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
        try:
            ws=int(round(float(r['wind_speed']))); d=round(float(r['bias'])/div,1)
            if d<0.1: continue
            key=(r['policy'], r.get('pattern','-'), ws, d, r['episode']+r['cell_idx'])
            eps[key].append((int(r['step']),float(r['nis_g_raw']),float(r['nis_v_raw'])))
        except: pass
    trk=defaultdict(lambda:[0,0]); hov=defaultdict(lambda:[0,0])   # (pat/-,ws,d)->[surv,tot] / (ws,d)
    atg=defaultdict(list); atv=defaultdict(list)                   # (ws,d)->공격구간 압축NIS
    ph=defaultdict(lambda: defaultdict(list))                      # (pat,ws,d)->{phase g/v}
    for (pol,pat,ws,d,ek),rows in eps.items():
        rows.sort(); a=np.array(rows); mx=int(a[-1,0]); sv=1 if mx>=SURV_STEP else 0
        if pol=='track':
            trk[(pat,ws,d)][0]+=sv; trk[(pat,ws,d)][1]+=1
            for st,g,v in a:
                st=int(st)
                p='cl' if 10<=st<WS_S else 'wi' if WS_S<=st<AT_S else 'ov' if AT_S<=st<WS_E else 'at' if WS_E<=st<AT_E else None
                if p: ph[(pat,ws,d)][p+'g'].append(comp(g)); ph[(pat,ws,d)][p+'v'].append(comp(v))
                if WS_E<=st<AT_E: atg[(ws,d)].append(comp(g)); atv[(ws,d)].append(comp(v))
        elif pol=='hover':
            hov[(ws,d)][0]+=sv; hov[(ws,d)][1]+=1
    return dict(trk=trk,hov=hov,atg=atg,atv=atv,ph=ph,eps=eps)

def trk_all(trk,ws,d):  # 전패턴 평균 track 생존
    tot=[0,0]
    for p in PATS:
        s=trk.get((p,ws,d),[0,0]); tot[0]+=s[0]; tot[1]+=s[1]
    return tot[0]/tot[1] if tot[1] else None

L=[]; P=lambda s='':(L.append(str(s)),print(s))
S=load('single'); B=load('balanced')

for tag,dat in [('단일축(roll)',S),('동시축(roll=pitch,45°)',B)]:
    P('='*94); P(f' 【{tag}】'); P('='*94)
    if dat is None: P(' 아직 데이터 없음\n'); continue
    trk,hov,atg,atv=dat['trk'],dat['hov'],dat['atg'],dat['atv']
    P('\n track 생존 (전패턴 평균, 공격받으며 추적):')
    P('  δ＼ws | '+' '.join(f'{w:>4d}' for w in WINDS))
    for d in DS:
        P(f'  {d:.1f}   | '+' '.join((lambda v:f'{v:.1f}' if v is not None else ' - ')(trk_all(trk,w,d)).rjust(4) for w in WINDS))
    P('\n hover 생존 (원점호버=대응):')
    P('  δ＼ws | '+' '.join(f'{w:>4d}' for w in WINDS))
    for d in DS:
        P(f'  {d:.1f}   | '+' '.join((lambda s:f'{s[0]/s[1]:.1f}' if s[1] else ' - ')(hov.get((w,d),[0,0])).rjust(4) for w in WINDS))
    P('\n ★밴드 (track추락∧hover생존):')
    P('  δ＼ws | '+' '.join(f'{w:>4d}' for w in WINDS))
    for d in DS:
        row=[]
        for w in WINDS:
            t=trk_all(trk,w,d); h=hov.get((w,d),[0,0])
            if t is None or not h[1]: row.append('-'); continue
            hs=h[0]/h[1]
            row.append('밴드' if (t<0.5 and hs>0.5) else ('생존' if t>=0.5 else '추락'))
        P(f'  {d:.1f}   | '+' '.join(x.rjust(4) for x in row))
    P('\n 공격구간 NIS (압축 중앙값, gyro/vel):')
    P('  δ＼ws | '+' '.join(f'{w:>9d}' for w in WINDS))
    for d in DS:
        cells=[]
        for w in WINDS:
            g=atg.get((w,d),[]); v=atv.get((w,d),[])
            cells.append(f'{np.median(g):.1f}/{np.median(v):.1f}' if g else '-')
        P(f'  {d:.1f}   | '+' '.join(c.rjust(9) for c in cells))
    P()

# 단일축 vs 동시축 밴드 비교
if S and B:
    P('='*94); P(' 【단일축 vs 동시축 밴드차】 (동시축생존 − 단일축생존, track 전패턴)'); P('='*94)
    P('  δ＼ws | '+' '.join(f'{w:>5d}' for w in WINDS))
    for d in DS:
        row=[]
        for w in WINDS:
            ts=trk_all(S['trk'],w,d); tb=trk_all(B['trk'],w,d)
            row.append(f'{tb-ts:+.1f}' if (ts is not None and tb is not None) else '  -  ')
        P(f'  {d:.1f}   | '+' '.join(x.rjust(5) for x in row))
    P(' → ±0.1 내면 방향무관(roll≈pitch). 크면 축의존.')

# ── plot: 밴드 히트맵 (단일/동시 × track/hover) ──
def bandgrid(dat,kind):
    M=np.full((len(DS),len(WINDS)),np.nan)
    for i,d in enumerate(DS):
        for j,w in enumerate(WINDS):
            if kind=='track':
                v=trk_all(dat['trk'],w,d)
                if v is not None: M[i,j]=v
            elif kind=='hover':
                s=dat['hov'].get((w,d),[0,0])
                if s[1]: M[i,j]=s[0]/s[1]
            elif kind=='band':
                t=trk_all(dat['trk'],w,d); h=dat['hov'].get((w,d),[0,0])
                if t is not None and h[1]: M[i,j]=1.0 if (t<0.5 and h[0]/h[1]>0.5) else 0.0
    return M
panels=[('단일축 track',S,'track'),('단일축 hover',S,'hover'),('단일축 ★밴드',S,'band'),
        ('동시축 track',B,'track'),('동시축 hover',B,'hover'),('동시축 ★밴드',B,'band')]
fig,axes=plt.subplots(2,3,figsize=(17,8))
for ax,(ttl,dat,kind) in zip(axes.flat,panels):
    if dat is None: ax.text(0.5,0.5,'데이터없음',ha='center',transform=ax.transAxes); ax.set_title(ttl); continue
    M=bandgrid(dat,kind); cmap='RdYlGn' if kind!='band' else 'Blues'
    ax.imshow(M,aspect='auto',cmap=cmap,vmin=0,vmax=1,origin='lower')
    ax.set_xticks(range(len(WINDS))); ax.set_xticklabels(WINDS); ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS)
    ax.set_xlabel('ws'); ax.set_ylabel('δ'); ax.set_title(ttl,fontsize=12)
    for i in range(len(DS)):
        for j in range(len(WINDS)):
            if np.isfinite(M[i,j]):
                lab='밴' if (kind=='band' and M[i,j]>0.5) else (f'{M[i,j]:.1f}' if kind!='band' else '')
                ax.text(j,i,lab,ha='center',va='center',fontsize=8)
fig.suptitle('밴드 분석: track추락 ∧ hover생존 (arm0.02, Q2e-3, 지속20초)',fontsize=14,y=1.01); fig.tight_layout()
fig.savefig(f'{PFX}_band_heatmap.png',dpi=110,bbox_inches='tight'); plt.close(fig)

# ── plot: NIS 히트맵 (단일/동시 × gyro/vel, 공격구간) ──
fig,axes=plt.subplots(2,2,figsize=(13,9))
for row,(tag,dat) in enumerate([('단일축',S),('동시축',B)]):
    for col,(ch,key) in enumerate([('gyro','atg'),('vel','atv')]):
        ax=axes[row,col]
        if dat is None: ax.text(0.5,0.5,'데이터없음',ha='center',transform=ax.transAxes); continue
        M=np.full((len(DS),len(WINDS)),np.nan)
        for i,d in enumerate(DS):
            for j,w in enumerate(WINDS):
                vv=dat[key].get((w,d),[])
                if vv: M[i,j]=np.median(vv)
        ax.imshow(M,aspect='auto',cmap='viridis',vmin=0,vmax=3,origin='lower')
        ax.set_xticks(range(len(WINDS))); ax.set_xticklabels(WINDS); ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS)
        ax.set_xlabel('ws'); ax.set_ylabel('δ'); ax.set_title(f'{tag} 공격 {ch} NIS',fontsize=12)
        for i in range(len(DS)):
            for j in range(len(WINDS)):
                if np.isfinite(M[i,j]): ax.text(j,i,f'{M[i,j]:.1f}',ha='center',va='center',fontsize=7,color='w')
fig.suptitle('공격구간 압축NIS (min(log(1+√NIS),3)) — gyro=쉬운앵커, vel=POMDP',fontsize=13,y=1.01); fig.tight_layout()
fig.savefig(f'{PFX}_nis_heatmap.png',dpi=110,bbox_inches='tight'); plt.close(fig)

# ── plot: 시계열 (단일축, 패턴별 대표 ws8 δ0.6) ──
def rep(dat,pat,ws,d):
    best=None;bl=0
    for (pol,p,w,dd,ek),rows in dat['eps'].items():
        if pol=='track' and p==pat and w==ws and abs(dd-d)<0.01 and len(rows)>bl: bl=len(rows);best=rows
    return np.array(sorted(best)) if best else None
if S:
    fig,axes=plt.subplots(len(PATS),1,figsize=(14,3.0*len(PATS)))
    for ax,pat in zip(axes,PATS):
        a=rep(S,pat,8,0.6)
        if a is None: a=rep(S,pat,6,0.6)
        if a is None or len(a)<20: ax.text(0.5,0.5,f'{pat}:데이터없음',ha='center',transform=ax.transAxes); continue
        st=a[:,0]; g=compv(a[:,1]); v=compv(a[:,2])
        ax.axvspan(WS_S,WS_E,color='skyblue',alpha=0.22,label='바람'); ax.axvspan(AT_S,AT_E,color='red',alpha=0.13,label='공격')
        ax.plot(st,g,color='#C44E52',lw=1.5,label='gyro NIS'); ax.plot(st,v,color='#4C72B0',lw=1.4,label='vel NIS')
        ax.set_title(f'{pat} (ws8, δ0.6): clean→바람→겹침→공격→복귀',fontsize=11); ax.set_ylabel('NIS'); ax.set_ylim(0,3.15); ax.legend(loc='upper left',fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle('시간창 시계열 (단일축, arm0.02, Q2e-3)',fontsize=13,y=1.005); fig.tight_layout()
    fig.savefig(f'{PFX}_timeseries.png',dpi=110,bbox_inches='tight'); plt.close(fig)

open(f'results_{PFX}_band_summary.txt','w').write("\n".join(str(x) for x in L))
P(f'\n[plot] {PFX}_band_heatmap.png / {PFX}_nis_heatmap.png / {PFX}_timeseries.png')
P(f'[저장] results_{PFX}_band_summary.txt')
