"""QR sweep + 약공격 종합분석.
표1 Q_gyro sweep(1e-3~1e-2): ①공격NIS ②오프셋감쇠 ③평시/aliasing바닥 ④d′
표2 Q_vel sweep(1e-3/5e-3/2e-2): 감쇠 vs 공격NIS
표3 약공격 landing(δ0.05~0.8): gyro/vel NIS + 결과성(생존)
plot: 약공격 시계열(δ×ws) / δ오버레이 / Q_gyro효과 / Q_vel감쇠
사용: python qr_weak_analysis.py"""
import os, csv, warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: float(min(np.log1p(np.sqrt(max(x,0.0))),3.0))
compv=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)
DIV=4.36  # 단일축 bias→δ
CL,WI,AT_S,AT_E,EP=(10,100),(100,180),180,380,450
QG={'G1_qg1e-3':1e-3,'G2_qg2e-3':2e-3,'G4_qg3e-3':3e-3,'G3_qg5e-3':5e-3,'G5_qg7e-3':7e-3,'G6_qg1e-2':1e-2}
QV={'G2_qg2e-3':5e-3,'V1_qv1e-3':1e-3,'V2_qv2e-2':2e-2}
L=[]; P=lambda s='':(L.append(str(s)),print(s))

def load(cfg):
    fn=f'results_qr_{cfg}/sweep_detail.csv'
    if not os.path.exists(fn): return None
    ep=defaultdict(list)
    for r in csv.DictReader(open(fn)):
        if r['policy']!='track': continue
        try:
            d=round(float(r['bias'])/DIV,2); ws=int(round(float(r['wind_speed']))); st=int(r['step'])
            ep[(d,ws,r['pattern'],r['episode']+r['cell_idx'])].append((st,float(r['nis_g_raw']),float(r['nis_v_raw'])))
        except: pass
    return ep
def load_sum(cfg):
    fn=f'results_qr_{cfg}/sweep_summary.csv'
    return list(csv.DictReader(open(fn))) if os.path.exists(fn) else []

def seg_stat(ep,d,ws,pat,s,e,ch):  # 구간 압축NIS 리스트
    out=[]
    for (dd,w,p,ek),rows in ep.items():
        if abs(dd-d)<0.03 and w==ws and (pat is None or p==pat):
            for st,g,v in rows:
                if s<=st<e: out.append(comp(g if ch=='g' else v))
    return np.array(out)
def dprime(a,b):
    if len(a)<2 or len(b)<2: return np.nan
    return (np.mean(a)-np.mean(b))/np.sqrt(0.5*(np.var(a)+np.var(b))+1e-9)
def decay_steps(ep,d,ws,pat,ch):  # 공격끝(380)후 절반까지 스텝
    prof=defaultdict(list)
    for (dd,w,p,ek),rows in ep.items():
        if abs(dd-d)<0.03 and w==ws and (pat is None or p==pat):
            for st,g,v in rows:
                if 360<=st<=EP: prof[st].append(comp(g if ch=='g' else v))
    if not prof: return np.nan,{}
    xs=sorted(prof); prof={k:np.median(prof[k]) for k in xs}
    atk=np.mean([prof[k] for k in xs if k<AT_E]) if any(k<AT_E for k in xs) else np.nan
    post=[prof[k] for k in xs if k>=AT_E]
    if not post: return np.nan,prof
    floor=min(post); half=floor+0.5*(atk-floor)
    for k in xs:
        if k>=AT_E and prof[k]<=half: return (k-AT_E),prof
    return np.nan,prof

# ═══ 표1: Q_gyro sweep ═══
P('='*86); P('[표1] Q_gyro sweep — 공격NIS / 오프셋감쇠 / 평시·aliasing바닥 / d′  (aggr δ0.5)'); P('='*86)
P(' Qg     |공격 g/v | 평시 g | aliasing_g(ws8) | 감쇠 g/v(스텝) | d′ g/v')
gq_rows=[]
for cfg,q in QG.items():
    ep=load(cfg)
    if ep is None: P(f' {q:.0e} | (미완)'); continue
    ag=seg_stat(ep,0.5,0,'aggressive',AT_S,370,'g'); av=seg_stat(ep,0.5,0,'aggressive',AT_S,370,'v')
    cg=seg_stat(ep,0.5,0,'aggressive',*CL,'g')   # 평시(공격전 clean)
    alg=seg_stat(ep,0.5,8,'aggressive',WI[0],WI[1],'g')  # 바람+기동 구간 = aliasing 근사
    dg,_=decay_steps(ep,0.5,0,'aggressive','g'); dv,_=decay_steps(ep,0.5,0,'aggressive','v')
    dpg=dprime(ag,cg); dpv=dprime(av,seg_stat(ep,0.5,0,'aggressive',*CL,'v'))
    med=lambda a: np.median(a) if len(a) else np.nan
    P(f' {q:.0e} | {med(ag):.2f}/{med(av):.2f} | {med(cg):.3f} | {med(alg):.3f}         | {dg:.0f}/{dv:.0f}          | {dpg:.1f}/{dpv:.1f}')
    gq_rows.append((q,med(ag),med(av),med(cg),med(alg),dg,dv,dpg,dpv))

# ═══ 표2: Q_vel sweep ═══
P('\n'+'='*86); P('[표2] Q_vel sweep — vel 공격NIS / vel 오프셋감쇠 / vel d′  (aggr δ0.5)'); P('='*86)
P(' Qv     | vel 공격NIS | vel 감쇠(스텝) | vel d′')
for cfg,q in QV.items():
    ep=load(cfg)
    if ep is None: P(f' {q:.0e} | (미완)'); continue
    av=seg_stat(ep,0.5,0,'aggressive',AT_S,370,'v'); cv=seg_stat(ep,0.5,0,'aggressive',*CL,'v')
    dv,_=decay_steps(ep,0.5,0,'aggressive','v')
    P(f' {q:.0e} | {np.median(av):.2f}       | {dv:.0f}            | {dprime(av,cv):.1f}')

# ═══ 표3: 약공격 landing + 결과성 (G2) ═══
P('\n'+'='*86); P('[표3] 약공격 landing + 결과성 (G2 baseline, 지속공격, 단일축)'); P('='*86)
ep=load('G2_qg2e-3'); sm=load_sum('G2_qg2e-3')
if ep:
    P(' δ     | gyro NIS | vel NIS | 압축 vs 바람최대(1.44) | 생존(ws0/ws8)')
    surv=defaultdict(lambda:[0,0])
    for r in sm:
        if r['policy']!='track': continue
        d=round(float(r['bias'])/DIV,2); ws=int(round(float(r['wind_speed'])))
        surv[(d,ws)][0]+=1 if r['survived']=='1' else 0; surv[(d,ws)][1]+=1
    for d in [0.05,0.1,0.15,0.2,0.3,0.5,0.8]:
        g=seg_stat(ep,d,0,None,AT_S,370,'g'); v=seg_stat(ep,d,0,None,AT_S,370,'v')
        if not len(g): P(f' {d:.2f}  | (미완)'); continue
        gm=np.median(g); ovl='겹침' if gm<1.44 else ('경계' if gm<1.9 else '분리')
        s0=surv.get((d,0),[0,0]); s8=surv.get((d,8),[0,0])
        f=lambda s:f'{s[0]/s[1]:.1f}' if s[1] else '-'
        P(f' {d:.2f}  | {gm:.2f}     | {np.median(v):.2f}    | {ovl:6s}              | {f(s0)}/{f(s8)}')

open('results_qr_summary.txt','w').write("\n".join(L))
print('\n[저장] results_qr_summary.txt')

# ═══ PLOTS ═══
# 시계열 helper
def ts(ep,d,ws,pat='aggressive'):
    prof=defaultdict(lambda:[[],[]])
    for (dd,w,p,ek),rows in ep.items():
        if abs(dd-d)<0.03 and w==ws and p==pat:
            for st,g,v in rows: prof[st][0].append(g); prof[st][1].append(v)
    if not prof: return None
    xs=sorted(prof); return np.array(xs),compv(np.array([np.mean(prof[s][0]) for s in xs])),compv(np.array([np.mean(prof[s][1]) for s in xs]))
def shade(ax): ax.axvspan(*WI,color='skyblue',alpha=0.16); ax.axvspan(AT_S,AT_E,color='red',alpha=0.11)

ep=load('G2_qg2e-3')
if ep:
    # Plot A: 약공격 시계열 (행 δ, 열 ws{0,8})
    DDS=[0.1,0.2,0.3,0.5]; WSS=[0,8]  # δ0.8 제외(추락으로 곡선 잘림); δ0.5가 이미 포화
    fig,axes=plt.subplots(len(DDS),2,figsize=(13,2.4*len(DDS)),sharex=True,sharey=True)
    for r,d in enumerate(DDS):
        for c,ws in enumerate(WSS):
            ax=axes[r,c]; shade(ax); s=ts(ep,d,ws)
            if s is not None:
                st,g,v=s; ax.plot(st,g,color='#C44E52',lw=1.4); ax.plot(st,v,color='#4C72B0',lw=1.3)
            ax.axhline(1.44,color='gray',ls=':',lw=0.8)  # 바람 최대 기준선
            ax.set_ylim(0,3.15); ax.set_xlim(0,EP); ax.grid(alpha=0.2)
            ax.set_title(f'δ{d} · ws{ws}',fontsize=10)
            if c==0: ax.set_ylabel(f'δ{d}\nNIS',fontsize=9)
            if r==len(DDS)-1: ax.set_xlabel('step (10Hz)',fontsize=9)
    from matplotlib.lines import Line2D; from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([0],[0],color='#C44E52',lw=2,label='gyro'),Line2D([0],[0],color='#4C72B0',lw=2,label='vel'),
        Line2D([0],[0],color='gray',ls=':',label='바람최대 1.44'),Patch(facecolor='red',alpha=0.2,label='공격창')],
        loc='upper center',ncol=4,fontsize=9,bbox_to_anchor=(0.5,1.003))
    fig.suptitle('약공격 time-step NIS (단일축 지속, 행=δ 열=ws) — 약할수록 바람선(1.44)에 근접',fontsize=13,y=1.015)
    fig.tight_layout(); fig.savefig('qr_weak_timeseries.png',dpi=108,bbox_inches='tight'); plt.close(fig)
    print('[plot] qr_weak_timeseries.png')

    # Plot B: δ 오버레이 (gyro/vel, 약→강)
    fig,axes=plt.subplots(1,2,figsize=(15,4.5),sharex=True); cmap=plt.cm.plasma
    for ci,ch in enumerate(['gyro','vel']):
        ax=axes[ci]; shade(ax)
        for d in [0.05,0.1,0.15,0.2,0.3,0.5]:
            s=ts(ep,d,0)
            if s is None: continue
            st,g,v=s; ax.plot(st,g if ch=='gyro' else v,color=cmap(0.1+0.8*d/0.8),lw=1.3,label=f'δ{d}')
        ax.axhline(1.44,color='gray',ls=':',lw=0.9)
        ax.set_ylim(0,3.15); ax.set_xlim(0,EP); ax.grid(alpha=0.2); ax.set_xlabel('step'); ax.set_ylabel('NIS')
        ax.set_title(f'{ch} — δ 오버레이 (ws0)',fontsize=11); ax.legend(fontsize=8,ncol=2)
    fig.suptitle('약공격 NIS 오버레이 δ0.05~0.5 (단일축 ws0) — δ0.15부터 gyro 포화·δ0.05만 바람겹침',fontsize=13,y=1.02)
    fig.tight_layout(); fig.savefig('qr_weak_overlay.png',dpi=110,bbox_inches='tight'); plt.close(fig)
    print('[plot] qr_weak_overlay.png')

# Plot C: Q_gyro 효과 (공격NIS·바닥·감쇠·d′ vs Q)
if len(gq_rows)>=3:
    gq_rows.sort()
    qs=[r[0] for r in gq_rows]
    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    axes[0].plot(qs,[r[1] for r in gq_rows],'o-',color='#C44E52',label='공격 gyro NIS')
    axes[0].plot(qs,[r[3] for r in gq_rows],'s-',color='#55A868',label='평시 바닥')
    axes[0].plot(qs,[r[4] for r in gq_rows],'^--',color='#8172B3',label='aliasing(ws8)')
    axes[0].set_xscale('log'); axes[0].set_xlabel('Q_gyro'); axes[0].set_ylabel('압축 NIS'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_title('①③ 공격/바닥/aliasing')
    axes[1].plot(qs,[r[5] for r in gq_rows],'o-',color='#C44E52',label='gyro 감쇠')
    axes[1].plot(qs,[r[6] for r in gq_rows],'s-',color='#4C72B0',label='vel 감쇠')
    axes[1].set_xscale('log'); axes[1].set_xlabel('Q_gyro'); axes[1].set_ylabel('오프셋 감쇠(스텝)'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3); axes[1].set_title('② 공격끝 감쇠')
    axes[2].plot(qs,[r[7] for r in gq_rows],'o-',color='#C44E52',label="d′ gyro")
    axes[2].plot(qs,[r[8] for r in gq_rows],'s-',color='#4C72B0',label="d′ vel")
    axes[2].set_xscale('log'); axes[2].set_xlabel('Q_gyro'); axes[2].set_ylabel("d′"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3); axes[2].set_title('④ 분리도')
    fig.suptitle('Q_gyro 효과 (aggr δ0.5) — ①공격 ②감쇠 ③바닥/aliasing ④d′ tradeoff',fontsize=13,y=1.02)
    fig.tight_layout(); fig.savefig('qr_qgyro_effect.png',dpi=110,bbox_inches='tight'); plt.close(fig)
    print('[plot] qr_qgyro_effect.png')

# Plot D: Q_vel 감쇠곡선
figok=False
fig,ax=plt.subplots(figsize=(11,4.5)); shade(ax)
for cfg,q in QV.items():
    ep=load(cfg)
    if ep is None: continue
    _,prof=decay_steps(ep,0.5,0,'aggressive','v')
    if prof:
        xs=sorted(prof); ax.plot(xs,[prof[k] for k in xs],'o-',ms=3,label=f'Q_vel={q:.0e}'); figok=True
if figok:
    ax.set_xlim(360,EP); ax.set_ylim(0,2); ax.set_xlabel('step'); ax.set_ylabel('vel 압축 NIS'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Q_vel별 vel NIS 오프셋 감쇠 (공격끝 380 이후) — 높을수록 빠른 복귀·낮은 공격NIS',fontsize=12)
    fig.tight_layout(); fig.savefig('qr_qvel_decay.png',dpi=110,bbox_inches='tight'); plt.close(fig); print('[plot] qr_qvel_decay.png')
else: plt.close(fig)
