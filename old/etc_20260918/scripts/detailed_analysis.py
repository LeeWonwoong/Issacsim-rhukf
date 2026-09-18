"""종합 상세분석: 지속 vs burst, 단일축 vs 동시축.
표: 생존+crash_reason / 패턴별생존 / gt_err궤적이탈 / 자세excursion / NIS구간별
plot: 패턴별생존히트맵 / gt_err궤적이탈 / 지속vs burst생존 / 펄스시그니처
사용: python detailed_analysis.py"""
import os, csv
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)
DIV={'single':4.36,'balanced':3.0829}
PATS=['circle','figure8','waypoint','aggressive']
WS_S,WS_E,AT_S,AT_E=100,300,180,380
L=[]; P=lambda s='':(L.append(str(s)),print(s))

def d_of(pfx,tag,bias): return round(float(bias)/DIV[tag],1)

# ── summary 로드 (생존/crash/자세) ──
def load_sum(pfx,tag):
    fn=f'results_{pfx}_{tag}/sweep_summary.csv'
    if not os.path.exists(fn): return []
    return list(csv.DictReader(open(fn)))
# ── detail 로드 (NIS/gt_err/궤적) ──
def load_det(pfx,tag):
    fn=f'results_{pfx}_{tag}/sweep_detail.csv'
    if not os.path.exists(fn): return {}
    ep=defaultdict(list)
    for r in csv.DictReader(open(fn)):
        try:
            d=d_of(pfx,tag,r['bias']); ws=int(round(float(r['wind_speed'])))
            if d<0.1: continue
            ep[(r['policy'],r['pattern'],ws,d,r['episode']+r['cell_idx'])].append(
                (int(r['step']),float(r['nis_g_raw']),float(r['nis_v_raw']),float(r['gt_err']),
                 abs(float(r['roll'])),abs(float(r['pitch']))))
        except: pass
    return ep

# ══════════════════════════════════════════════════════════════
P('#'*78); P('# 종합 상세분석 — 지속20초 vs burst(rand:4,10,4,8), 단일축 vs 동시축'); P('#'*78)

# ── [표1] 생존 + crash_reason (policy별) ──
P('\n'+'='*78); P('[표1] 생존율 & 사망원인 (policy별, 전 셀 집계)'); P('='*78)
P(f'{"데이터":16s} {"policy":6s} | {"생존":>5s} {"drift":>6s} {"flip":>5s} {"alt":>5s} {"기타":>5s}  (drift=궤적이탈)')
for pfx,lbl in [('axis','지속20초'),('burst','burst')]:
    for tag in ['single','balanced']:
        for pol in ['track','hover']:
            rows=[r for r in load_sum(pfx,tag) if r['policy']==pol]
            if not rows: continue
            n=len(rows); sv=sum(1 for r in rows if r['survived']=='1')
            cr=defaultdict(int)
            for r in rows:
                if r['survived']!='1': cr[r['crash_reason']]+=1
            drift=cr.get('crash_drift',0); flip=cr.get('crash_flip',0); alt=cr.get('crash_altitude',0)
            etc=n-sv-drift-flip-alt
            P(f'{lbl+"/"+tag:16s} {pol:6s} | {sv/n:5.2f} {drift:6d} {flip:5d} {alt:5d} {etc:5d}  (n={n})')

# ── [표2] 패턴별 track 생존 (지속, δ×패턴, ws0/8/12) ──
for tag in ['single','balanced']:
    P('\n'+'='*78); P(f'[표2-{tag}] 지속 패턴별 track 생존 (δ × 패턴, ws0=공격만 / ws8 / ws12)'); P('='*78)
    rows=load_sum('axis',tag)
    agg=defaultdict(lambda:[0,0])
    for r in rows:
        if r['policy']!='track': continue
        d=d_of('axis',tag,r['bias']); ws=int(round(float(r['wind_speed'])))
        agg[(r['pattern'],d,ws)][0]+=1 if r['survived']=='1' else 0; agg[(r['pattern'],d,ws)][1]+=1
    for ws in [0,8,12]:
        P(f'  --- ws{ws} ---')
        P('  δ＼pat | '+' '.join(f'{p:>10s}' for p in PATS))
        for d in [0.5,0.6,0.7,0.8]:
            cells=[]
            for p in PATS:
                s=agg.get((p,d,ws),[0,0]); cells.append(f'{s[0]/s[1]:.2f}' if s[1] else '-')
            P(f'  {d:.1f}    | '+' '.join(c.rjust(10) for c in cells))

# ── [표3] gt_err 궤적이탈 (공격구간, ws0) — 지속 vs burst, track ──
P('\n'+'='*78); P('[표3] ★궤적이탈 gt_err (공격구간 step180-380, ws0=공격만, track)'); P('='*78)
P('  평시(공격전 clean) 기준선과 비교 → 이탈량이 결과성(추락 아니어도 임무실패)')
for tag in ['single','balanced']:
    det=load_det('axis',tag); detb=load_det('burst',tag)
    P(f'\n【{tag}】  δ | 지속 중앙/90p/max | burst 중앙/90p/max  (m)')
    # clean 기준선
    base=[]
    for k,v in det.items():
        if k[0]=='track' and k[2]==0:
            for st,g,vl,ge,rr,pp in v:
                if 10<=st<WS_S: base.append(ge)
    P(f'      평시기준(clean, δ무관): 중앙 {np.median(base):.2f}m' if base else '')
    for d in [0.5,0.6,0.7,0.8]:
        def gt(dd,src):
            out=[]
            for k,v in src.items():
                if k[0]=='track' and k[2]==0 and abs(k[3]-dd)<0.01:
                    for st,g,vl,ge,rr,pp in v:
                        if AT_S<=st<AT_E: out.append(ge)
            return np.array(out) if out else None
        a=gt(d,det); b=gt(d,detb)
        af=f'{np.median(a):.2f}/{np.percentile(a,90):.2f}/{a.max():.2f}' if a is not None else '     -     '
        bf=f'{np.median(b):.2f}/{np.percentile(b,90):.2f}/{b.max():.2f}' if b is not None else '     -     '
        P(f'      {d:.1f} | {af:>18s} | {bf:>18s}')

# ── [표4] 자세 excursion (max_roll/pitch, deg) ──
P('\n'+'='*78); P('[표4] 최대 자세각 excursion (track, ws0, 공격이 얼마나 기울이나) deg'); P('='*78)
for tag in ['single','balanced']:
    rows=load_sum('axis',tag)
    P(f'\n【{tag}】 δ | max_roll | max_pitch (rad→deg)')
    agg=defaultdict(lambda:[[],[]])
    for r in rows:
        if r['policy']!='track': continue
        ws=int(round(float(r['wind_speed'])))
        if ws!=0: continue
        d=d_of('axis',tag,r['bias'])
        agg[d][0].append(abs(float(r['max_roll']))); agg[d][1].append(abs(float(r['max_pitch'])))
    for d in [0.5,0.6,0.7,0.8]:
        if d in agg:
            rr=np.degrees(np.median(agg[d][0])); pp=np.degrees(np.median(agg[d][1]))
            P(f'      {d:.1f} | {rr:6.1f}° | {pp:6.1f}°')

# ── [표5] NIS 구간별 (clean/바람/겹침/공격, δ0.6 지속 단일축) ──
P('\n'+'='*78); P('[표5] NIS 구간별 (지속 단일축 δ0.6, 압축[0,3]) — 패턴 × 구간'); P('='*78)
det=load_det('axis','single')
P('  패턴    | clean_g wind_g atk_g | clean_v wind_v atk_v  (g=gyro v=vel)')
for pat in PATS:
    ph=defaultdict(list)
    for k,v in det.items():
        if k[0]=='track' and k[1]==pat and abs(k[3]-0.6)<0.01 and k[2]==8:  # ws8 대표
            for st,g,vl,ge,rr,pp in v:
                seg='cl' if 10<=st<WS_S else 'wi' if WS_S<=st<AT_S else 'at' if AT_S<=st<AT_E else None
                if seg: ph[seg+'g'].append(comp(g)); ph[seg+'v'].append(comp(vl))
    f=lambda k:np.median(ph[k]) if ph[k] else 0
    P(f'  {pat:8s}| {f("clg"):6.2f} {f("wig"):6.2f} {f("atg"):5.2f} | {f("clv"):6.2f} {f("wiv"):6.2f} {f("atv"):5.2f}')

open('results_detailed_summary.txt','w').write("\n".join(L))
print('\n[저장] results_detailed_summary.txt')

# ══════════════════════════════════════════════════════════════
# PLOTS
# ── plot1: 패턴별 생존 히트맵 (지속 단일축 track, 4패턴) ──
WINDS=[0,4,5,6,7,8,9,10,11,12]; DS=[0.3,0.4,0.5,0.6,0.7,0.8]
rows=load_sum('axis','single'); agg=defaultdict(lambda:[0,0])
for r in rows:
    if r['policy']!='track': continue
    d=d_of('axis','single',r['bias']); ws=int(round(float(r['wind_speed'])))
    agg[(r['pattern'],d,ws)][0]+=1 if r['survived']=='1' else 0; agg[(r['pattern'],d,ws)][1]+=1
fig,axes=plt.subplots(1,4,figsize=(22,4.5))
for ax,pat in zip(axes,PATS):
    M=np.full((len(DS),len(WINDS)),np.nan)
    for i,d in enumerate(DS):
        for j,ws in enumerate(WINDS):
            s=agg.get((pat,d,ws),[0,0])
            if s[1]: M[i,j]=s[0]/s[1]
    ax.imshow(M,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1,origin='lower')
    ax.set_xticks(range(len(WINDS))); ax.set_xticklabels(WINDS); ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS)
    ax.set_xlabel('ws'); ax.set_ylabel('δ'); ax.set_title(f'{pat}',fontsize=13)
    for i in range(len(DS)):
        for j in range(len(WINDS)):
            if np.isfinite(M[i,j]): ax.text(j,i,f'{M[i,j]:.1f}',ha='center',va='center',fontsize=7)
fig.suptitle('지속20초 단일축 track 생존 — 패턴별 (δ×ws, ws0=공격만)',fontsize=14,y=1.03); fig.tight_layout()
fig.savefig('detail_survival_perpattern.png',dpi=110,bbox_inches='tight'); plt.close(fig)

# ── plot2: gt_err 궤적이탈 (δ × 지속/burst, track, ws0) ──
det=load_det('axis','single'); detb=load_det('burst','single')
def gtd(src,dd):
    out=[]
    for k,v in src.items():
        if k[0]=='track' and k[2]==0 and abs(k[3]-dd)<0.01:
            for st,g,vl,ge,rr,pp in v:
                if AT_S<=st<AT_E: out.append(ge)
    return out
base=[]
for k,v in det.items():
    if k[0]=='track' and k[2]==0:
        for st,g,vl,ge,rr,pp in v:
            if 10<=st<WS_S: base.append(ge)
b0=np.median(base)
DD=[0.5,0.6,0.7,0.8]; sus=[]; bur=[]; suse=[]; bure=[]
for d in DD:
    a=gtd(det,d); b=gtd(detb,d)
    sus.append(np.median(a) if a else 0); suse.append(np.percentile(a,90) if a else 0)
    bur.append(np.median(b) if b else 0); bure.append(np.percentile(b,90) if b else 0)
fig,ax=plt.subplots(figsize=(10,5.5)); x=np.arange(len(DD)); w=0.35
ax.bar(x-w/2,sus,w,yerr=[np.zeros(len(DD)),np.array(suse)-np.array(sus)],capsize=4,color='#C44E52',label='지속20초 (중앙/90pct)')
ax.bar(x+w/2,bur,w,yerr=[np.zeros(len(DD)),np.array(bure)-np.array(bur)],capsize=4,color='#4C72B0',label='burst (중앙/90pct)')
ax.axhline(b0,color='gray',ls='--',label=f'평시 기준 {b0:.2f}m')
ax.set_xticks(x); ax.set_xticklabels([f'δ{d}' for d in DD]); ax.set_ylabel('궤적오차 gt_err (m)')
ax.set_title('★ 궤적이탈 (공격구간, ws0=공격만, track) — 추락 아니어도 임무이탈이 결과성',fontsize=12)
ax.legend(); ax.grid(alpha=0.25,axis='y')
fig.tight_layout(); fig.savefig('detail_trajectory_deviation.png',dpi=120,bbox_inches='tight'); plt.close(fig)

# ── plot3: 지속 vs burst 생존 (single+balanced, 측정셀) ──
def survmap(pfx,tag):
    rows=load_sum(pfx,tag); m=defaultdict(lambda:[0,0])
    for r in rows:
        if r['policy']!='track': continue
        d=d_of(pfx,tag,r['bias']); ws=int(round(float(r['wind_speed'])))
        m[(ws,d)][0]+=1 if r['survived']=='1' else 0; m[(ws,d)][1]+=1
    return {k:v[0]/v[1] for k,v in m.items() if v[1]}
fig,axes=plt.subplots(1,2,figsize=(15,5))
for ax,tag in zip(axes,['single','balanced']):
    su=survmap('axis',tag); bu=survmap('burst',tag)
    DD=[0.6,0.7,0.8]; WSS=[0,8,12]; x=np.arange(len(DD)); w=0.13
    for k,ws in enumerate(WSS):
        ax.bar(x+(k-1)*2*w-w/2,[su.get((ws,d),np.nan) for d in DD],w,color=plt.cm.Reds(0.4+0.2*k),label=f'지속 ws{ws}')
        ax.bar(x+(k-1)*2*w+w/2,[bu.get((ws,d),np.nan) for d in DD],w,color=plt.cm.Blues(0.4+0.2*k),label=f'burst ws{ws}')
    ax.set_xticks(x); ax.set_xticklabels([f'δ{d}' for d in DD]); ax.set_ylim(0,1.05)
    ax.set_ylabel('track 생존'); ax.set_title(f'{"단일축" if tag=="single" else "동시축"} 지속(빨강) vs burst(파랑)',fontsize=12)
    ax.legend(fontsize=7,ncol=3); ax.grid(alpha=0.25,axis='y')
fig.suptitle('지속20초 vs burst track 생존 — burst는 OFF-gap 회복으로 생존↑',fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig('detail_sustained_vs_burst.png',dpi=120,bbox_inches='tight'); plt.close(fig)

# ── plot4: 펄스 시그니처 (burst 단일축 δ0.8, gyro/vel/gt_err + 공격창) ──
detb=load_det('burst','single')
best=None;bl=0
for k,v in detb.items():
    if k[0]=='track' and abs(k[3]-0.8)<0.01 and k[2]==0 and len(v)>bl: bl=len(v);best=v
if best:
    a=np.array(sorted(best)); st=a[:,0]
    fig,ax1=plt.subplots(figsize=(14,4.5))
    ax1.axvspan(AT_S,AT_E,color='red',alpha=0.06)
    ax1.plot(st,comp(a[:,1]),color='#C44E52',lw=1.4,label='gyro NIS')
    ax1.plot(st,comp(a[:,2]),color='#4C72B0',lw=1.3,label='vel NIS')
    ax1.set_ylabel('NIS (압축)'); ax1.set_ylim(0,3.15); ax1.set_xlabel('step (10Hz)')
    ax2=ax1.twinx(); ax2.plot(st,a[:,3],color='#55A868',lw=1.2,ls='--',label='gt_err(m)'); ax2.set_ylabel('궤적오차 gt_err (m)',color='#55A868')
    ax1.set_title('펄스 시그니처 (burst 단일축 δ0.8, ws0): ON구간 NIS 스파이크 + gt_err 드리프트↔OFF 회복',fontsize=12)
    ax1.legend(loc='upper left',fontsize=9); ax2.legend(loc='upper right',fontsize=9); ax1.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig('detail_pulse_signature.png',dpi=120,bbox_inches='tight'); plt.close(fig)

print('[plot] detail_survival_perpattern.png / detail_trajectory_deviation.png / detail_sustained_vs_burst.png / detail_pulse_signature.png')
