# 확정 관측 log1p(√NIS)/4 ∈[0,1] 의 time-step 그림 (Isaac 캡처, track 고정): (1) 공격 클래스 × 풍속 (2) 기동 패턴 × 풍속(평시). 공격창 붉은 음영, 오른쪽 눈금 = 대응 원시 √NIS.
import numpy as np,glob,json,collections,math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R='/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/'
files=sorted(glob.glob(R+'capture_pool/capture/ep*.npz')+glob.glob(R+'capture_pool_b/capture/ep*.npz')+glob.glob(R+'capture_branching/capture/ep*.npz'))
obs=lambda e: np.minimum(np.log1p(np.sqrt(np.maximum(e,0))),4.0)/4.0
E=[]
for f in files:
    z=np.load(f,allow_pickle=True); cols=[str(c) for c in z['cols']]; Rw=z['rows']; C={k:Rw[:,i] for i,k in enumerate(cols)}
    if 'events' in z.files and z['events'].shape==() and str(z['events']) not in ('','[]'): ev=json.loads(str(z['events']))
    else:
        de0=C['delta'] if 'delta' in C else C['delta_eff']; on=np.flatnonzero(de0>0); ev=[dict(start=int(C['step'][on[0]]),end=int(C['step'][on[-1]])+1,cls=str(z['kind']))] if len(on) else []
    E.append(dict(pat=str(z['pattern']),ws=float(z['wind_speed']),pol=str(z['policy']),ev=ev,st=C['step'].astype(int),g=C['nis_g_raw'],v=C['nis_v_raw'],a=C['prev_action'].astype(int),d0=max([e.get('d0', float(z['d0']) if 'd0' in z.files else 0.0) for e in ev],default=0)))
T=[e for e in E if e['pol']=='track']
tiers=[(0,3,'ws<3'),(3,6,'3≤ws<6'),(6,99,'ws≥6')]
def pick(cands,key):
    # 대표 에피 = key 값이 중앙인 것 (이상치 회피); 250 스텝 이상 에피 우선
    long=[e for e in cands if len(e['st'])>=250]; cands=long or cands
    if not cands: return None
    ks=[key(e) for e in cands]; return cands[int(np.argsort(ks)[len(ks)//2])]
def draw(ax,e,title):
    x=e['st']; ax.plot(x,obs(e['g']),color='#c94a3f',lw=1.1,label='gyro'); ax.plot(x,obs(e['v']),color='#3b5fc9',lw=1.1,label='vel')
    for ev in e['ev']: ax.axvspan(ev['start'],ev['end'],color='#e04848',alpha=.13)
    ax.axhline(obs(4.4),color='gray',ls=':',lw=.9)   # 평시 gyro 90% (원시 4.4)
    ax.set_ylim(0,1.02); ax.set_xlim(x[0],x[-1]); ax.set_title(title,fontsize=10); ax.grid(alpha=.25)
    ax2=ax.twinx(); ax2.set_ylim(0,1.02); ax2.set_yticks([0.25,0.5,0.75,1.0]); ax2.set_yticklabels([f'{math.exp(4*y)-1:.0f}' for y in (0.25,0.5,0.75,1.0)],fontsize=7,color='gray'); ax2.tick_params(length=0)
# (1) 공격 클래스 × 풍속
cls=['weak_burst','weak_persist','trans_persist','strong_persist']
fig,ax=plt.subplots(len(cls),3,figsize=(15,10),sharex=False)
for i,c in enumerate(cls):
    for j,(lo,hi,lab) in enumerate(tiers):
        cands=[e for e in T if lo<=e['ws']<hi and e['ev'] and all(ev['cls']==c for ev in e['ev'])]
        if not cands: cands=[e for e in T if lo<=e['ws']<hi and any(ev['cls']==c for ev in e['ev'])]
        e=pick(cands,lambda e: obs(e['g']).max())
        if e is None: ax[i,j].set_visible(False); continue
        draw(ax[i,j],e,f"{c} · δ{e['d0']:.2f} · {e['pat']} · ws {e['ws']:.1f}")
        if j==0: ax[i,j].set_ylabel('관측 log1p(√NIS)/4')
        if i==len(cls)-1: ax[i,j].set_xlabel('step (10 Hz)')
ax[0,0].legend(loc='upper left',fontsize=8)
fig.suptitle('확정 관측의 time-step — 공격 클래스(행) × 풍속(열), track 고정 · 붉은 음영 = 공격창 · 점선 = 평시 gyro 90% · 오른쪽 눈금 = 원시 √NIS'); fig.tight_layout(); fig.savefig(R+'figs/fig_obs_timestep_attack.png',dpi=120)
# (2) 기동 패턴 × 풍속 (평시)
pats=['waypoint','circle','figure8','scurve','aggressive']
fig,ax=plt.subplots(len(pats),3,figsize=(15,11))
for i,p in enumerate(pats):
    for j,(lo,hi,lab) in enumerate(tiers):
        cands=[e for e in T if lo<=e['ws']<hi and not e['ev'] and e['pat']==p]
        e=pick(cands,lambda e: np.percentile(obs(e['g']),95))
        if e is None: ax[i,j].set_visible(False); continue
        draw(ax[i,j],e,f"{p} · 평시 · ws {e['ws']:.1f}")
        if j==0: ax[i,j].set_ylabel('관측 log1p(√NIS)/4')
        if i==len(pats)-1: ax[i,j].set_xlabel('step (10 Hz)')
ax[0,0].legend(loc='upper left',fontsize=8)
fig.suptitle('확정 관측의 time-step — 기동 패턴(행) × 풍속(열), 평시(공격 없음)·track 고정 · 점선 = 평시 gyro 90% · 오른쪽 눈금 = 원시 √NIS'); fig.tight_layout(); fig.savefig(R+'figs/fig_obs_timestep_pattern.png',dpi=120)
print('track 에피', len(T), collections.Counter(c for e in T for c in {ev['cls'] for ev in e['ev']}))
