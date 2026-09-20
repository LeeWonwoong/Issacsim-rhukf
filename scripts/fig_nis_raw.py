# Isaac 캡처(고정 정책)의 원시 NIS 시계열: (A) 평시 — 기동 패턴 × 풍속 티어, (B) 온셋 정렬 — 공격 클래스 × 풍속 티어 (track 정책만) + 압축 비교 수치
import numpy as np,glob,json,collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R='/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/'
files=sorted(glob.glob(R+'capture_pool/capture/ep*.npz')+glob.glob(R+'capture_pool_b/capture/ep*.npz')+glob.glob(R+'capture_branching/capture/ep*.npz'))
E=[]
for f in files:
    z=np.load(f,allow_pickle=True); cols=[str(c) for c in z['cols']]; Rw=z['rows']; C={k:Rw[:,i] for i,k in enumerate(cols)}
    if 'events' in z.files and z['events'].shape==() and str(z['events']) not in ('','[]'): ev=json.loads(str(z['events']))
    else:   # 구 포맷: 사건 1개, δ>0 연속 구간을 사건으로, 클래스 = kind
        de0=C['delta'] if 'delta' in C else C['delta_eff']; on=np.flatnonzero(de0>0); ev=[]
        if len(on): ev=[dict(start=int(C['step'][on[0]]),end=int(C['step'][on[-1]])+1,cls=str(z['kind']))]
    E.append(dict(pat=str(z['pattern']),ws=float(z['wind_speed']),pol=str(z['policy']),ev=ev,st=C['step'].astype(int),g=C['nis_g_raw'],v=C['nis_v_raw'],a=C['prev_action'].astype(int),de=C['delta_eff']))
print('에피', len(E), collections.Counter(e['pat'] for e in E), collections.Counter(e['pol'].split(':')[0] for e in E))
tiers=[(0,3,'ws<3'),(3,6,'3≤ws<6'),(6,99,'ws≥6')]; pats=['waypoint','circle','figure8','scurve','aggressive']
# (A) 평시: 사건 없음 · 전 구간 track
fig,ax=plt.subplots(2,5,figsize=(19,6.4),sharex=True)
for j,p in enumerate(pats):
    for (lo,hi,lab),col in zip(tiers,('#2f62e6','#c98b2a','#d1483a')):
        es=[e for e in E if e['pat']==p and not e['ev'] and lo<=e['ws']<hi and e['a'].max()==0 and len(e['st'])>=250]
        if not es: continue
        n=min(len(e['st']) for e in es); G=np.array([e['g'][:n] for e in es]); V=np.array([e['v'][:n] for e in es])
        for i,(X,nm) in enumerate(((G,'gyro'),(V,'vel'))):
            m=np.median(X,0); ax[i,j].plot(np.arange(n),m,color=col,lw=1.2,label=f'{lab} (n={len(es)})'); ax[i,j].fill_between(np.arange(n),np.percentile(X,25,0),np.percentile(X,75,0),color=col,alpha=.15)
    ax[0,j].set_title(p); ax[1,j].set_xlabel('스텝 (10 Hz)')
    for i in range(2): ax[i,j].set_yscale('log'); ax[i,j].grid(alpha=.3,which='both')
ax[0,0].set_ylabel('원시 NIS gyro (중앙값·IQR)'); ax[1,0].set_ylabel('원시 NIS vel'); ax[0,0].legend(fontsize=8); ax[1,0].legend(fontsize=8)
fig.suptitle('평시 원시 NIS 시계열 — 기동 패턴 × 풍속 (Isaac 캡처, 사건 없음·track 정책)'); fig.tight_layout(); fig.savefig(R+'figs/fig_nis_raw_patterns.png',dpi=120)
# (B) 온셋 정렬: track 정책 에피의 사건, 클래스별, 풍속 티어별. t=0 = δ 활성 시작(관측은 +1 스텝 지연)
cls=['weak_burst','weak_persist','trans_burst','trans_persist','strong_burst','strong_persist']; ccol={'weak':'#3a9a63','trans':'#c98b2a','strong':'#d1483a'}; ls={'burst':'-','persist':'--'}
W0,W1=10,40
fig,ax=plt.subplots(2,3,figsize=(16,6.8),sharex=True)
clean_g=np.concatenate([e['g'][e['a']==0] for e in E if not e['ev'] and e['a'].max()==0]); clean_v=np.concatenate([e['v'][e['a']==0] for e in E if not e['ev'] and e['a'].max()==0])
for j,(lo,hi,lab) in enumerate(tiers):
    for i,(ch,cl) in enumerate((('g',clean_g),('v',clean_v))):
        ax[i,j].axhspan(np.percentile(cl,5),np.percentile(cl,95),color='gray',alpha=.15,label='평시 5–95%'); ax[i,j].axhline(np.median(cl),color='gray',lw=1,ls=':')
        for c in cls:
            segs=[]
            for e in E:
                if not (e['pol']=='track' and lo<=e['ws']<hi): continue
                for ev in e['ev']:
                    if ev['cls']!=c: continue
                    s0=ev['start']-int(e['st'][0]); idx=np.arange(s0-W0,s0+W1); ok=(idx>=0)&(idx<len(e[ch]))   # 행 인덱스 = step − 첫 step(워밍업 제외)
                    x=np.full(W0+W1,np.nan); x[ok]=e[ch][idx[ok]]; segs.append(x)
            if len(segs)<3: continue
            S=np.array(segs); ax[i,j].plot(np.arange(-W0,W1),np.nanmedian(S,0),color=ccol[c.split('_')[0]],ls=ls[c.split('_')[1]],lw=1.4,label=f'{c} (n={len(segs)})')
        ax[i,j].axvline(0,color='k',lw=.8); ax[i,j].set_yscale('log'); ax[i,j].grid(alpha=.3,which='both')
    ax[0,j].set_title(lab); ax[1,j].set_xlabel('온셋 기준 스텝')
ax[0,0].set_ylabel('원시 NIS gyro (사건 중앙값)'); ax[1,0].set_ylabel('원시 NIS vel'); ax[0,2].legend(fontsize=7.5,loc='upper right'); ax[1,2].legend(fontsize=7.5,loc='upper right')
fig.suptitle('공격 온셋 정렬 원시 NIS — 공격 클래스(색=세기, 실선 burst/점선 persist) × 풍속 (track 정책)'); fig.tight_layout(); fig.savefig(R+'figs/fig_nis_raw_onset.png',dpi=120)
# 압축 비교: 평시 vs 클래스별 공격 스텝(δ_eff>0, track) 의 원시 NIS 분위수와 두 변환값
import math
def q(x): return np.percentile(x,[50,90,99])
def evsteps(e,ev,ch):
    i0=np.searchsorted(e['st'],ev['start']+1); i1=np.searchsorted(e['st'],min(ev['end']+1,e['st'][-1]+1)); m=(e['a'][i0:i1]==0); return e[ch][i0:i1][m]
atk={c:np.concatenate([evsteps(e,ev,'g') for e in E if e['pol']=='track' for ev in e['ev'] if ev['cls']==c] or [np.array([])]) for c in cls}
atkv={c:np.concatenate([evsteps(e,ev,'v') for e in E if e['pol']=='track' for ev in e['ev'] if ev['cls']==c] or [np.array([])]) for c in cls}
print('vel  원시 NIS 분위수(50/90/99):', '평시', q(clean_v).round(1), {c:q(atkv[c]).round(1).tolist() for c in cls if len(atkv[c])})
print('gyro 원시 NIS 분위수(50/90/99):', '평시', q(clean_g).round(1), {c:q(atk[c]).round(1).tolist() for c in cls if len(atk[c])})
f1=lambda e: math.log1p(e); f2=lambda e: math.log1p(math.sqrt(e))
for name,f in (('log1p(ε)',f1),('log1p(√ε)',f2)):
    cm=f(np.median(clean_g)); c95=f(np.percentile(clean_g,95)); print(f'{name}: 평시 중앙 {cm:.2f} 95% {c95:.2f} | '+' '.join(f"{c.split('_')[0][0]}{c.split('_')[1][0]} {f(np.median(atk[c])):.2f}" for c in cls if len(atk[c]))+f' | ε=50→{f(50):.2f} 500→{f(500):.2f} 3000→{f(3000):.2f}')
