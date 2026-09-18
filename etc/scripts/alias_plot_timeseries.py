import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'etc/scripts'); sys.path.insert(0,'.')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
from alias_core import overlap, dprime, bayes_err, PATS
R=np.load('scratchpad/A_cache.npy',allow_pickle=True).item()
D=np.load('results_zu_v2/zu_log.npz',allow_pickle=True)['data']
delay=D[:,21]; reset=D[:,1]; n=len(delay)
so=np.full(n,9999.); c=9999
for i in range(n):
    if reset[i]==1: c=9999
    if delay[i]>=0: c=0
    else: c+=1
    so[i]=c
f=R['fresh']; atk=R['atk']==1
idx=np.where(f)[0]
E=dict(ep=R['ep'][idx],g=R['gyro'][idx],v=R['vel'][idx],a=atk[idx],w=R['w'][idx],
       dl=delay[idx],so=so[idx],ws=R['ws'][idx],pat=R['pat'][idx],d=R['delta'][idx])
BEN=(~E['a'])&(E['so']>15); ST=E['a']&(E['dl']>=3); ON=E['a']&(E['dl']<3); TL=(~E['a'])&(E['so']<=15)
THR=np.percentile(E['g'][ST],10)

# ══════ Fig 1: 시계열 (time-step plot) ══════
eps=np.unique(E['ep'])
atk_eps=[e for e in eps if E['a'][E['ep']==e].sum()>20]
ben_eps=[e for e in eps if E['a'][E['ep']==e].sum()==0]
ben_eps=sorted(ben_eps,key=lambda e:-E['g'][E['ep']==e].max())      # 평시 중 gyro 최고 = 최악 aliasing 후보
sel=[(atk_eps[0],'공격 에피소드'),(atk_eps[1] if len(atk_eps)>1 else atk_eps[0],'공격 에피소드'),
     (ben_eps[0],'평시(무공격) — gyro 최대 에피'),(ben_eps[1],'평시(무공격)')]
fig,ax=plt.subplots(4,1,figsize=(15,11),sharex=False)
for k,(e,ttl) in enumerate(sel):
    m=E['ep']==e; t=np.arange(m.sum())*0.1
    A=ax[k]
    aa=E['a'][m]
    on=np.where(np.r_[aa[0],np.diff(aa.astype(int))]==1)[0]; off=np.where(np.diff(aa.astype(int))==-1)[0]+1
    if len(off)<len(on): off=np.r_[off,len(aa)-1]
    for s,ee in zip(on,off): A.axvspan(t[s],t[min(ee,len(t)-1)],color='#d62728',alpha=.13,lw=0)
    A.plot(t,E['g'][m],c='#d62728',lw=1.5,label='gyro NIS(압축)')
    A.plot(t,E['v'][m],c='#1f77b4',lw=1.3,label='vel NIS(압축)')
    A.plot(t,E['w'][m],c='#888',lw=.9,ls='--',alpha=.8,label='|ω| [rad/s]')
    A.axhline(THR,c='k',ls=':',lw=1,label=f'aliasing 판정선 {THR:.2f}')
    ws=np.median(E['ws'][m]); pat=PATS[int(np.median(E['pat'][m]))]
    dd=E['d'][m&E['a']]; dtxt=f' δ={np.median(dd):.2f}' if len(dd) else ''
    A.set_title(f'ep{int(e)} — {ttl} | {pat} | 바람 {ws:.1f} m/s{dtxt}  (붉은 음영 = 공격 ON)',fontsize=10.5)
    A.set_ylabel('압축 NIS'); A.set_ylim(0,3.2); A.grid(alpha=.25)
    if k==0: A.legend(ncol=4,fontsize=9,loc='upper right')
ax[-1].set_xlabel('시간 [s]  (10 Hz 관측 격자)')
fig.suptitle('Fig 1. 관측 시계열 — 공격 버스트 vs 평시 기동 (확정 프레임워크, results_zu_v2)',fontsize=13)
fig.tight_layout(); fig.savefig('alias_fig1_timeseries.png',dpi=110); plt.close(fig)

# ══════ Fig 2: 이벤트 정렬 프로파일 (온셋/오프셋) ══════
fig,ax=plt.subplots(1,2,figsize=(14,4.6))
def prof(key,lo,hi,cond):
    xs=np.arange(lo,hi); mg=[];qg=[];mv=[];qv=[];ns=[]
    for k in xs:
        m=cond(k)
        if m.sum()<8: mg.append(np.nan);qg.append((np.nan,np.nan));mv.append(np.nan);qv.append((np.nan,np.nan));ns.append(0); continue
        mg.append(np.median(E['g'][m])); qg.append((np.percentile(E['g'][m],25),np.percentile(E['g'][m],75)))
        mv.append(np.median(E['v'][m])); qv.append((np.percentile(E['v'][m],25),np.percentile(E['v'][m],75)))
        ns.append(m.sum())
    return xs,np.array(mg),np.array(qg),np.array(mv),np.array(qv),np.array(ns)
xs,mg,qg,mv,qv,ns=prof('on',0,25,lambda k: E['a']&(E['dl']==k))
ax[0].fill_between(xs,qg[:,0],qg[:,1],color='#d62728',alpha=.18)
ax[0].plot(xs,mg,c='#d62728',lw=2,marker='o',ms=3.5,label='gyro NIS')
ax[0].fill_between(xs,qv[:,0],qv[:,1],color='#1f77b4',alpha=.15)
ax[0].plot(xs,mv,c='#1f77b4',lw=2,marker='s',ms=3.5,label='vel NIS')
ax[0].axhline(np.median(E['g'][BEN]),c='#d62728',ls=':',lw=1.2,label='평시 gyro 중앙')
ax[0].axhline(np.median(E['v'][BEN]),c='#1f77b4',ls=':',lw=1.2,label='평시 vel 중앙')
ax[0].axvspan(-0.5,2.5,color='k',alpha=.10)
ax[0].text(1,2.9,'구분 불가 구간\n(관측에 정보 없음)',ha='center',fontsize=9)
ax[0].set_xlabel('공격 버스트 온셋 이후 경과 [RL 스텝, 10Hz]'); ax[0].set_ylabel('압축 NIS 중앙(IQR)')
ax[0].set_title('공격 ON — 관측은 3스텝(0.3s) 늦게 오른다'); ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.25); ax[0].set_ylim(0,3.2)
xs,mg,qg,mv,qv,ns=prof('off',1,45,lambda k: (~E['a'])&(E['so']==k))
ax[1].fill_between(xs,qg[:,0],qg[:,1],color='#d62728',alpha=.18)
ax[1].plot(xs,mg,c='#d62728',lw=2,marker='o',ms=3.5,label='gyro NIS')
ax[1].fill_between(xs,qv[:,0],qv[:,1],color='#1f77b4',alpha=.15)
ax[1].plot(xs,mv,c='#1f77b4',lw=2,marker='s',ms=3.5,label='vel NIS')
ax[1].axhline(np.median(E['g'][BEN]),c='#d62728',ls=':',lw=1.2); ax[1].axhline(np.median(E['v'][BEN]),c='#1f77b4',ls=':',lw=1.2)
ax[1].axvspan(0,20,color='k',alpha=.08); ax[1].text(10,2.9,'공격 끝났는데\n관측은 아직 높다',ha='center',fontsize=9)
ax[1].set_xlabel('공격 버스트 종료 이후 경과 [RL 스텝, 10Hz]')
ax[1].set_title('공격 OFF — gyro 는 2-5스텝, vel 은 20-40스텝 끌린다'); ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.25); ax[1].set_ylim(0,3.2)
fig.suptitle('Fig 2. 이벤트 정렬 프로파일 — aliasing 의 본체는 진폭이 아니라 "타이밍"',fontsize=13)
fig.tight_layout(); fig.savefig('alias_fig2_event_profile.png',dpi=110); plt.close(fig)
print('fig1,fig2 saved')
