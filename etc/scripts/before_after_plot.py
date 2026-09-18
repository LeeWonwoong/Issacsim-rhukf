# -*- coding: utf-8 -*-
"""기동 수정(①②③) + Q/R 변경(config E) 전후 비교 — 시계열 + 분포 + 지표.
  BEFORE : results_zu_v2 (구 기동 amp0.7/freq0.25, R2.8/ω0.5) × config A (Qg5e-3 Rg0.2 Rv0.1)
  AFTER  : results_zu_v3 (신 기동 amp0.5/freq1.0,  R1.6/ω0.75) × config E (Qg2e-2 Rg2e-2 Rv0.01 Qe2e-3)
  MID    : results_zu_v3 × config A  (기동 효과만 분리)
"""
import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.'); sys.path.insert(0,'etc/scripts')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration
PATS=['circle','figure8','waypoint','aggressive']; CLIP=3.0
CFG_A=dict(q_pos=1e-3,q_eul=5e-4,q_vel=5e-3,q_gyr=5e-3,r_pos=0.5,r_vel=0.1, r_gyr=0.2)
CFG_E=dict(q_pos=1e-3,q_eul=2e-3,q_vel=5e-3,q_gyr=2e-2,r_pos=0.5,r_vel=0.01,r_gyr=0.02)
SLOT=dict(q_pos=(0,3,'Q'),q_eul=(3,6,'Q'),q_vel=(6,9,'Q'),q_gyr=(9,12,'Q'),
          r_pos=(0,3,'R'),r_vel=(3,6,'R'),r_gyr=(6,9,'R'))

def run(path,cfg,com=0.05,seed=0):
    rng=np.random.RandomState(seed)
    D=np.load(f'{path}/zu_log.npz',allow_pickle=True); data=D['data']; dt=float(D['dt'])
    calib=load_calibration('calibration.json'); ukf=DynamicsUKF(dt=dt,calib=calib)
    for k,(a,b,M) in SLOT.items():
        t=ukf.Q if M=='Q' else ukf.R
        for i in range(a,b): t[i,i]=cfg[k]
    n=len(data); g=np.empty(n); v=np.empty(n); fr=np.zeros(n,bool); cb=np.zeros(2); sie=0
    for i,r in enumerate(data):
        if r[1]==1:
            ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0
            cb=rng.uniform(-com,com,2)
        z=r[4:13].astype(float); u=r[13:17].astype(float).copy(); u[1]+=cb[0]; u[2]+=cb[1]
        fresh=(sie%5==0); fr[i]=fresh
        res,Pzz=ukf.step(z,u,gps_fresh=fresh)
        g[i]=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0,clip=CLIP)[1]
        v[i]=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0,clip=CLIP)[1]
        sie+=1
    delay=data[:,21]; reset=data[:,1]
    so=np.full(n,9999.); c=9999
    for i in range(n):
        if reset[i]==1: c=9999
        if delay[i]>=0: c=0
        else: c+=1
        so[i]=c
    z=data[:,4:13]; u=data[:,13:17]
    return dict(g=g,v=v,fresh=fr,ep=data[:,0],atk=(data[:,2]==1),delay=delay,so=so,
                w=np.linalg.norm(z[:,6:9],axis=1),vxy=np.linalg.norm(z[:,3:5],axis=1),
                utq=np.linalg.norm(u[:,1:4],axis=1),pat=data[:,22].astype(int),ws=data[:,23],
                act=data[:,3],delta=data[:,20])

def stats(R):
    f=R['fresh']; BEN=f&(~R['atk'])&(R['so']>15); ST=f&R['atk']&(R['delay']>=3); ON=f&R['atk']&(R['delay']<3)
    o={}
    for nm,x in [('gyro',R['g']),('vel',R['v'])]:
        b=x[BEN]; s=x[ST]
        def ovl(a,c,bins=60):
            e=np.linspace(0,3.05,bins+1); h0,_=np.histogram(a,e); h1,_=np.histogram(c,e)
            return float(np.minimum(h0/max(h0.sum(),1),h1/max(h1.sum(),1)).sum())
        o[nm]=dict(base=np.median(b),plat=np.median(s),sigma=b.std(),
                   snr=(np.median(s)-np.median(b))/max(b.std(),1e-9),
                   berr=ovl(b,s)/2, p99=np.percentile(b,99),
                   alias=float((b>=np.percentile(s,10)).mean())*100)
    m=f&(~R['atk'])&(R['act']==0)
    o['duty_w1']=float((R['w'][m]>1.0).mean())*100
    o['duty_w15']=float((R['w'][m]>1.5).mean())*100
    o['utq_med']=float(np.median(R['utq'][m])); o['utq_p90']=float(np.percentile(R['utq'][m],90))
    o['v_med']=float(np.median(R['vxy'][m])); o['v_p90']=float(np.percentile(R['vxy'][m],90))
    return o

if __name__=='__main__':
    B=run('results_zu_v2',CFG_A); A=run('results_zu_v3',CFG_E); M=run('results_zu_v3',CFG_A)
    sB,sA,sM=stats(B),stats(A),stats(M)
    print("="*112)
    print(f"{'':26s} {'BEFORE (구기동×A)':>18s} {'MID (신기동×A)':>16s} {'AFTER (신기동×E)':>18s}")
    print("="*112)
    rows=[('평시 |v_xy| 중앙 [m/s]','v_med',None),('평시 |v_xy| 90p','v_p90',None),
          ('평시 |u_τ| 중앙 [N·m]','utq_med',None),('평시 |u_τ| 90p','utq_p90',None),
          ('|ω|>1.0 듀티 [%]','duty_w1',None),('|ω|>1.5 듀티 [%]','duty_w15',None)]
    for lbl,k,_ in rows:
        print(f"{lbl:26s} {sB[k]:18.3f} {sM[k]:16.3f} {sA[k]:18.3f}")
    print("-"*112)
    for ch in ['gyro','vel']:
        for lbl,k in [('바닥 중앙','base'),('평시 99p','p99'),('공격 고원','plat'),
                      ('SNR (↓낮을수록 애매)','snr'),('1스텝 Bayes오류 (↑좋음)','berr'),('aliasing율 [%]','alias')]:
            print(f"{ch+' '+lbl:26s} {sB[ch][k]:18.3f} {sM[ch][k]:16.3f} {sA[ch][k]:18.3f}")
        print("-"*112)
    # ── 시계열 플롯: 공격 섞인 대표 에피 ──
    def pick(R):
        eps=np.unique(R['ep']); best=None
        for e in eps:
            m=(R['ep']==e)&R['fresh']
            if R['atk'][m].sum()>15 and np.median(R['ws'][m])>3: return e
            if R['atk'][m].sum()>15 and best is None: best=e
        return best
    fig,ax=plt.subplots(3,1,figsize=(15,10))
    for k,(R,cfgn,ttl) in enumerate([(B,'A','BEFORE — 구 기동(amp0.7·freq0.25·R2.8·ω0.5) × 구 Q/R'),
                                     (M,'A','MID — 신 기동 × 구 Q/R  (기동 효과만)'),
                                     (A,'E','AFTER — 신 기동 × 신 Q/R (config E)')]):
        e=pick(R); m=(R['ep']==e)&R['fresh']; t=np.arange(m.sum())*0.1; Ax=ax[k]
        aa=R['atk'][m]
        on=np.where(np.r_[aa[0],np.diff(aa.astype(int))]==1)[0]; off=np.where(np.diff(aa.astype(int))==-1)[0]+1
        if len(off)<len(on): off=np.r_[off,len(aa)-1]
        for s_,e_ in zip(on,off): Ax.axvspan(t[s_],t[min(e_,len(t)-1)],color='#d62728',alpha=.13,lw=0)
        Ax.plot(t,R['g'][m],c='#d62728',lw=1.6,label='gyro NIS')
        Ax.plot(t,R['v'][m],c='#1f77b4',lw=1.3,label='vel NIS')
        Ax.plot(t,R['w'][m],c='#666',lw=.9,ls='--',alpha=.85,label='|ω| [rad/s]')
        Ax.plot(t,R['vxy'][m]/2,c='#2ca02c',lw=.9,ls=':',alpha=.85,label='|v_xy|/2 [m/s]')
        st=stats(R)
        Ax.axhline(st['gyro']['base'],c='#d62728',ls=':',lw=1,alpha=.7)
        ws=np.median(R['ws'][m]); pat=PATS[int(np.median(R['pat'][m]))]
        dd=R['delta'][m&R['atk']]
        Ax.set_title(f"{ttl}   |   ep{int(e)} · {pat} · 바람 {ws:.1f} m/s"
                     + (f" · δ≈{np.median(dd):.2f}" if len(dd) else "")
                     + f"   [gyro SNR {st['gyro']['snr']:.2f} · BErr {st['gyro']['berr']:.3f}]",fontsize=10.5)
        Ax.set_ylabel('압축 NIS'); Ax.set_ylim(0,3.2); Ax.grid(alpha=.25)
        if k==0: Ax.legend(ncol=4,fontsize=9,loc='upper right')
    ax[-1].set_xlabel('시간 [s] (10 Hz 관측 격자)   ·   붉은 음영 = 공격 ON')
    fig.suptitle('기동 수정 + Q/R 변경 전후 — 정상·바람·공격이 섞인 에피소드',fontsize=13.5)
    fig.tight_layout(); fig.savefig('before_after_timeseries.png',dpi=110); plt.close(fig)
    # ── 분포 + 지표 ──
    fig,ax=plt.subplots(1,3,figsize=(16,4.6))
    bins=np.linspace(0,3.05,50)
    for i,(R,ttl) in enumerate([(B,'BEFORE'),(M,'MID'),(A,'AFTER')]):
        f=R['fresh']; BEN=f&(~R['atk'])&(R['so']>15); ST=f&R['atk']&(R['delay']>=3)
        ax[i].hist(R['g'][BEN],bins,density=True,histtype='stepfilled',alpha=.45,color='#2ca02c',label=f'평시 (n={BEN.sum()})')
        ax[i].hist(R['g'][ST],bins,density=True,histtype='stepfilled',alpha=.45,color='#d62728',label=f'공격 정상상태 (n={ST.sum()})')
        s=stats(R)['gyro']
        ax[i].set_title(f"{ttl} — gyro\nSNR {s['snr']:.2f} · Bayes오류 {s['berr']:.3f} · 겹침↑",fontsize=10.5)
        ax[i].set_xlabel('압축 gyro NIS'); ax[i].legend(fontsize=8.5); ax[i].grid(alpha=.25)
    ax[0].set_ylabel('밀도')
    fig.suptitle('평시 vs 공격 분포 — 겹침이 커질수록 POMDP 난이도↑ (RHUKF 무대)',fontsize=13)
    fig.tight_layout(); fig.savefig('before_after_dist.png',dpi=110); plt.close(fig)
    print('\nsaved before_after_timeseries.png, before_after_dist.png')
