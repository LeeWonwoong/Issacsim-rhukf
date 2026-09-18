"""오프라인 UKF 재구동: C-error × Q/R sweet spot.
zu_log.npz(z,u_phys,attack,action) 로드 → UKF 재구동(u를 (1±C오차)로 스케일 + Q/R override).
측정: 평시바닥 / 공격레벨 / 온셋 gap(d') / 오프셋 반감기(공격 빠질때 빠르게 빠지나).
목표: 온셋 너무 안 튀게(중간) + 오프셋 빠르게. C오차 있을때/없을때.
사용: python offline_cerr_qr.py"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
import sys; sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration

D=np.load('results_zu_capture/zu_log.npz', allow_pickle=True)
data=D['data']; dt=float(D['dt'])
# cols: 0ep 1reset 2attack 3action 4-12 z(9) 13-16 u(thrust,tx,ty,tz) 17-19 euler 20 scale 21 delay
print(f'zu_log: {len(data)}행, dt={dt:.4f}, 에피 {len(np.unique(data[:,0]))}개, 공격스텝 {int(data[:,2].sum())}')

def replay(mass_err=0.0, I_err=0.0, tq_bias=0.0, c_thr=0.0, Qv=5e-3, Qg=2e-3, Qp=1e-3, Rv=0.1, Rg=0.2, Rp=0.5):
    # mass_err: UKF 질량 오차(호버 vel바닥↑). tq_bias: 토크 바이어스 N·m(COM오프셋→gyro바닥↑). c_thr: C_thrust 곱셈오차.
    calib=load_calibration('calibration.json')
    ukf=DynamicsUKF(dt=dt, calib=calib)
    ukf.m *= (1+mass_err); ukf.I=[x*(1+I_err) for x in ukf.I]
    for i in (6,7,8): ukf.Q[i,i]=Qv
    for i in (9,10,11): ukf.Q[i,i]=Qg
    for i in (0,1,2): ukf.Q[i,i]=Qp
    for i in (3,4,5): ukf.R[i,i]=Rv
    for i in (6,7,8): ukf.R[i,i]=Rg
    for i in (0,1,2): ukf.R[i,i]=Rp
    out=[]; sie=0
    for r in data:
        if r[1]==1: ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0
        z=r[4:13].astype(float); u=r[13:17].astype(float).copy()
        u[0]*=(1+c_thr); u[1]+=tq_bias; u[2]+=tq_bias   # 추력 곱셈오차 + 토크 가산바이어스
        fresh=(sie%5==0)
        res,Pzz=ukf.step(z, u, gps_fresh=fresh)
        gr,gs=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0)
        vr,vs=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0)
        # vel은 fresh 스텝만 유효(stale=inflate로 NIS≈0)
        out.append((int(r[2]),int(r[3]),float(gs),float(vs) if fresh else np.nan,float(gr),float(vr) if fresh else np.nan)); sie+=1
    return np.array(out)

def metrics(a):
    atk=a[:,0]; act=a[:,1]; g=a[:,2]; v=a[:,3]
    # 평시바닥(attack=0) / 공격레벨(attack=1, track=action0 우선)
    floor_g=np.median(g[atk==0]); floor_v=np.nanmedian(v[atk==0])
    tatk=(atk==1)&(act==0)
    if tatk.sum()<5: tatk=(atk==1)
    lvl_g=np.median(g[tatk]); lvl_v=np.nanmedian(v[tatk])
    # d' (공격 vs 평시)
    def dp(x,m):
        aa=x[tatk]; bb=x[atk==0]; aa=aa[~np.isnan(aa)]; bb=bb[~np.isnan(bb)]
        return (np.mean(aa)-np.mean(bb))/np.sqrt(0.5*(np.var(aa)+np.var(bb))+1e-9) if len(aa)>1 and len(bb)>1 else np.nan
    dpg=dp(g,None); dpv=dp(v,None)
    # 오프셋 반감기: 공격 1→0 전환 후 gyro가 (레벨→바닥) 절반까지 몇 스텝
    tr=np.where((atk[:-1]==1)&(atk[1:]==0))[0]
    halfs=[]
    for t in tr:
        half=floor_g+0.5*(lvl_g-floor_g)
        for k in range(1,12):
            if t+k>=len(g): break
            if g[t+k]<=half: halfs.append(k); break
    off=np.median(halfs) if halfs else np.nan
    return dict(fg=floor_g,fv=floor_v,lg=lvl_g,lv=lvl_v,dpg=dpg,dpv=dpv,off=off)

L=[]; P=lambda s='':(L.append(str(s)),print(s))
# ── Sweep A: C-error (baseline Q/R) ──
P('='*88); P('[A] C-error sweep (baseline Q/R) — 바닥↑·온셋gap↓(POMDP)·오프셋 확인'); P('='*88)
P(' C오차(thr/tq) | 평시바닥 g/v | 공격레벨 g/v | 온셋gap g | d′ g/v | 오프셋반감(스텝)')
A_rows=[]
P(' (mass오차%, tq바이어스Nm) →')
for me,tb in [(0.0,0.0),(0.03,0.005),(0.05,0.01),(0.05,0.02),(0.10,0.02),(0.10,0.03)]:
    m=metrics(replay(mass_err=me, tq_bias=tb))
    gap=m['lg']-m['fg']
    P(f'  m{me*100:2.0f}% tq{tb:.3f}  | {m["fg"]:.2f}/{m["fv"]:.2f}   | {m["lg"]:.2f}/{m["lv"]:.2f}   | {gap:+.2f}     | {m["dpg"]:.1f}/{m["dpv"]:.1f} | {m["off"]:.0f}')
    A_rows.append((me,tb,m,gap))
P(' → 바닥↑·gap↓·d′↓ 가 POMDP화. 너무 크면 gap<0(구분불가). 5% 부근 sweet spot?')

# ── Sweep B: Q/R (C-error 5% 고정) — 온셋 중간 + 오프셋 빠르게 ──
P('\n'+'='*88); P('[B] Q/R sweep (C-error 5% 고정) — 목표: 온셋 중간 + 오프셋 빠름'); P('='*88)
P(' 설정               | 공격레벨 g/v | d′ g/v | 오프셋반감 | 평시바닥 g')
CFG=[('baseline Qg2e-3 Rg0.2',dict(Qg=2e-3,Rg=0.2)),
     ('Qg↑5e-3(흡수·빠름)',dict(Qg=5e-3,Rg=0.2)),
     ('Qg↑1e-2',dict(Qg=1e-2,Rg=0.2)),
     ('Rg↓0.1(잔차↑)',dict(Qg=2e-3,Rg=0.1)),
     ('Rg↑0.5(잔차↓·둔감)',dict(Qg=2e-3,Rg=0.5)),
     ('Qv↑2e-2(vel흡수)',dict(Qg=2e-3,Rg=0.2,Qv=2e-2)),
     ('Rv↓0.05(vel잔차↑)',dict(Qg=2e-3,Rg=0.2,Rv=0.05))]
for lbl,kw in CFG:
    m=metrics(replay(mass_err=0.05,tq_bias=0.01,**kw))
    P(f'  {lbl:20s}| {m["lg"]:.2f}/{m["lv"]:.2f}   | {m["dpg"]:.1f}/{m["dpv"]:.1f} | {m["off"]:.0f}스텝     | {m["fg"]:.2f}')

open('results_cerr_qr.txt','w').write("\n".join(L))
print('\n[저장] results_cerr_qr.txt')

# ── plot: C-error별 온셋/오프셋 시계열 (공격 전환 정렬 평균) ──
def aligned(a, win=15):
    atk=a[:,0]; g=a[:,2]
    on=np.where((atk[:-1]==0)&(atk[1:]==1))[0]; off=np.where((atk[:-1]==1)&(atk[1:]==0))[0]
    def stack(idx):
        M=[]
        for t in idx:
            if t-3>=0 and t+win<len(g): M.append(g[t-3:t+win])
        return np.mean(M,0) if M else None
    return stack(on), stack(off)
fig,axes=plt.subplots(1,2,figsize=(14,4.5))
for e in [0.0,0.05,0.10]:
    a=replay(mass_err=e,tq_bias=e*0.2)
    on,off=aligned(a)
    x=np.arange(-3,15)
    if on is not None: axes[0].plot(x,on,marker='o',ms=3,label=f'C{e*100:.0f}%')
    if off is not None: axes[1].plot(x,off,marker='o',ms=3,label=f'C{e*100:.0f}%')
axes[0].axvline(0,color='r',ls=':'); axes[0].set_title('공격 온셋 정렬 (gyro 압축NIS)'); axes[0].set_xlabel('공격 시작 기준 스텝'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].axvline(0,color='r',ls=':'); axes[1].set_title('공격 오프셋 정렬 (빠르게 빠지나)'); axes[1].set_xlabel('공격 종료 기준 스텝'); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle('C-error별 온셋/오프셋 (C↑ → 바닥↑·온셋 덜 튐·오프셋?)',fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig('offline_cerr_onset_offset.png',dpi=110,bbox_inches='tight'); plt.close(fig)
print('[plot] offline_cerr_onset_offset.png')
