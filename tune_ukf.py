#!/usr/bin/env python3
"""
tune_ukf.py — UKF Q/R 오프라인 튜닝 (sim 불필요, CPU 병렬)
==========================================================
online_rl_main.py --log-zu 의 zu_log.npz (z,u 시계열)로 NIS를 online과
동일하게 재계산하여, 12차원 Q / 9차원 R(채널별)을 튜닝한다.

한 번 실행하면 두 결과를 모두 출력:
  [1] SWEEP-1D : 각 파라미터를 '하나씩' 현재값 중심으로 훑어
                 (나머지는 현재값 고정) → 어느 축이 효과 있는지 한눈에.
  [2] GRID     : 핵심 축들의 곱집합 정밀탐색 + ★현재 대비 추천.

  python3 tune_ukf.py results_zu/zu_log.npz                # 전체(병렬)
  python3 tune_ukf.py results_zu/zu_log.npz --episodes 8   # 앞 8ep(빠름)
  python3 tune_ukf.py results_zu/zu_log.npz --jobs 16
  python3 tune_ukf.py results_zu/zu_log.npz --no-grid      # 1D만(빠름)

파라미터(7축): posQ, eulerQ, velQ, gyroQ, posR, velR, gyroR
NIS 재계산 = online_rl_main._rl_step_10hz 와 동일(res[3:6], res[6:9]).
"""
import argparse, itertools, sys, time, math
import numpy as np
from multiprocessing import Pool, cpu_count

try:
    from ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration
except Exception:
    try:
        from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration
    except Exception as e:
        sys.exit(f"[!] ukf_filter 임포트 실패: {e}\n    ukf_filter.py와 같은 폴더(또는 repo 루트)에서 실행하세요.")

_DATA=None; _DT=None; _CALIB=None

def _init(data, dt, calib):
    global _DATA,_DT,_CALIB
    _DATA,_DT,_CALIB = data,dt,calib

# ── 7축 파라미터 p = {posQ,eulerQ,velQ,gyroQ,posR,velR,gyroR} ──
def _build(p):
    u = DynamicsUKF(dt=_DT, calib=_CALIB, ff=1.0, q_gate=0.0)
    u.Q = np.diag([p['posQ']]*3 + [p['eulerQ']]*3 + [p['velQ']]*3 + [p['gyroQ']]*3)
    u.R = np.diag([p['posR']]*3 + [p['velR']]*3 + [p['gyroR']]*3)
    return u

def _replay(p):
    d=_DATA; rst=d[:,1]; z=d[:,4:13]; u=d[:,13:17]; eul=d[:,17:20]
    nv=np.empty(len(d)); ng=np.empty(len(d)); ukf=None
    for i in range(len(d)):
        if rst[i]>0.5 or ukf is None:
            ukf=_build(p)
            ukf.x[0:3]=z[i,0:3]; ukf.x[3:6]=eul[i]; ukf.x[6:9]=z[i,3:6]; ukf.x[9:12]=z[i,6:9]
        res,Pzz=ukf.step(z[i],u[i])
        a,_=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0,offset=0.5)  # (raw 사용이라 offset 무영향; 시그니처 일관성용)
        b,_=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0)
        nv[i]=a; ng[i]=b
    return nv,ng

def _dp(x,a):
    xn,xa=x[a==0],x[a==1]
    if len(xn)<5 or len(xa)<5: return 0.0
    return (np.nanmean(xa)-np.nanmean(xn))/np.sqrt(0.5*(np.nanvar(xn)+np.nanvar(xa))+1e-9)

def _score(nv,ng):
    atk=_DATA[:,2]; u=_DATA[:,13:17]; vn=nv[atk==0]
    p90=float(np.nanpercentile(vn,90)) if len(vn) else float('nan')
    persist=float(np.nanmean(nv[atk==1]>p90)) if (atk==1).any() else 0.0
    tau=np.linalg.norm(u[:,1:4],axis=1); m=(atk==0)
    if m.sum()>20:
        thr=np.nanpercentile(tau[m],70); hi=nv[m&(tau>=thr)]; lo=nv[m&(tau<thr)]
        man=(np.nanmedian(hi)/(np.nanmedian(lo)+1e-9)) if len(hi) and len(lo) else 1.0
    else: man=1.0
    return dict(dv=_dp(nv,atk), dg=_dp(ng,atk), persist=persist,
                base_med=float(np.nanmedian(vn)) if len(vn) else float('nan'),
                base_p90=p90, man=float(man))

def _eval(p):
    nv,ng=_replay(p); s=_score(nv,ng); s.update(p); return s

# ══════════════════════════════════════════════════════════════════
#  압축기(NIS q → [0,1]) 비교  —  raw q를 복원해 4종을 같은 데이터에 씌움
# ══════════════════════════════════════════════════════════════════
_verf = np.vectorize(math.erf)
def _comp_raw(q): return q/3.0                                  # 압축 안함(per-DOF, 기준)
def _comp_log(q):                                              # 현재: log1p/(log1p+1)
    r=q/3.0; l=np.log1p(r); return l/(l+1.0)
def _comp_cdf(q):                                             # χ²₃ CDF (nz=3 닫힌형)
    q=np.maximum(q,0.0)
    v=_verf(np.sqrt(q/2.0)) - np.sqrt(2.0*q/np.pi)*np.exp(-q/2.0)
    return np.clip(v,0.0,1.0)
def _comp_sig(q, T=7.815, w=3.0):                            # 경계중심 시그모이드(T=χ²₃ 95%)
    return 1.0/(1.0+np.exp(-(q-T)/w))
_COMPRESSORS = {'raw':_comp_raw, 'log':_comp_log, 'cdf':_comp_cdf, 'sig':_comp_sig}

def _auc(score, atk):
    """Mann-Whitney AUC = P(공격score > 평시score). 단조압축에 불변 → 정보 상한."""
    s_n=score[atk==0]; s_a=score[atk==1]
    if len(s_n)<5 or len(s_a)<5: return 0.5
    allv=np.concatenate([s_n,s_a])
    order=np.argsort(allv,kind='mergesort')
    ranks=np.empty(len(allv)); ranks[order]=np.arange(1,len(allv)+1)
    r_a=ranks[len(s_n):].sum()
    return float((r_a - len(s_a)*(len(s_a)+1)/2.0)/(len(s_n)*len(s_a)))

def _score_arr(nv, ng, atk, u):
    """압축된 nis 배열로 d'/지속/p90/기동비 계산 (_score 일반화)."""
    vn=nv[atk==0]
    p90=float(np.nanpercentile(vn,90)) if len(vn) else float('nan')
    persist=float(np.nanmean(nv[atk==1]>p90)) if (atk==1).any() else 0.0
    tau=np.linalg.norm(u[:,1:4],axis=1); m=(atk==0)
    if m.sum()>20:
        thr=np.nanpercentile(tau[m],70); hi=nv[m&(tau>=thr)]; lo=nv[m&(tau<thr)]
        man=(np.nanmedian(hi)/(np.nanmedian(lo)+1e-9)) if len(hi) and len(lo) else 1.0
    else: man=1.0
    return dict(dv=_dp(nv,atk), dg=_dp(ng,atk), persist=persist, base_p90=p90, man=float(man))

def _eval_compressors(p):
    """한 파라미터셋: raw q 복원 → 압축기별 d'/지속/p90 + AUC(불변)."""
    nv,ng=_replay(p)                          # per-DOF raw (q/3)
    qv=nv*3.0; qg=ng*3.0                       # q 복원 (χ²(3))
    atk=_DATA[:,2]; u=_DATA[:,13:17]
    rows={name: _score_arr(fn(qv), fn(qg), atk, u) for name,fn in _COMPRESSORS.items()}
    return dict(rows=rows, auc_v=_auc(qv,atk), auc_g=_auc(qg,atk), p=p)

# 현재값 = 실제 ukf_filter.py의 튜닝된 Q/R (euler/gyro Q=5e-4, vel Q=3e-3, velR/gyroR=1.0)
CUR = dict(posQ=1e-3, eulerQ=5e-4, velQ=3e-3, gyroQ=5e-4, posR=0.5, velR=1.0, gyroR=1.0)

# 축별 탐색 후보 (요청 반영: gyroQ에 1e-2,5e-4 / R에 1.0,0.05)
AXES = dict(
    posQ  = [5e-4, 1e-3, 2e-3],
    eulerQ= [5e-4, 1e-3, 2e-3],
    velQ  = [1e-3, 2e-3, 3e-3, 5e-3],
    gyroQ = [5e-4, 1e-3, 1e-2],
    posR  = [0.05, 0.1, 0.25, 0.5, 1.0],
    velR  = [0.05, 0.1, 0.25, 0.5, 1.0],
    gyroR = [0.05, 0.1, 0.25, 0.5, 1.0],
)

def _fmt_row(p, s, tag=''):
    return (f"pQ{p['posQ']:.0e} eQ{p['eulerQ']:.0e} vQ{p['velQ']:.0e} gQ{p['gyroQ']:.0e} | "
            f"pR{p['posR']:.2f} vR{p['velR']:.2f} gR{p['gyroR']:.2f} | "
            f"d'v={s['dv']:5.2f} d'g={s['dg']:5.2f} 지속={s['persist']*100:4.0f}% "
            f"p90={s['base_p90']:.3f} 기동비={s['man']:.2f}{tag}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('zu'); ap.add_argument('--calib',default='calibration.json')
    ap.add_argument('--episodes',type=int,default=None,help='앞 N에피소드만(속도↑)')
    ap.add_argument('--max-steps',type=int,default=None)
    ap.add_argument('--jobs',type=int,default=None)
    ap.add_argument('--top',type=int,default=15)
    ap.add_argument('--no-grid',action='store_true',help='GRID 생략(1D만)')
    ap.add_argument('--no-compress',action='store_true',help='압축기 비교 생략')
    args=ap.parse_args()

    npz=np.load(args.zu,allow_pickle=True); data=npz['data']
    dt=float(npz['dt']) if 'dt' in npz else 0.02
    if args.episodes is not None:
        eps=np.unique(data[:,0])[:args.episodes]; data=data[np.isin(data[:,0],eps)]
    if args.max_steps is not None: data=data[:args.max_steps]
    calib=load_calibration(args.calib)
    jobs=args.jobs or cpu_count()
    print(f"로드: {len(data)}스텝, {len(np.unique(data[:,0]))}에피소드, dt={dt}, "
          f"평시={int((data[:,2]==0).sum())} 공격={int((data[:,2]==1).sum())} | {jobs}코어")

    pool=Pool(processes=jobs,initializer=_init,initargs=(data,dt,calib))
    try:
        # 현재값 기준 평가
        cur_s=pool.map(_eval,[CUR])[0]
        print("\n[현재설정] "+_fmt_row(CUR,cur_s))
        cur_p90=cur_s['base_p90']
        best_cfg=None

        # ── [1] SWEEP-1D: 한 축씩 (나머지 현재값 고정) ──
        print("\n"+"="*92)
        print("[1] SWEEP-1D : 각 축을 하나씩 변경(나머지 현재값 고정) → 효과 있는 축 식별")
        print("="*92)
        for ax, vals in AXES.items():
            combos=[{**CUR, ax:v} for v in vals]
            res=pool.map(_eval,combos)
            print(f"\n── {ax} (현재={CUR[ax]:g}) ──   [d'v↑·지속↑ 좋음 / p90↓ 좋음]")
            for v,s in zip(vals,res):
                t=' ← 현재' if abs(v-CUR[ax])<1e-12 else ''
                print(f"   {ax:6s}={v:<9g} | d'v={s['dv']:5.2f} d'g={s['dg']:5.2f} "
                      f"지속={s['persist']*100:4.0f}% p90={s['base_p90']:.3f} 기동비={s['man']:.2f}{t}")

        # ── [2] GRID: 핵심 축 곱집합 + ★추천 ──
        if not args.no_grid:
            print("\n"+"="*92)
            print("[2] GRID : 핵심 3축(velQ × velR × gyroR) 곱집합 + ★추천  (posR은 1D로 판단)")
            print("="*92)
            gvelQ=[1e-3,2e-3,3e-3,5e-3]; gvelR=[0.05,0.1,0.25,0.5,1.0]; ggyroR=[0.05,0.1,0.5]
            combos=[{**CUR,'velQ':vq,'velR':vr,'gyroR':gr}
                    for vq,vr,gr in itertools.product(gvelQ,gvelR,ggyroR)]
            est=5.3e-3*len(data)*len(combos)/max(jobs,1)
            print(f"  조합 {len(combos)}개 → 예상 ~{est/60:.1f}분\n")
            t0=time.time(); rows=pool.map(_eval,combos); print(f"  (완료 {time.time()-t0:.0f}s)\n")
            rows.sort(key=lambda r:(r['dv']+0.5*r['persist']),reverse=True)
            for r in rows[:args.top]:
                print("  "+_fmt_row(r,r))
            # ★추천: 평시p90 ≤ 현재 인 것 중 d'v 최대
            ok=[r for r in rows if r['base_p90']<=cur_p90+1e-9]
            ok.sort(key=lambda r:r['dv'],reverse=True)
            print(f"\n  현재 평시p90={cur_p90:.3f} (이 이하로 오탐 유지)")
            if ok:
                b=ok[0]; best_cfg=b
                print(f"  ★ 추천: velQ={b['velQ']:.0e} velR={b['velR']:.2f} gyroR={b['gyroR']:.2f}")
                print(f"     → d'v {cur_s['dv']:.2f}→{b['dv']:.2f}, 지속 {cur_s['persist']*100:.0f}%→{b['persist']*100:.0f}%, p90 {cur_p90:.3f}→{b['base_p90']:.3f}")
            else:
                print("  ★ 현재 p90 이하 개선 없음 → 평시천장 약간 허용 시 위 1등 고려")

        # ── [3] 압축기 비교: NIS q → [0,1] 매핑 (log vs cdf vs sigmoid vs raw) ──
        if not args.no_compress:
            print("\n"+"="*92)
            print("[3] 압축기 비교 : NIS를 [0,1]로 어떻게 매핑하나 (같은 q에 4종 적용)")
            print("="*92)
            targets=[('현재설정', CUR)]
            if best_cfg is not None:
                targets.append(('추천설정', {k:best_cfg[k] for k in CUR}))
            comp_res=pool.map(_eval_compressors, [t[1] for t in targets])
            for (title,_p), comp in zip(targets, comp_res):
                print(f"\n── {title} ──   [AUC=정보상한(압축불변): vel={comp['auc_v']:.3f} gyr={comp['auc_g']:.3f}]")
                print("   압축기 |  d'vel  d'gyr   지속%   평시p90*  기동비   설명")
                _desc={'raw':'압축안함(per-DOF)','log':'현재(log1p)','cdf':'χ²₃ CDF(원리적)','sig':'경계중심 시그모이드'}
                for name in ['raw','log','cdf','sig']:
                    s=comp['rows'][name]
                    print(f"   {name:5s}  | {s['dv']:6.2f} {s['dg']:6.2f}  {s['persist']*100:5.0f}%   "
                          f"{s['base_p90']:7.3f}   {s['man']:5.2f}   {_desc[name]}")
            print("\n  * 평시p90은 압축기마다 단위가 달라(raw=q/3, 나머지=[0,1]) 압축기 간 직접비교 불가.")
            print("    비교 기준: d'vel(분리도, 무차원)·지속%(분수) — 둘은 압축기 간 비교 가능.")
            print("    AUC가 모든 압축기에서 동일 = 정보상한. d'가 가장 큰 압축기가 그 정보를 [0,1]에 가장 잘 펼침.")
    finally:
        pool.close(); pool.join()

    print("\n읽는 법: d'v↑+지속%↑=공격분리↑(목표). p90↓=오탐↓.")
    print("  SWEEP-1D에서 바꿔도 d'·p90 거의 안 변하는 축 = 효과없음(고정). 크게 변하는 축 = 핵심레버.")

if __name__=='__main__':
    main()
