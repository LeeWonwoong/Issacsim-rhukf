# -*- coding: utf-8 -*-
"""후보 Q/R 조합 검증 (병렬) + CartPole 비교용 SNR/Bayes 지표 동시 산출."""
import os,sys,json,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.'); sys.path.insert(0,'etc/scripts')
import numpy as np
from multiprocessing import Pool
import qr_ofat_sweep as S

CAND={
 'A_현행':        dict(q_gyr=5e-3, r_gyr=0.2,  r_vel=0.1,  q_eul=5e-4, q_vel=5e-3),
 'B_빠른onoff':   dict(q_gyr=2e-2, r_gyr=5e-3, r_vel=0.02, q_eul=5e-4, q_vel=5e-3),
 'C_균형':        dict(q_gyr=2e-2, r_gyr=2e-2, r_vel=0.05, q_eul=5e-4, q_vel=5e-3),
 'D_신호유지':     dict(q_gyr=5e-3, r_gyr=1e-3, r_vel=0.02, q_eul=5e-4, q_vel=5e-3),
 'E_vel강조':     dict(q_gyr=2e-2, r_gyr=2e-2, r_vel=0.01, q_eul=2e-3, q_vel=5e-3),
 'F_최비자명':     dict(q_gyr=2e-2, r_gyr=1e-3, r_vel=0.01, q_eul=2e-3, q_vel=5e-3),
}
def extra(cfg):
    """SNR(CartPole 축) + 1스텝 Bayes 오류 추가 산출"""
    g,v,fr=S.replay(cfg)
    atk=S.DATA[:,2]==1
    BEN=fr&(~atk)&(S.SINCE>15); ST=fr&atk&(S.DELAY>=3); ON=fr&atk&(S.DELAY<3)
    out={}
    for nm,x in [('gyro',g),('vel',v)]:
        b=x[BEN]
        d={}
        for st,m in [('steady',ST),('onset',ON)]:
            sig=np.median(x[m])-np.median(b)
            d[st+'_snr']=float(sig/max(b.std(),1e-9))
            d[st+'_berr']=float(S.overlap(b,x[m])/2)
        d['sigma_norm']=float(b.std()/S.CLIP)
        out[nm]=d
    return out
def job(kv):
    nm,over=kv
    cfg=dict(S.BASE); cfg.update(over)
    return nm,cfg,S.score(cfg),extra(cfg)
if __name__=='__main__':
    with Pool(6) as p: res=p.map(job,list(CAND.items()))
    json.dump([(n,c,m,e) for n,c,m,e in res],open('scratchpad/qr_cand.json','w'),default=str)
    print("="*126)
    print(f"{'config':12s} | {'Qg':>6s} {'Rg':>6s} {'Rv':>5s} {'Qe':>6s} || {'g바닥':>5s} {'g고원':>5s} {'gτon':>4s} {'gτoff':>5s} {'g고집':>4s} {'g d′':>5s} {'gOVL':>5s} || {'v바닥':>5s} {'v고원':>5s} {'vτoff':>5s} {'v d′':>5s} {'vOVL':>5s}")
    print("="*126)
    for nm,cfg,m,e in res:
        g=m['gyro']; v=m['vel']
        print(f"{nm:12s} | {cfg['q_gyr']:6g} {cfg['r_gyr']:6g} {cfg['r_vel']:5g} {cfg['q_eul']:6g} || "
              f"{g['base']:5.2f} {g['plateau']:5.2f} {g['tau_on']:4.0f} {g['tau_off']:5.0f} {g['sustain']:4.2f} {g['dprime']:5.2f} {g['ovl']:5.3f} || "
              f"{v['base']:5.2f} {v['plateau']:5.2f} {v['tau_off']:5.0f} {v['dprime']:5.2f} {v['ovl']:5.3f}")
    print()
    print("── CartPole 비교축: 등가 σ_norm / SNR / 1스텝 Bayes 오류 ──")
    print(f"{'config':12s} | {'gσnorm':>7s} {'g SNR정상':>9s} {'g SNR온셋':>9s} {'gBErr정상':>9s} {'gBErr온셋':>9s} | {'vσnorm':>7s} {'v SNR정상':>9s} {'vBErr정상':>9s}")
    for nm,cfg,m,e in res:
        eg=e['gyro']; ev=e['vel']
        print(f"{nm:12s} | {eg['sigma_norm']:7.3f} {eg['steady_snr']:9.2f} {eg['onset_snr']:9.2f} {eg['steady_berr']:9.3f} {eg['onset_berr']:9.3f} "
              f"| {ev['sigma_norm']:7.3f} {ev['steady_snr']:9.2f} {ev['steady_berr']:9.3f}")
