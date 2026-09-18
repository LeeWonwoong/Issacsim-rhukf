# -*- coding: utf-8 -*-
"""Q_gyro × R_gyro 2D 격자 (병렬). OFAT 에서 이 둘만 gyro 채널에 유효함이 확인됨.
τon 은 Q_gyro 가, τoff 는 R_gyro 가 지배 → 조합 최적점 탐색."""
import os,sys,json,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import numpy as np
from multiprocessing import Pool
sys.path.insert(0,'etc/scripts')
import qr_ofat_sweep as S

QG=[1e-4,3e-4,1e-3,3e-3,5e-3,2e-2]
RG=[1e-3,5e-3,2e-2,5e-2,0.2,0.5]
def job(a):
    qg,rg=a
    cfg=dict(S.BASE); cfg['q_gyr']=qg; cfg['r_gyr']=rg
    return dict(qg=qg,rg=rg,m=S.score(cfg))
if __name__=='__main__':
    jobs=[(q,r) for q in QG for r in RG]
    with Pool(min(len(jobs),24)) as p: res=p.map(job,jobs)
    json.dump(res,open('scratchpad/qr_grid2d.json','w'),default=str)
    print(f"{'Q_gyro':>8s} {'R_gyro':>8s} | {'바닥':>5s} {'고원':>5s} {'τon':>4s} {'τoff':>5s} {'고집':>5s} {'d′':>5s} {'OVL':>6s} | {'v바닥':>5s} {'vτoff':>5s} {'v d′':>5s}")
    for r in sorted(res,key=lambda x:(x['qg'],x['rg'])):
        g=r['m']['gyro']; v=r['m']['vel']
        print(f"{r['qg']:8g} {r['rg']:8g} | {g['base']:5.2f} {g['plateau']:5.2f} {g['tau_on']:4.0f} {g['tau_off']:5.0f} "
              f"{g['sustain']:5.2f} {g['dprime']:5.2f} {g['ovl']:6.3f} | {v['base']:5.2f} {v['tau_off']:5.0f} {v['dprime']:5.2f}")
