#!/usr/bin/env python3
"""
peek_accel.py — zu_log의 가속도 채널이 공격 분리에 쓸모있는지 빠르게 판단
============================================================================
새 zu_log.npz(가속도 3축 포함)를 읽어, 각 축(bx,by,bz)에 대해:
  1) 좌표/부호 정합 검증: 평시(공격X) 잔차 평균이 0 근처인가?
     (예측 비력 = (추력 + 항력)/m 과 측정 가속도가 같은 프레임/부호인지)
  2) 잔차 d' (공격 vs 평시): 그 축이 공격을 얼마나 분리하나?
     - raw 가속도 d'도 같이 (기동에 안 흔들리는지 비교용)

잔차 = 측정가속도(body) - 예측비력(body),  예측비력 = (f_thrust + drag)/m  ← _f와 동일식

사용: python3 peek_accel.py results_zu/zu_log.npz
"""
import sys, numpy as np
try:
    from ukf_filter import load_calibration
except Exception:
    from env.ukf_filter import load_calibration

def predicted_specific_force_body(euler, v_ned, thrust, drag, m):
    """_f(95~123줄)과 동일한 body-frame 비력 = (추력 + 항력)/m."""
    phi, th, psi = euler
    vx, vy, vz = v_ned
    cp,sp=np.cos(phi),np.sin(phi); ct,st=np.cos(th),np.sin(th); cps,sps=np.cos(psi),np.sin(psi)
    # ned vel → body vel (─ _f 110~114줄과 동일 ─)
    vbx = cps*ct*vx + sps*ct*vy - st*vz
    vby = (cps*st*sp - sps*cp)*vx + (sps*st*sp + cps*cp)*vy + ct*sp*vz
    vbz = (cps*st*cp + sps*sp)*vx + (sps*st*cp - cps*sp)*vy + ct*cp*vz
    fd = np.array([-drag[0]*vbx, -drag[1]*vby, -drag[2]*vbz])
    f_thrust_body = np.array([0.0, 0.0, -thrust])
    return (f_thrust_body + fd) / m

def dprime(x, a):
    xn, xa = x[a==0], x[a==1]
    if len(xn)<5 or len(xa)<5: return 0.0
    return (np.nanmean(xa)-np.nanmean(xn))/np.sqrt(0.5*(np.nanvar(xn)+np.nanvar(xa))+1e-9)

def main():
    if len(sys.argv) < 2:
        sys.exit("사용: python3 peek_accel.py results_zu/zu_log.npz")
    npz = np.load(sys.argv[1], allow_pickle=True); d = npz['data']
    if d.shape[1] < 23:
        sys.exit(f"[!] 컬럼이 {d.shape[1]}개 — 가속도(3축)가 안 들어있습니다. "
                 f"online_rl_main 갱신 후 --log-zu로 새로 수집하세요. (기대 23+)")
    calib = load_calibration(sys.argv[2] if len(sys.argv)>2 else 'calibration.json')
    m = calib['drone']['mass']; drag = np.array(calib.get('drag',[0.1,0.1,0.2]), float)

    atk   = d[:,2]
    v_ned = d[:,7:10]      # z_9d[3:6] = gps_vel_ned
    thr   = d[:,13]        # u_phys[0] = thrust
    eul   = d[:,17:20]
    acc   = d[:,20:23]     # 측정 body 가속도 (m/s²)

    # 예측 비력 + 잔차
    pred = np.array([predicted_specific_force_body(eul[i], v_ned[i], thr[i], drag, m)
                     for i in range(len(d))])
    resid = acc - pred

    print(f"로드 {len(d)}스텝 | 평시 {int((atk==0).sum())} 공격 {int((atk==1).sum())} | mass={m:.3f}")
    print("\n[1] 좌표/부호 정합 검증 — 평시 잔차 평균이 0 근처여야 정합 OK")
    print("    (|평균|이 g(9.81)이나 큰 상수면 → 프레임/부호 불일치 → 먼저 고쳐야 함)")
    for i,ax in enumerate(['bx','by','bz']):
        rn = resid[atk==0,i]
        print(f"   accel_{ax}: 평시잔차 평균={np.nanmean(rn):8.3f}  std={np.nanstd(rn):6.3f}  "
              f"(측정평시평균={np.nanmean(acc[atk==0,i]):7.3f} 예측평시평균={np.nanmean(pred[atk==0,i]):7.3f})")

    print("\n[2] 잔차 d' (공격 vs 평시) — 클수록 그 축이 공격을 잘 분리")
    print("    raw d'(원시 가속도)도 비교: 잔차 d'≫raw d'이면 '예측이 기동을 잘 빼줘서 공격만 남음'(좋음)")
    for i,ax in enumerate(['bx','by','bz']):
        dr = dprime(resid[:,i], atk); rw = dprime(acc[:,i], atk)
        print(f"   accel_{ax}: 잔차 d'={dr:6.2f}   raw d'={rw:6.2f}")
    # 잔차 norm (3축 합) 채널도
    rn3 = np.linalg.norm(resid, axis=1)
    print(f"   accel_norm: 잔차 d'={dprime(rn3,atk):6.2f}  (3축 합)")

    print("\n해석:")
    print("  [1] 평시잔차 평균≈0 → 정합 OK. 큰 상수 → 부호/프레임 고쳐야(특히 bz는 ±g 확인).")
    print("  [2] 잔차 d'가 큰 축 = UKF 관측에 추가할 가치 있음. bz↑=추력공격, bx/by↑=수평(자세경유).")
    print("      기존 nis_vel d'(~0.84)과 비교해 더 크면 → 해상도 이득.")

if __name__=='__main__':
    main()
