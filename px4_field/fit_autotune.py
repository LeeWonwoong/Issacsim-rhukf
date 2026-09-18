#!/usr/bin/env python3
"""fit_autotune.py — PX4 오토튠 로그에서 G = C_torque/I 를 뽑는다 (2026-08-06)

    ~/isaacsim/python.sh px4_field/fit_autotune.py <ulg> [--I Ixx,Iyy,Izz]

무엇을 읽나
  `autotune_attitude_control_status` (기본 로깅 프로파일, 10 Hz)
      coeff[5] = [a1, a2, b0, b1, b2]     식별된 이산모델 계수
      coeff_var[5]                        계수 분산 (< 50 이어야 수렴)
      dt_model                            모델 샘플시간 [s]
      state  ★ PX4 실제 정의 (v1.16.2 AutotuneAttitudeControlStatus.msg)
                 0 IDLE  1 INIT  2 ROLL  3 ROLL_PAUSE  4 PITCH  5 PITCH_PAUSE
                 6 YAW   7 YAW_PAUSE  8 VERIFICATION  9 APPLY  10 TEST
                 11 COMPLETE  12 FAIL  13 WAIT_FOR_DISARM
             ⚠ 2026-08-06 수정: 이전 코드가 3/6/9 를 roll/pitch/yaw 로 읽었다.
               실제로는 ROLL_PAUSE / YAW / APPLY 라서 축이 통째로 어긋났었다.
               (증상: roll 이 +27%, pitch 가 −42%, yaw 는 n=3 으로 완전 실패)

모델 (PX4 ArxRls<2,2,1>, arx_rls.hpp)
      y[k] + a1·y[k−1] + a2·y[k−2] = b0·u[k−1] + b1·u[k−2] + b2·u[k−3]
      u = 정규화 토크 setpoint (vehicle_torque_setpoint — 우리 UKF 입력과 동일)
      y = 각속도 [rad/s]

환산
  토크→각속도 플랜트는 적분기(ω̇ = G·u)이므로 A(q⁻¹) 에 (1−q⁻¹) 인자가 있어야 한다.
      1 + a1·q⁻¹ + a2·q⁻² = (1 − q⁻¹)(1 + c·q⁻¹),  c = −a2  ⟹  a1 + a2 = −1
  따라서
                 b0 + b1 + b2
      G  =  ──────────────────────      [rad/s² per 정규화 단위]
            dt_model · (1 − a2)
  그리고
      C_torque = G · I          (실기 UKF 계수)
      I        = C_torque / G   (sim 관성 재조정)

  ★ 이 스크립트는 sim 에서 먼저 검증한다. sim 은 참값을 알고 있으므로
    (calibration.json 의 C_torque 와 I) 복원 여부로 절차 자체를 판정할 수 있다.
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    from pyulog import ULog
except ImportError:
    sys.exit('pyulog 가 없다.  ~/isaacsim/python.sh 로 실행할 것.')

AXES = [('roll', 2), ('pitch', 4), ('yaw', 6)]   # STATE_ROLL / STATE_PITCH / STATE_YAW
VAR_THR = 50.0          # PX4 의 수렴 판정 임계 (mc_autotune: converged_thr)


STATE_YAW, STATE_COMPLETE = 6, 11
STATE_NAME = {0: 'IDLE', 1: 'INIT', 2: 'ROLL', 3: 'ROLL_PAUSE', 4: 'PITCH',
              5: 'PITCH_PAUSE', 6: 'YAW', 7: 'YAW_PAUSE', 8: 'VERIFICATION',
              9: 'APPLY', 10: 'TEST', 11: 'COMPLETE', 12: 'FAIL',
              13: 'WAIT_FOR_DISARM'}


def split_runs(state):
    """state 배열을 오토튠 실행 단위로 쪼갠다 → [(i0, i1), ...]

    ⚠ 왜 필요한가 (2026-08-06):
      한 ulog 에 오토튠이 여러 번 들어갈 수 있다. v1.16.2 는 '부팅당 1회' 제약이
      없어서 현장에서 재시도하면 반드시 그렇게 된다. 실행을 안 나누고 축별로
      state 를 통째로 모으면 **중단된 시도와 완주분이 섞여** 엉뚱한 계수가 나온다.
      (실측: 3회분이 섞여 roll n=147 로 잡혔다)
      IDLE(0) 이 아닌 연속 구간 하나 = 실행 1회.
    """
    runs, cur = [], None
    for i, v in enumerate(state):
        if v != 0 and cur is None:
            cur = i
        elif v == 0 and cur is not None:
            runs.append((cur, i)); cur = None
    if cur is not None:
        runs.append((cur, len(state)))
    return runs


def run_completed(seg):
    """YAW 까지 갔고 COMPLETE 를 찍었으면 완주."""
    return bool((seg == STATE_YAW).any() and (seg == STATE_COMPLETE).any())


def g_from_coeff(c, dt):
    """coeff=[a1,a2,b0,b1,b2] → (G, a1+a2 잔차)"""
    a1, a2, b0, b1, b2 = c
    integ_resid = a1 + a2 + 1.0            # 0 이어야 적분기
    denom = dt * (1.0 - a2)
    if abs(denom) < 1e-12:
        return float('nan'), integ_resid
    return (b0 + b1 + b2) / denom, integ_resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ulg')
    ap.add_argument('--I', default=None,
                    help='Ixx,Iyy,Izz [kg m^2]. 생략하면 calibration.json')
    ap.add_argument('--calib', default=None, help='calibration.json 경로')
    ap.add_argument('--tail', type=int, default=10,
                    help='축별로 마지막 N 샘플의 중앙값을 쓴다')
    ap.add_argument('--run', type=int, default=None,
                    help='몇 번째 실행을 쓸지 (1-based). 생략하면 마지막 완주분')
    ap.add_argument('--all-runs', action='store_true',
                    help='실행 목록만 보여주고 끝')
    a = ap.parse_args()

    ulog = ULog(a.ulg, ['autotune_attitude_control_status'])
    if not ulog.data_list:
        sys.exit('✗ autotune_attitude_control_status 가 로그에 없다.\n'
                 '  MC_AT_EN=1 인지, 오토튠이 실제로 시작됐는지 확인할 것.')
    d = ulog.data_list[0].data
    n = len(d['timestamp'])
    print(f'샘플 {n} 개')

    state = d['state'].astype(int)
    dt_model = d['dt_model']
    coeff = np.stack([d[f'coeff[{i}]'] for i in range(5)], axis=1)
    cvar = np.stack([d[f'coeff_var[{i}]'] for i in range(5)], axis=1)
    fitness = d.get('fitness', np.full(n, np.nan))

    # ── 관성 ──
    if a.I:
        I = np.array([float(x) for x in a.I.split(',')])
        src = 'CLI'
    else:
        p = a.calib or os.path.join(os.path.dirname(__file__), '..',
                                    'calibration', 'calibration.json')
        with open(p) as f:
            cal = json.load(f)
        dr = cal['drone']
        I = np.array([dr['Ixx'], dr['Iyy'], dr['Izz']])
        Ct = np.array([cal.get('C_torque_x', cal['C_torque_xy']),
                       cal.get('C_torque_y', cal['C_torque_xy']),
                       cal['C_torque_z']])
        src = os.path.normpath(p)
    print(f'I = {I}   ({src})')

    # ── 실행 분할: 섞이지 않게 하나만 고른다 ──────────────
    t = d['timestamp'] / 1e6
    runs = split_runs(state)
    if not runs:
        sys.exit('✗ 로그에 오토튠 실행이 하나도 없다 (state 가 계속 IDLE).')

    print(f'\n오토튠 실행 {len(runs)} 회')
    done_idx = []
    for k, (i0, i1) in enumerate(runs, 1):
        seg = state[i0:i1]
        done = run_completed(seg)
        if done:
            done_idx.append(k)
        far = STATE_NAME.get(int(seg.max()), '?')
        print(f'   [{k}] {t[i0]-t[0]:7.1f}s ~ {t[i1-1]-t[0]:7.1f}s '
              f'({t[i1-1]-t[i0]:5.1f}s)  최종상태={far:12s} '
              f'{"✅ 완주" if done else "✗ 중단"}')

    if a.all_runs:
        return

    if a.run is not None:
        if not 1 <= a.run <= len(runs):
            sys.exit(f'✗ --run 은 1~{len(runs)} 범위여야 한다.')
        pick = a.run
    elif done_idx:
        pick = done_idx[-1]                       # 마지막 완주분
    else:
        pick = len(runs)
        print('\n  ⚠ 완주한 실행이 없다. 마지막 실행으로 진행하지만 계수를 믿지 말 것.')
    i0, i1 = runs[pick - 1]
    print(f'\n→ 실행 [{pick}] 사용  ({t[i1-1]-t[i0]:.1f}s)')

    sel = np.zeros(n, bool)
    sel[i0:i1] = True

    # ── 축별 best-run 선택 (--repeat 활용) ────────────────
    #  각 축에서 maxVar 최소(가장 수렴)인 실행을 따로 고른다. 요가 2회차에만 수렴해도 그걸 쓴다.
    per_axis = (len(runs) > 1) and (a.run is None)
    def _axis_idx(st):
        """(idx, run_k, maxvar) 또는 None. per_axis 면 전 실행 중 maxVar 최소."""
        if not per_axis:
            m = (state == st) & sel
            if m.sum() == 0:
                return None
            idx = np.where(m)[0][-a.tail:]
            return idx, pick, float(np.median(cvar[idx], axis=0).max())
        best = None
        for rk, (r0, r1) in enumerate(runs, 1):
            mm = np.zeros(n, bool); mm[r0:r1] = True; mm &= (state == st)
            if mm.sum() == 0:
                continue
            ii = np.where(mm)[0][-a.tail:]
            vv = float(np.median(cvar[ii], axis=0).max())
            if best is None or vv < best[2]:
                best = (ii, rk, vv)
        return best
    if per_axis:
        print('  (반복 실행 감지 → 축별로 가장 수렴한 실행을 자동 선택)')

    print()
    print(f'{"축":6} {"n":>4} {"run":>4} {"dt[ms]":>7} {"a1+a2+1":>9} {"maxVar":>8} '
          f'{"G[rad/s²/u]":>12} {"C_torque=G·I":>13}')
    print('─' * 70)

    out = {}
    for k, (name, st) in enumerate(AXES):
        r = _axis_idx(st)
        if r is None:
            print(f'{name:6} {"—":>4}  (해당 구간 없음 — 그 축은 수렴 전 중단)')
            continue
        idx, run_k, _mv = r
        c = np.median(coeff[idx], axis=0)
        v = np.median(cvar[idx], axis=0)
        dt = float(np.median(dt_model[idx]))
        G, resid = g_from_coeff(c, dt)
        ok_int = abs(resid) < 0.15
        ok_var = v.max() < VAR_THR
        flag = '' if (ok_int and ok_var) else '  ⚠'
        if not ok_int:
            flag += ' 적분기아님'
        if not ok_var:
            flag += ' 미수렴'
        print(f'{name:6} {len(idx):4d} {run_k:4d} {dt*1e3:7.2f} {resid:+9.3f} '
              f'{v.max():8.1f} {G:12.1f} {G*I[k]:13.3f}{flag}')
        out[name] = dict(G=float(G), dt=dt, coeff=c.tolist(), run=int(run_k),
                         resid=float(resid), maxvar=float(v.max()),
                         C_torque=float(G * I[k]))

    if not a.I and out:
        print()
        print('참값 대조 (calibration.json — sim 리허설에서만 의미 있음)')
        print(f'{"축":6} {"측정 G":>10} {"참값 G":>10} {"오차":>8}')
        print('─' * 40)
        for k, (name, _) in enumerate(AXES):
            if name not in out:
                continue
            g_true = Ct[k] / I[k]
            e = out[name]['G'] / g_true - 1.0
            print(f'{name:6} {out[name]["G"]:10.1f} {g_true:10.1f} {e:+7.1%}')
        print()
        print('  판정 기준: |오차| ≤ 20% → 근사치 유지.  > 20% → I ← C_torque/G 재조정')

    with open(os.path.splitext(a.ulg)[0] + '_autotune.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\n저장: {os.path.splitext(a.ulg)[0]}_autotune.json')


if __name__ == '__main__':
    main()
