import numpy as np
import json
import glob
import os
from scipy.signal import butter, filtfilt, savgol_filter

# ⚠ 2026-07-28: 여기에 질량/관성을 하드코딩하면 calibration.json 의 drone 블록과 갈라진다.
#   실제로 그 사고가 났었다 — 스크립트는 1.5kg 가정으로 계수를 적합했는데 calibration.json 의
#   drone 만 1.372kg 로 손수정돼, UKF 가 쓰는 C/m 비율이 어긋난 채로 오래 돌아갔다.
#   (게다가 Iris 의 실제 비행질량은 로터 4개 포함 1.6186kg 이었다.)
#   → DRONE 은 항상 calibration.json 에서 읽는다. 단일 출처 유지.
def load_drone(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    path = path or os.path.join(here, 'calibration.json')
    with open(path) as f:
        return json.load(f)['drone']


DRONE = load_drone()
def apply_zero_phase_filter(data, fs=50):
    """지연 없는 영위상 필터링 적용 (IEEE 표준)"""
    nyq = 0.5 * fs
    b, a = butter(4, min(8, 0.8 * nyq) / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)

def run_sysid_ols(files):
    m, g = DRONE['mass'], DRONE['g']
    I = np.array([DRONE['Ixx'], DRONE['Iyy'], DRONE['Izz']])

    Y_z, Phi_z = [], []
    Y_x, Phi_x = [], []
    Y_y, Phi_y = [], []
    Y_tq, Phi_tq = [], []

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        dt = float(data['dt'])
        fs = 1.0 / dt

        gyro_f = apply_zero_phase_filter(data['gyro'], fs=fs)
        ang_accel = savgol_filter(gyro_f, window_length=11, polyorder=3, deriv=1, delta=dt, axis=0)

        euler = data['euler']
        accel_body = data['accelerometer']
        vel_ned = data['velocity']
        thrust_in = data['thrust'][:, 2]
        torque_in = data['torque']

        for k in range(int(len(euler)*0.1), int(len(euler)*0.9)):
            phi, th, psi = euler[k]
            cp, sp, ct, st, cps, sps = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)

            R = np.array([
                [ct*cps,  sp*st*cps - cp*sps,  cp*st*cps + sp*sps],
                [ct*sps,  sp*st*sps + cp*cps,  cp*st*sps - sp*cps],
                [-st,     sp*ct,               cp*ct             ]
            ])

            a_ned = R @ accel_body[k]

            Y_z.append(m * a_ned[2])
            Phi_z.append([thrust_in[k], -vel_ned[k, 2]])

            Y_tq.append(I * ang_accel[k])
            Phi_tq.append(torque_in[k])

    res_z = np.linalg.lstsq(np.array(Phi_z), np.array(Y_z), rcond=None)[0]
    c_thrust, c_drag_z = float(res_z[0]), float(res_z[1])

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        for k in range(int(len(data['euler'])*0.1), int(len(data['euler'])*0.9)):
            phi, th, psi = data['euler'][k]
            cp, sp, ct, st, cps, sps = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
            R = np.array([
                [ct*cps,  sp*st*cps - cp*sps,  cp*st*cps + sp*sps],
                [ct*sps,  sp*st*sps + cp*cps,  cp*st*sps - sp*cps],
                [-st,     sp*ct,               cp*ct             ]
            ])

            f_thr_ned = R @ np.array([0, 0, c_thrust * data['thrust'][k, 2]])

            a_ned = R @ data['accelerometer'][k]
            f_net_ned = m * a_ned - f_thr_ned

            Y_x.append(f_net_ned[0])
            Phi_x.append([-data['velocity'][k, 0]])

            Y_y.append(f_net_ned[1])
            Phi_y.append([-data['velocity'][k, 1]])

    res_x = np.linalg.lstsq(np.array(Phi_x), np.array(Y_x), rcond=None)[0]
    res_y = np.linalg.lstsq(np.array(Phi_y), np.array(Y_y), rcond=None)[0]
    c_drag_x, c_drag_y = float(res_x[0]), float(res_y[0])

    Phi_tq_arr, Y_tq_arr = np.array(Phi_tq), np.array(Y_tq)
    c_torques = [float(np.linalg.lstsq(Phi_tq_arr[:, i:i+1], Y_tq_arr[:, i], rcond=None)[0][0]) for i in range(3)]

    return c_thrust, c_torques, [c_drag_x, c_drag_y, abs(c_drag_z)]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='data_raw/ep*.npz',
                    help='입력 npz 글롭 (online_rl_main --log-sysid 산출물도 동일 포맷)')
    ap.add_argument('--out', default=None, help='기록할 json (미지정 시 출력만)')
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[!] 입력 데이터가 없습니다: {args.glob}")
        return

    print(f"[*] DRONE(= calibration.json drone) = {DRONE}")
    print(f"[*] 입력 {len(files)}개: {files}")
    c_thr, c_tqs, drags = run_sysid_ols(files)

    calib = {
        'C_thrust': c_thr,
        'C_torque_xy': (c_tqs[0] + c_tqs[1]) / 2.0,
        'C_torque_z': c_tqs[2],
        'drag': drags,
        'drone': DRONE,
        'note': "SysId via OLS with Zero-phase filtering (Gravity bug fixed)"
    }

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(calib, f, indent=2, ensure_ascii=False)
        print(f"[*] 기록 → {args.out}")

    print(f"[*] 시스템 식별(캘리브레이션) 완료!")
    print(f"    - C_torque(축별): x={c_tqs[0]:.4f} y={c_tqs[1]:.4f} z={c_tqs[2]:.4f}")
    print(f"    - C_thrust: {c_thr:.4f} (정상: 20~50)")
    print(f"    - C_torque: XY={calib['C_torque_xy']:.4f}, Z={c_tqs[2]:.4f}")
    print(f"    - Drag:     X={drags[0]:.4f}, Y={drags[1]:.4f}, Z={drags[2]:.4f} (정상: 양수 0.05~0.5)")

if __name__ == '__main__':
    main()
