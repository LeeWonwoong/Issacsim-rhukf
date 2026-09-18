#!/usr/bin/env python3
"""probe_rotor_geom.py — Iris USD 로터 위치 → 배분행렬 B (기하). 결과를 파일로 직접 기록."""
import numpy as np
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

OUT = 'allocation_B_report.txt'
f = open(OUT, 'w')
def w(s):
    f.write(s + '\n'); f.flush()

try:
    import omni.usd
    from pxr import Usd, UsdGeom
    from pegasus.simulator.params import ROBOTS

    usd_path = ROBOTS['Iris']
    stage = Usd.Stage.Open(usd_path)                       # USD 직접 열기(물리 불필요)
    w(f"USD: {usd_path}")

    rotors = [p for p in stage.Traverse()
              if 'rotor' in p.GetName().lower() and p.IsA(UsdGeom.Xformable)]
    # 최상위 로터 Xform 만(중복 자식 제외): path depth 최소
    tops, seen = [], set()
    for p in sorted(rotors, key=lambda x: str(x.GetPath())):
        key = p.GetName().lower()
        if key not in seen and p.GetTypeName() == 'Xform':
            tops.append(p); seen.add(key)
    w(f"로터 Xform {len(tops)}: {[p.GetName() for p in tops]}")

    def wp(p):
        m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        return np.array([t[0], t[1], t[2]])

    rel = np.array([wp(p) for p in tops[:4]])
    w("\nbody 원점 기준 위치 (m):")
    for i, p in enumerate(tops[:4]):
        w(f"  {p.GetName()}: x={rel[i,0]:+.4f} y={rel[i,1]:+.4f} z={rel[i,2]:+.4f}")

    if len(rel) >= 4:
        rot_dir = np.array([-1, -1, 1, 1])
        K_ROT, CM = 8.54858e-6, 7.2e-7
        kappa = CM / K_ROT
        B = np.zeros((4, 4))
        for i in range(4):
            B[:, i] = [-1.0, -rel[i, 1], rel[i, 0], rot_dir[i] * kappa]
        names = ['Fz', 'tau_x(roll)', 'tau_y(pitch)', 'tau_z(yaw)']
        w(f"\nkappa=CM/k={kappa:.5f} m  rot_dir={rot_dir.tolist()}")
        w("배분행렬 B (wrench = B · T_motor[N], 열=모터):")
        w(f"  {'':13s} {'M1':>9s} {'M2':>9s} {'M3':>9s} {'M4':>9s}")
        for j in range(4):
            w(f"  {names[j]:13s} " + ' '.join(f'{B[j,i]:9.4f}' for i in range(4)))
        w("\n모터 i 침해 → wrench 커플링 (열 정규화 |δτx|=1):")
        w(f"  {'모터':>5s} {'δFz':>8s} {'δτx':>8s} {'δτy':>8s} {'δτz':>8s}")
        for i in range(4):
            c = B[:, i] / (abs(B[1, i]) if abs(B[1, i]) > 1e-9 else 1)
            w(f"  M{i+1:>4d} {c[0]:8.3f} {c[1]:8.3f} {c[2]:8.3f} {c[3]:8.3f}")
        np.savez('allocation_B.npz', B=B, rel=rel, rot_dir=rot_dir, kappa=kappa,
                 cols='Fz,tau_x,tau_y,tau_z', note='geometric 2026-08-13')
        w("\n저장: allocation_B.npz")
    else:
        w(f"⚠ 로터 4개 못 찾음 ({len(rel)})")
except Exception as e:
    import traceback
    w("ERROR:\n" + traceback.format_exc())
finally:
    f.close()
    app.close()
