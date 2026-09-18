#!/usr/bin/env python3
"""check_traj_equiv — sim(online_rl_main._compute_setpoint) vs px4_field/traj_common.TrajGen 동치 검증.
같은 dt 로 나란히 전진, 위치·속도·yaw 최대차. 사용: source ROS; python3 etc/scripts/check_traj_equiv.py"""
import sys, os, math, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf/px4_field')
os.chdir('/home/acsl/projects/Issacsim-rhukf')
os.environ.setdefault('SPEED_MOD_AMP', '0.5'); os.environ.setdefault('SPEED_MOD_FREQ', '0.5')
import swrl_config
from traj_common import TrajGen
import online_rl_main as M
cfg = swrl_config.Config()
class Fake:  # _compute_setpoint 가 쓰는 속성만
    pass
def run_sim(pattern, dt, n):
    f = Fake(); f.cfg = cfg; f.scenario = {'pattern': pattern}; f.step_dt = dt; f.tick_count = 0
    f._speed_mod_amp = float(os.environ['SPEED_MOD_AMP']); f._speed_mod_freq = float(os.environ['SPEED_MOD_FREQ'])
    f._sim_flight_t = 0.0; f._traj_t = 0.0; f._dt_sim_last = dt; f._wp_s = 0.0; f.theta = 0.0
    out = []
    for i in range(n):
        sp = M.OnlineRLController._compute_setpoint(f) if hasattr(M, 'OnlineRLController') else None
        out.append(sp); f._sim_flight_t += dt; f.tick_count += 1
    return np.array(out, dtype=float)
def run_f5(pattern, dt, n):
    g = TrajGen(pattern, R=cfg.flight_radius, omega=cfg.flight_omega, fig8_scale=cfg.fig8_radius_scale,
                wp_box_halfwidth=cfg.wp_box_halfwidth, wp_v_corner=cfg.wp_v_corner, wp_v_cruise=cfg.wp_v_cruise, wp_a_prof=cfg.wp_a_prof,
                scurve_radius=cfg.scurve_radius, scurve_arcs=cfg.scurve_arcs, scurve_dz=cfg.scurve_dz,
                agg_radius=cfg.agg_radius, agg_dz1=cfg.agg_dz1, agg_dz2=cfg.agg_dz2, agg_phase_s=cfg.agg_phase_s,
                speed_mod_amp=float(os.environ['SPEED_MOD_AMP']), speed_mod_freq=float(os.environ['SPEED_MOD_FREQ']))
    alt = -abs(cfg.flight_altitude)
    out = []
    for i in range(n):
        x, y, dz, yaw, vx, vy, vz = g.step(dt)
        out.append((x, y, alt + dz, yaw, vx, vy, vz))
    return np.array(out)
cls = [c for c in dir(M) if 'Controller' in c or 'Node' in c]
print('sim 클래스 후보:', cls)
SimCls = M.OnlineRLNode
def run_sim2(pattern, dt, n):
    f = Fake(); f.cfg = cfg; f.scenario = {'pattern': pattern}; f.step_dt = dt; f.tick_count = 0
    f._speed_mod_amp = float(os.environ['SPEED_MOD_AMP']); f._speed_mod_freq = float(os.environ['SPEED_MOD_FREQ'])
    f._sim_flight_t = 0.0; f._traj_t = 0.0; f._dt_sim_last = dt; f._wp_s = 0.0; f.theta = 0.0
    out = []
    for i in range(n):
        out.append(SimCls._compute_setpoint(f)); f._sim_flight_t += dt; f.tick_count += 1
    return np.array(out, dtype=float)
dt = 0.02; n = int(80 / dt)
print(f"{'pattern':11s} {'max|Δpos|':>10s} {'max|Δv|':>9s} {'max|Δyaw|':>10s}  (dt={dt}, {n*dt:.0f}s)")
for p in ('circle', 'figure8', 'waypoint', 'scurve', 'aggressive'):
    a = run_sim2(p, dt, n); b = run_f5(p, dt, n)
    dp = np.linalg.norm(a[:, :3] - b[:, :3], axis=1).max(); dv = np.linalg.norm(a[:, 4:7] - b[:, 4:7], axis=1).max()
    dy = np.abs((a[:, 3] - b[:, 3] + np.pi) % (2 * np.pi) - np.pi).max()
    print(f"{p:11s} {dp:10.4f} {dv:9.4f} {dy:10.4f}")
