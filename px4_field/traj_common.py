#!/usr/bin/env python3
"""traj_common — sim(online_rl_main._compute_setpoint) 과 **같은 식·같은 상태기계**의 궤적 생성기 (실기 이식용).

2026-09-10: f5_pattern.py 가 sim 과 어긋나 있던 것(R 2.8/ω 0.5, waypoint 등속, scurve 없음, 속도변조 없음,
aggressive Tp) 을 전부 sim frozen_v3 스펙으로 통일한다. 검증: etc/scripts/check_traj_equiv.py 가
sim 메서드와 이 클래스를 같은 dt 로 나란히 전진시켜 최대 위치차를 잰다.

기본값(= sim swrl_config + frozen_v3.env):
  R 3.4 · ω 0.38 · fig8 1/√2 · waypoint 박스 반폭 3.2 코너감속(vc 0.5 / vcr 1.7 / ap 2.0)
  scurve Rs 2.0 · arcs 3 · dz 0.8 · aggressive Ra 2.24 / dz1 1.0 / dz2 0.7 / Tp 5.6 (반원 1.26 m/s, 변조 피크 1.88 < 2.0)
  속도변조 amp 0.5 / freq 0.5 Hz (waypoint 제외 — 코너감속이 대체)
좌표: 패턴 로컬(NED 축 정렬, 원점=진입점). z 는 진입 고도 기준 오프셋(sim 의 alt 변수 대응).
반환: (x, y, dz, yaw, vx, vy, vz)
"""
import math

TWO_PI = 2.0 * math.pi


class TrajGen:
    def __init__(self, pattern, R=3.4, omega=0.38, fig8_scale=0.70710678,
                 wp_box_halfwidth=3.2, wp_v_corner=0.5, wp_v_cruise=1.7, wp_a_prof=2.0,
                 scurve_radius=2.0, scurve_arcs=3, scurve_dz=0.8,
                 agg_radius=2.24, agg_dz1=1.0, agg_dz2=0.7, agg_phase_s=5.6,
                 speed_mod_amp=0.5, speed_mod_freq=0.5):
        self.pattern = pattern
        self.R, self.w = float(R), float(omega)
        self.fig8_scale = float(fig8_scale)
        self.wp_k = float(wp_box_halfwidth) / 5.0
        self.vc, self.vcr, self.ap = float(wp_v_corner), float(wp_v_cruise), float(wp_a_prof)
        self.Rs, self.Na, self.dz_s = float(scurve_radius), int(scurve_arcs), float(scurve_dz)
        self.Ra, self.dz1, self.dz2, self.Tp = float(agg_radius), float(agg_dz1), float(agg_dz2), float(agg_phase_s)
        self.amp, self.fmod = float(speed_mod_amp), float(speed_mod_freq)
        # waypoint 박스 (sim: [[0,0],[5,0],[5,5],[-5,5],[-5,0],[0,0]] × kk)
        k = self.wp_k
        self.wps = [(0.0, 0.0), (5 * k, 0.0), (5 * k, 5 * k), (-5 * k, 5 * k), (-5 * k, 0.0), (0.0, 0.0)]
        self.seg_len = [math.hypot(self.wps[i + 1][0] - self.wps[i][0], self.wps[i + 1][1] - self.wps[i][1])
                        for i in range(len(self.wps) - 1)]
        self.L_tot = sum(self.seg_len)
        self.reset()

    def reset(self):
        self.t_real = 0.0     # 진입 후 실시간 [s]  (sim: _sim_flight_t)
        self.t_warp = 0.0     # 워프시간           (sim: _traj_t)
        self.wp_s = 0.0       # waypoint 호길이 상태 (sim: _wp_s)

    # ── 한 스텝 전진: sim 과 동일하게 "현재 상태로 setpoint 계산 → 시간 전진" ──
    def step(self, dt):
        """dt [s] 만큼 전진하며 이 스텝의 setpoint 를 돌려준다. 첫 호출은 t=0 의 setpoint."""
        if self.amp > 0.0 and self.pattern != 'waypoint':
            mod = max(0.15, 1.0 + self.amp * math.sin(TWO_PI * self.fmod * self.t_real))
            t = self.t_warp
            self.t_warp += mod * dt
        else:
            mod = 1.0
            t = self.t_real
        sp = self._sp(t, mod, dt)
        self.t_real += dt
        return sp

    def _sp(self, t, mod, dt):
        R, w = self.R, self.w
        p = self.pattern
        if p == 'hover':
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if p == 'circle':
            th = w * t
            x = R * math.cos(th) - R
            y = R * math.sin(th)
            vx = -R * w * math.sin(th) * mod
            vy = R * w * math.cos(th) * mod
            yaw = (th + math.pi / 2 + math.pi) % TWO_PI - math.pi
            return (x, y, 0.0, yaw, vx, vy, 0.0)
        if p == 'figure8':
            Rf = R * self.fig8_scale
            x = Rf * math.sin(w * t); y = (Rf / 2) * math.sin(2 * w * t)
            vx = Rf * w * math.cos(w * t) * mod; vy = Rf * w * math.cos(2 * w * t) * mod
            yaw = math.atan2(vy, vx) if (vx != 0 or vy != 0) else 0.0
            return (x, y, 0.0, yaw, vx, vy, 0.0)
        if p == 'waypoint':
            # sim wp_corner_decel=True 경로 그대로 (호길이 상태 전진, 코너 vc → 가속 → vcr → 코너 전 감속)
            sarc = self.wp_s % self.L_tot
            idx, acc0 = 0, 0.0
            for idx in range(len(self.seg_len)):
                if sarc < acc0 + self.seg_len[idx]:
                    break
                acc0 += self.seg_len[idx]
            d_in = sarc - acc0
            d_out = self.seg_len[idx] - d_in
            v_here = min(self.vcr,
                         math.sqrt(self.vc * self.vc + 2.0 * self.ap * max(d_in, 0.0)),
                         math.sqrt(self.vc * self.vc + 2.0 * self.ap * max(d_out, 0.0)))
            self.wp_s += v_here * dt
            f = d_in / self.seg_len[idx] if self.seg_len[idx] > 0 else 0.0
            x0, y0 = self.wps[idx]; x1, y1 = self.wps[idx + 1]
            dx, dy = x1 - x0, y1 - y0
            ux, uy = dx / self.seg_len[idx], dy / self.seg_len[idx]
            return (x0 + dx * f, y0 + dy * f, 0.0, math.atan2(dy, dx), v_here * ux, v_here * uy, 0.0)
        if p == 'scurve':
            Rs, Na = self.Rs, self.Na
            v_s = R * w
            L1 = math.pi * Rs * Na
            u_par = (v_s * t) % (2.0 * L1)
            fwd = u_par < L1
            s_arc = u_par if fwd else (2.0 * L1 - u_par)
            i = min(int(s_arc / (math.pi * Rs)), Na - 1)
            ph = s_arc / Rs - i * math.pi
            Cx = (2 * i + 1) * Rs
            if i % 2 == 0:
                th_a = math.pi - ph
                x = Cx + Rs * math.cos(th_a); y = Rs * math.sin(th_a)
                dxds = math.sin(th_a); dyds = -math.cos(th_a)
            else:
                th_a = math.pi + ph
                x = Cx + Rs * math.cos(th_a); y = Rs * math.sin(th_a)
                dxds = -math.sin(th_a); dyds = math.cos(th_a)
            x = -x; dxds = -dxds
            sgn = 1.0 if fwd else -1.0
            vx = sgn * v_s * dxds * mod; vy = sgn * v_s * dyds * mod
            lam = s_arc / Rs
            z_off = -self.dz_s * math.sin(lam)                       # NED: 음수 = 위
            vz = -sgn * self.dz_s * (v_s / Rs) * math.cos(lam) * mod
            yaw = math.atan2(vy, vx) if (vx != 0 or vy != 0) else 0.0
            return (x, y, z_off, yaw, vx, vy, vz)
        if p == 'aggressive':
            Ra, dz1, dz2, Tp = self.Ra, self.dz1, self.dz2, self.Tp
            # ★2026-09-21 sim(09-15 수정)과 동일: 위상을 궤적 시간 t 에서 직접 (구 틱 기반 spp/tk 는 dt 가 다르면 어긋났다)
            T = Tp
            phase = int(t // T) % 4
            f = (t % T) / T
            k = (math.pi / T) * mod
            if phase == 0:
                return (0.0, 0.0, -dz1 * math.sin(math.pi * f), 0.0, 0.0, 0.0, -dz1 * k * math.cos(math.pi * f))
            if phase == 1:
                a = math.pi * f
                return (Ra * math.cos(a) - Ra, Ra * math.sin(a), 0.0, a, -Ra * k * math.sin(a), Ra * k * math.cos(a), 0.0)
            if phase == 2:
                return (-2.0 * Ra, 0.0, dz2 * math.sin(math.pi * f), math.pi, 0.0, 0.0, dz2 * k * math.cos(math.pi * f))
            a = math.pi * (1 - f)
            return (Ra * math.cos(a) - Ra, Ra * math.sin(a), 0.0, a, Ra * k * math.sin(a), -Ra * k * math.cos(a), 0.0)
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # ── 한 바퀴(닫힌 궤적) 소요 실시간 추정 [s] (T_run 계산용) ──
    def lap_time(self):
        p = self.pattern
        if p in ('circle', 'figure8'):
            return TWO_PI / self.w                      # 속도변조 평균배율 1.0 → 실시간 ≈ 워프시간
        if p == 'scurve':
            return 2.0 * math.pi * self.Rs * self.Na / (self.R * self.w)
        if p == 'aggressive':
            return 4.0 * self.Tp
        if p == 'waypoint':                             # 사다리꼴 프로파일 적분 (dt 0.05 시뮬)
            g = TrajGen('waypoint', wp_box_halfwidth=self.wp_k * 5.0, wp_v_corner=self.vc,
                        wp_v_cruise=self.vcr, wp_a_prof=self.ap, speed_mod_amp=0.0)
            t = 0.0
            while g.wp_s < self.L_tot - 1e-6 and t < 600.0:
                g.step(0.05); t += 0.05
            return t
        return 10.0

    def reach(self):
        """origin 기준 최대 이탈 거리 [m] (안전반경용)."""
        p = self.pattern
        if p == 'circle': return 2.0 * self.R
        if p == 'figure8': return self.R * self.fig8_scale
        if p == 'waypoint': return math.hypot(5 * self.wp_k, 5 * self.wp_k)
        if p == 'scurve': return 2.0 * self.Rs * self.Na
        if p == 'aggressive': return 2.0 * self.Ra
        return 1.0
