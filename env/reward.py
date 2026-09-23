"""env/reward.py — 보상 정의의 단일 출처 (Isaac·surrogate 공용).

환경은 (공격 참값, 공격 경과 스텝, 추락 여부)만 알려준다. 보상은 오직 여기서 계산한다.
라벨 규약(Isaac 과 동일): 스텝 t 의 보상은 **직전에 실행된 행동 a_{t−1}** 과 현재 공격 상태의 조합.

모드 (YAML reward.mode)
  label4 : 4항 per-step 라벨 (2026-08-19 v3 ~ 09-18 격자 기준)
             평시 track +r_tn / hover r_fp,  공격 hover +r_tp / track FN(d)
             FN(d) = (fn_base + fn_per_step·min(d, delay_cap)) × (fn_onset_mult if 1≤d≤fn_onset_window)
  cost   : 탐지기형 (2026-09-18 확정) — 오경보 비용 · 탐지 지연 비용 · 탐지 사건 보상
             r = alive − c_fa·1{평시 hover} − c_d·1{공격 중 track} + bonus·1{사건별 첫 hover}
             · 지연 비용은 놓친 스텝마다 상수 → 누적 비용 = c_d × 탐지 지연 (고전 QCD 지연 비용)
             · bonus: 사건(버스트)마다 처음 hover 가 켜진 스텝 1회(이미 hover 중에 온셋이어도 지급) — 모든 공격을 잡는 감지기
             · alive: 모든 스텝 공통 상수(정책 불변, 종료 시에만 효과 = 추락 비용)
공통: 추락(terminated)이면 −terminal_penalty 추가.  마지막에 전체 × scale.

★09-23 P2′ 자세 퍼텐셜 성형 (shape_tilt=λ>0 일 때만; 기본 0 = 끔·비트 동일)
  r_t = r^G_t + F_t,  F_t = γ·Φ_t − Φ_{t−1},  Φ_t = −λ·θ̂_t/θ0 (θ0 = tilt_ref),  θ̂ = √(roll²+pitch²)
  · 동결: prev_action 이 바뀐 행부터 shape_freeze 행은 θ̂ = 직전 행의 유효 θ̂ (명령 과도 제거)
  · 시작: reset() 뒤 첫 호출 행은 Φ_prev 를 그 행 Φ 로 초기화하고 F=0 (그 행은 전이를 만들지 않는다 — prev_state 없음)
  · 종료: terminated 면 흡수상태 Φ=0 → F = −Φ_{T−1}.  timeout(절단)은 정상 계산
  · γ = shape_gamma (cfgload 가 agent.gamma 로 채우고 다르면 거부) → n-step 합이 γⁿΦ_{t+n} − Φ_t 로 망원
  · 정책 불변(Ng·Harada·Russell 1999). 지표·보고는 r^G 만(last_rG) — 반환값은 학습 보상(r^G+F)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardConfig:
    mode: str = 'label4'           # label4 | cost
    scale: float = 1.0
    terminal_penalty: float = 0.0  # 추락 시 빼는 크기(≥0)
    # ── label4 ──
    r_tp: float = 1.0
    r_tn: float = 0.5
    r_fp: float = -0.7
    fn_base: float = -0.4
    fn_per_step: float = -0.35
    fn_onset_mult: float = 1.8
    fn_onset_window: int = 5
    delay_cap: int = 5
    relapse_penalty: float = 0.0   # 공격 중 hover→track 재발 스텝 추가 벌점(음수로 적는다)
    fp_escalate: bool = False      # 평시 연속 오탐 에스컬레이션(첫 −0.2 … −1.2)
    # ── cost ──
    c_fa: float = 0.7
    c_d: float = 0.3
    bonus: float = 1.0
    alive: float = 0.0
    # ── ★09-23 P2′ 자세 퍼텐셜 성형 (0 = 끔) ──
    shape_tilt: float = 0.0        # λ
    tilt_ref: float = 0.1          # θ0 [rad]
    shape_freeze: int = 3          # 모드 전환 행부터 θ̂ 동결 행 수
    shape_gamma: float = 0.0       # F 의 γ. 0 = agent.gamma 를 cfgload 가 채움(다르면 오류)

    def __post_init__(self):
        if self.mode not in ('label4', 'cost'):
            raise ValueError(f'reward.mode={self.mode!r} (label4|cost)')
        self.terminal_penalty = abs(float(self.terminal_penalty))   # 옛 설정은 음수로 적었다 → 크기로 정규화
        if float(self.shape_tilt) < 0 or float(self.tilt_ref) <= 0 or int(self.shape_freeze) < 0:
            raise ValueError(f'reward.shape_tilt={self.shape_tilt} (≥0) · tilt_ref={self.tilt_ref} (>0) · '
                             f'shape_freeze={self.shape_freeze} (≥0)')


def label4_reward(rc: RewardConfig, prev_action: int, attack: bool, attack_delay: int,
                  fp_run: int = 0, relapse: bool = False) -> float:
    """4항 라벨 보상 (배율 적용 전). 구 calculate_reward 와 수치 동일(FN_MODE·FP_ESCAL env 는 폐기 → 설정)."""
    if attack:
        if prev_action == 1:
            return rc.r_tp
        d = min(max(attack_delay, 0), rc.delay_cap)
        pen = rc.fn_base + rc.fn_per_step * d
        if 1 <= d <= rc.fn_onset_window:
            pen *= rc.fn_onset_mult
        if relapse:
            pen += rc.relapse_penalty
        return pen
    if prev_action == 0:
        return rc.r_tn
    if rc.fp_escalate:
        return -0.2 - 0.25 * max(0, min(fp_run, 5) - 1)
    return rc.r_fp


class RewardTracker:
    """에피소드 단위 상태(버스트 첫 hover 지급 여부·연속 오탐·재발)를 들고 스텝 보상(배율 적용)을 낸다.
    호출 규약: 창이 찬 스텝마다 1회, prev_action = 이 스텝까지 실행되던 행동."""

    def __init__(self, rc: RewardConfig):
        self.rc = rc
        self.reset()

    def reset(self):
        self._prev_atk = False
        self._paid = False
        self._pprev_action = 0
        self._fp_run = 0
        # P2′ 성형 상태 (에피소드 시작: Φ_prev 미정 → 첫 호출 행에서 초기화)
        self._phi_prev = None
        self._th_prev = None
        self._frz_left = 0
        self._frz_val = 0.0
        self.last_rG = 0.0; self.last_F = 0.0; self.last_phi = 0.0; self.last_theta = float('nan')

    @property
    def shaping(self) -> bool:
        return float(self.rc.shape_tilt) > 0.0

    def _shape(self, prev_action: int, terminated: bool, roll, pitch) -> float:
        """F_t (배율 전). 호출 전 self._pprev_action = 직전 행의 prev_action."""
        rc = self.rc
        if roll is None:
            raise ValueError('reward.shape_tilt>0 인데 roll/pitch(θ) 입력이 없다 — Isaac 은 cur_euler, surrogate 는 env.surrogate.theta=true')
        th = math.hypot(float(roll), float(pitch or 0.0))
        if (self._th_prev is not None and prev_action != self._pprev_action and int(rc.shape_freeze) > 0):
            self._frz_left = int(rc.shape_freeze); self._frz_val = self._th_prev    # 전환 행 s: θ̂_{s−1}(유효값) 로 동결 시작
        if self._frz_left > 0:
            th_eff = self._frz_val; self._frz_left -= 1
        else:
            th_eff = th
        phi = 0.0 if terminated else -float(rc.shape_tilt) * th_eff / float(rc.tilt_ref)
        g = float(rc.shape_gamma)
        if not 0.0 < g <= 1.0:
            raise ValueError(f'reward.shape_gamma={g} — cfgload 가 agent.gamma 로 채운다(직접 쓸 때는 γ 를 넣을 것)')
        F = 0.0 if self._phi_prev is None else g * phi - self._phi_prev
        self._phi_prev = phi; self._th_prev = th_eff
        self.last_phi = phi; self.last_theta = th_eff
        return F

    def step(self, prev_action: int, attack: bool, attack_delay: int, terminated: bool = False,
             roll=None, pitch=None) -> float:
        """반환 = 학습 보상(배율 적용). 성형 끔이면 r^G 그대로(구 코드와 비트 동일). 성분은 last_rG·last_F."""
        rc = self.rc
        if attack and not self._prev_atk:
            self._paid = False                     # 새 버스트
        relapse = bool(attack and prev_action == 0 and self._pprev_action == 1)
        self._fp_run = self._fp_run + 1 if (not attack and prev_action == 1) else 0

        if rc.mode == 'label4':
            r = label4_reward(rc, prev_action, attack, attack_delay, self._fp_run, relapse)
        else:
            r = rc.alive
            if not attack:
                if prev_action == 1:
                    r -= rc.c_fa
            elif prev_action == 0:
                r -= rc.c_d
            elif not self._paid:
                r += rc.bonus
                self._paid = True
        if terminated:
            r -= rc.terminal_penalty
        self._prev_atk = attack
        if not self.shaping:
            self._pprev_action = prev_action
            out = r * rc.scale
            self.last_rG = out
            return out
        F = self._shape(prev_action, terminated, roll, pitch)
        self._pprev_action = prev_action
        self.last_rG = r * rc.scale
        self.last_F = F * rc.scale
        return self.last_rG + self.last_F


# ── 구 호출부 호환 (분석 스크립트용; 새 코드는 RewardTracker) ─────────────────────
DEFAULT_REWARD = RewardConfig()


def calculate_reward(current_action, is_under_attack, attack_delay: int = 0, fp_run: int = 0,
                     rc: RewardConfig = None, relapse: bool = False) -> float:
    return label4_reward(rc or DEFAULT_REWARD, current_action, is_under_attack, attack_delay, fp_run, relapse)
