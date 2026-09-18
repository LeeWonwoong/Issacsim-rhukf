"""env/reward.py — 보상 정의의 단일 출처 (Isaac·surrogate 공용).

환경은 (공격 참값, 공격 경과 스텝, 추락 여부)만 알려준다. 보상은 오직 여기서 계산한다.
라벨 규약(Isaac 과 동일): 스텝 t 의 보상은 **직전에 실행된 행동 a_{t−1}** 과 현재 공격 상태의 조합.

모드 (YAML reward.mode)
  label4 : 4항 per-step 라벨 (2026-08-19 v3 ~ 09-18 격자 기준)
             평시 track +r_tn / hover r_fp,  공격 hover +r_tp / track FN(d)
             FN(d) = (fn_base + fn_per_step·min(d, delay_cap)) × (fn_onset_mult if 1≤d≤fn_onset_window)
  cost   : 탐지 사건형 (2026-09-18 논의) — 오경보 비용 vs 탐지 지연 비용의 경쟁, TP 반복 보상 없음
             평시 track 0 / hover −c_fa,  공격 track −c_d·g(d) / 버스트 첫 hover +bonus / hover 유지 0
             g(d) = 분기 뒤 누적 가중(esc_*; 기본 끔 = 1)
             + alive(모든 스텝 공통 상수: 정책 불변, 종료 시에만 효과 = 추락 벌점 alive/(1−γ) 와 동치)
             버스트 첫 hover: 버스트 중 처음 hover 가 켜진 스텝(이미 hover 중에 온셋이어도 지급), 버스트당 1회.
공통: 추락(terminated)이면 −terminal_penalty 추가.  마지막에 전체 × scale.
"""
from __future__ import annotations

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
    # 분기 뒤 누적 지연 비용: r_FN = −c_d·g(d),  g(d) = 1 (d ≤ esc_after),  min(1 + (d − esc_after)/esc_tau, esc_max) (이후)
    #   d = 이번 공격 사건 온셋부터 경과 스텝. esc_tau = 0 이면 끔(상수 c_d).
    esc_after: int = 3
    esc_tau: float = 0.0
    esc_max: float = 4.0

    def __post_init__(self):
        if self.mode not in ('label4', 'cost'):
            raise ValueError(f'reward.mode={self.mode!r} (label4|cost)')
        self.terminal_penalty = abs(float(self.terminal_penalty))   # 옛 설정은 음수로 적었다 → 크기로 정규화


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

    def step(self, prev_action: int, attack: bool, attack_delay: int, terminated: bool = False) -> float:
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
                g = 1.0 if rc.esc_tau <= 0 else min(1.0 + max(0, attack_delay - rc.esc_after) / rc.esc_tau, rc.esc_max)
                r -= rc.c_d * g
            elif not self._paid:
                r += rc.bonus
                self._paid = True
        if terminated:
            r -= rc.terminal_penalty
        self._prev_atk = attack
        self._pprev_action = prev_action
        return r * rc.scale


# ── 구 호출부 호환 (분석 스크립트용; 새 코드는 RewardTracker) ─────────────────────
DEFAULT_REWARD = RewardConfig()


def calculate_reward(current_action, is_under_attack, attack_delay: int = 0, fp_run: int = 0,
                     rc: RewardConfig = None, relapse: bool = False) -> float:
    return label4_reward(rc or DEFAULT_REWARD, current_action, is_under_attack, attack_delay, fp_run, relapse)
