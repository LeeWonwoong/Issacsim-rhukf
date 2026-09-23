#!/usr/bin/env python3
"""pick_R.py <런> [목표비] — SWIRL 의 R 을 필터 자기 일관성으로 보정.
   로그의 NIS=(res²/P_zz 평균) 과 loss(=E[res²]) 로  P_zz_total = loss/nisf,  P_zz_σ = P_zz_total − R.
   ★09-22 워크플로 검증본: D ≜ loss/NIS − R 은 UKF 스펙트럼 척도 λ̃ 의 대용(pΔ0.01 6런에서 D/λ̃ = 0.064~0.080, ±10%).
     R 에 대한 루프 이득 +0.243 이라 1회 보정으로 25%, 2회로 6% 수렴.  ⚠ pΔ 를 바꾸면 비가 2~4배 깨지므로 pΔ 고정에서만 쓴다.
     규칙:  R_new = R_base × (D_new / D_base),  기준 = v5·ui1·pΔ0.01·R0.5 (D_base = 0.48, F1 0.781).
     (구 규칙 R ∝ Zvar 는 폐기: Zvar 가 R^+1.11 로 따라가 루프 이득 >1 = 단조 폭주)
   출력 한 줄: "R*  P_zz_σ  현재비  nisf"  (설명은 stderr)"""
import sys, re, numpy as np, yaml, os
d = sys.argv[1]; rho = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # R_base/D_base (기본 0.5/0.48 = 1.04)
R = float(yaml.safe_load(open(f'{d}/config.yaml'))['agent']['swirl']['R'])
t = open(f'{d}/train_stdout.log').read(); epi = 0; rows = []
for l in t.split('\n'):
    if 'Pattern:' in l and 'Attack:' in l: epi += 1
    m = re.search(r'loss=([0-9.]+) Zvar=([0-9.]+).*?NIS=([0-9.]+)', l)
    if m and epi >= 150: rows.append([float(x) for x in m.groups()])
a = np.array(rows).mean(0); loss, nisf = a[0], a[2]
Pzz_s = max(loss / max(nisf, 1e-6) - R, 1e-3)
Rstar = float(np.clip(round(1.042 * rho * Pzz_s, 1), 0.1, 2.0))
print(f'{Rstar} {Pzz_s:.3f} {R / Pzz_s:.2f} {nisf:.3f}')
print(f'[pick_R] {os.path.basename(d)}: R {R} loss {loss:.3f} nisf {nisf:.3f} → P_zz_σ {Pzz_s:.3f}, 현재비 {R/Pzz_s:.2f} → R* = 1.042·{rho}·D = {Rstar}', file=sys.stderr)
