#!/usr/bin/env python3
"""agg_ext — 09-16 새벽 연장 QSET ext_G 판정표 + 그림 (night/EXT_RESULT.md, night/ext_gblk_curves.png)
   셀: gblk(돌풍+첫 노출 급변 150 ep; 시드 42–43 은 dual300 앞 150 ep), dual300(같은 무대 300 ep), mixk4(조건 고정 + 벌점 ×4, 60 ep), gmix(돌풍 + 조건 고정).
   학습기: SW(GPU) · UKF-TD · EKF-TD(P₀ 0.03) · Adam 3e-4 · Adam 1e-3(CPU, AMSGrad off). 판정 규칙(EXT_DECISION_0916.md, 사전 등록): n≥5 에서 부호 ≥4/5 ∧ |짝 t| ≥ 2.78 → 성립, 그 밖은 방향."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp_ in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp_): font_manager.fontManager.addfont(fp_); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp_).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})
N = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night'; Q = f'{N}/tonight/q'
LS = ['SW', 'UKFp03', 'EKFp03', 'Adam3e-4ams', 'Adam1e-3ams', 'Adam3e-4', 'Adam1e-3']   # 헤드라인 Adam 비교군 = AMSGrad on(사용자 09-16 09:10), off(라이브러리 기본값)도 같이 싣는다
NM = {'SW': 'SWIRL', 'UKFp03': 'UKF-TD', 'EKFp03': 'EKF-TD', 'Adam3e-4': 'Adam 3e-4 (off, 기본값)', 'Adam1e-3': 'Adam 1e-3 (off, 기본값)', 'Adam3e-4ams': 'Adam 3e-4 AMSGrad', 'Adam1e-3ams': 'Adam 1e-3 AMSGrad'}
COL = {'SW': '#2a6fdb', 'UKFp03': '#7a9a3a', 'EKFp03': '#9b6bb3', 'Adam3e-4': '#e0592a', 'Adam1e-3': '#b8860b', 'Adam3e-4ams': '#c0392b', 'Adam1e-3ams': '#8d6e00'}
H = {}
for f in sorted(glob.glob(f'{Q}/ext_G_*.json') + glob.glob(f'{Q}/ext_H_*.json') + glob.glob(f'{Q}/ext_K_*.json') + glob.glob(f'{Q}/ext_ams_*.json')):
    for k, m in json.load(open(f)).items(): H[(m['cell'][1:], m['learner'], m['seed'])] = m['hist']
for (c, l, s), h in list(H.items()):          # gblk 42–43 = dual300 앞 150 ep (GPU 학습기)
    if c == 'dual300' and ('gblk', l, s) not in H: H[('gblk', l, s)] = h[:150]
for (c, l, s), h in list(H.items()):          # 긴 지평 합산: dual300(42–43) + dual200(44–45) 을 150–199 창에서 같이 본다
    if c in ('dual300', 'dual200') and len(h) >= 200: H[('long200', l, s)] = h[:200]
def wm(h, key, lo, hi, atk=False):
    v = [float(x[key]) for x in h[lo:hi] if x.get(key) is not None and (not atk or x.get('has_atk'))]
    v = [x for x in v if not np.isnan(x)]; return float(np.mean(v)) if v else np.nan
def pt(d):
    d = np.array([x for x in d if not np.isnan(x)], float); n = len(d)
    if n < 2: return n, (float(d.mean()) if n else np.nan), np.nan, d
    sd = d.std(ddof=1); return n, float(d.mean()), (float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan), d
def verdict(n, t, good):
    if n < 5: return '방향' if n else '—'
    return '**성립**' if (good >= 4 and not np.isnan(t) and abs(t) >= 2.776) else '미성립'
L = ['# 새벽 연장 ext_G 결과 (자동 생성 `etc/scripts/agg_ext.py`)', '',
     '- 새 풀 v5e · γ0.9·n3 · 버퍼 50k · 시드별 에피소드 RNG 고정. 괄호 = 짝 평균 (t, 앞 학습기가 나은 시드 수/n, 판정). 판정: n≥5 에서 ≥4/5 ∧ |t|≥2.78 → 성립.',
     '- 헤드라인 Adam 비교군은 **AMSGrad on**(사용자 결정). 괄호 off 는 PyTorch 기본값이며 같은 표에 싣는다.\n- gblk 시드 42–43 의 SW·UKF·EKF 는 dual300 런 앞 150 ep(같은 무대·ε 스텝 기준).', '']
def table(cell, rows, pairs):
    seeds = sorted({s for (c, l, s) in H if c == cell})
    if not seeds: L.extend([f'(자료 없음)', '']); return
    L.append(f'시드: {seeds} · 완료 런 ' + ', '.join(f"{NM[l]} {sum(1 for s in seeds if (cell, l, s) in H)}" for l in LS)); L.append('')
    L.append('| 지표 | ' + ' | '.join(NM[l] for l in LS) + ' | ' + ' | '.join(f'{NM[a]} − {NM[b]}' for a, b in pairs) + ' |')
    L.append('|' + '---|' * (1 + len(LS) + len(pairs)))
    for label, fn, hb in rows:
        vals = {l: [fn(H[(cell, l, s)]) if (cell, l, s) in H else np.nan for s in seeds] for l in LS}
        cells = [f"{np.nanmean(vals[l]):.3f}" if np.any(~np.isnan(vals[l])) else '—' for l in LS]
        ds = []
        for a_, b_ in pairs:
            n, mu, t, d = pt([x - y for x, y in zip(vals[a_], vals[b_])])
            good = int((d > 0).sum()) if hb else int((d < 0).sum())
            ds.append(f"{mu:+.3f} (t {t:+.2f}, {good}/{n}, {verdict(n, t, good)})" if n else '—')
        L.append(f'| {label} | ' + ' | '.join(cells) + ' | ' + ' | '.join(ds) + ' |')
    L.append('')
P = [('SW', 'UKFp03'), ('SW', 'EKFp03'), ('SW', 'Adam3e-4ams'), ('SW', 'Adam1e-3ams'), ('SW', 'Adam3e-4'), ('UKFp03', 'Adam3e-4ams'), ('EKFp03', 'Adam3e-4ams')]
L += ['## 1. gblk — 돌풍 + 첫 노출 급변 (조건 전환 + 이상치)', '']
table('gblk', [('진입 오탐 60–69', lambda h: wm(h, 'fpr', 60, 70), False), ('진입 보상 60–69', lambda h: wm(h, 'reward', 60, 70), True),
               ('블록 보상 60–109', lambda h: wm(h, 'reward', 60, 110), True), ('블록 돌풍 오탐 60–109', lambda h: wm(h, 'fpr_gust', 60, 110), False),
               ('후기 돌풍 오탐 110–149', lambda h: wm(h, 'fpr_gust', 110, 150), False), ('후기 F1 110–149 (공격 에피)', lambda h: wm(h, 'f1', 110, 150, True), True),
               ('후기 보상 110–149', lambda h: wm(h, 'reward', 110, 150), True), ('블록 전 보상 30–59', lambda h: wm(h, 'reward', 30, 60), True)], P)
L += ['## 2. dual300 — 같은 무대 300 ep (과정모형 오지정 → 칼만-TD 공분산 누적)', '']
table('dual300', [('후반 F1 250–299 (공격 에피)', lambda h: wm(h, 'f1', 250, 300, True), True), ('후반 오탐 250–299', lambda h: wm(h, 'fpr', 250, 300), False),
                  ('후반 보상 250–299', lambda h: wm(h, 'reward', 250, 300), True), ('150–299 보상', lambda h: wm(h, 'reward', 150, 300), True),
                  ('후반 pmax 250–299', lambda h: wm(h, 'pmax', 250, 300), False), ('후반 돌풍 오탐 250–299', lambda h: wm(h, 'fpr_gust', 250, 300), False)], P)
L += ['## 2b. 긴 지평 150–199 합산 (dual300 시드 42–43 + dual200 시드 44–45, 같은 무대)', '']
table('long200', [('F1 150–199 (공격 에피)', lambda h: wm(h, 'f1', 150, 200, True), True), ('오탐 150–199', lambda h: wm(h, 'fpr', 150, 200), False),
                  ('보상 150–199', lambda h: wm(h, 'reward', 150, 200), True), ('돌풍 오탐 150–199', lambda h: wm(h, 'fpr_gust', 150, 200), False),
                  ('pmax 150–199', lambda h: wm(h, 'pmax', 150, 200), False)], P)
L += ['## 3. mixk4 — 조건 고정 + FP·FN 벌점 ×4, 60 ep (초기조건·콜드 스타트)', '']
table('mixk4', [('초기 보상 0–29', lambda h: wm(h, 'reward', 0, 30), True), ('초기 오탐 0–29', lambda h: wm(h, 'fpr', 0, 30), False),
                ('후기 보상 30–59', lambda h: wm(h, 'reward', 30, 60), True), ('전 구간 보상 0–59', lambda h: wm(h, 'reward', 0, 60), True)], P)
L += ['## 3b. gmixk4 — 돌풍 + 벌점 ×4, 60 ep (요구 순서 SW ≫ 칼만-TD ≫ Adam 겨냥 무대)', '']
table('gmixk4', [('초기 보상 0–29', lambda h: wm(h, 'reward', 0, 30), True), ('초기 오탐 0–29', lambda h: wm(h, 'fpr', 0, 30), False),
                 ('초기 돌풍 오탐 0–29', lambda h: wm(h, 'fpr_gust', 0, 30), False), ('후기 보상 30–59', lambda h: wm(h, 'reward', 30, 60), True),
                 ('후기 F1 30–59 (공격 에피)', lambda h: wm(h, 'f1', 30, 60, True), True), ('후기 돌풍 오탐 30–59', lambda h: wm(h, 'fpr_gust', 30, 60), False),
                 ('전 구간 보상 0–59', lambda h: wm(h, 'reward', 0, 60), True)], P)
L += ['## 4. gmix — 돌풍 + 조건 고정 (gblk 의 조건 전환 끔 대조)', '']
table('gmix', [('후기 돌풍 오탐 110–149', lambda h: wm(h, 'fpr_gust', 110, 150), False), ('후기 F1 110–149 (공격 에피)', lambda h: wm(h, 'f1', 110, 150, True), True),
               ('후기 보상 110–149', lambda h: wm(h, 'reward', 110, 150), True), ('전 구간 보상', lambda h: wm(h, 'reward', 0, 150), True)], P)
L += ['## 5. 가치함수 몫 실험 (per5 무대, 09-16 오후) — 확약 길이·n-step', '',
      '- per5(D5·n3) = 기존 확정 설정. per5_D1 = 확약 제거(강제 스텝 0). per5_n1 = n-step 끄기. per5_D10 = 확약 10스텝. per5_D10n1 = 둘 다.',
      '- **D 비교 주의**: ε-탐험이 호버를 뽑으면 D−1 스텝이 강제되고 전부 오탐으로 잡힌다. 그래서 원시 오탐이 아니라 `fpr_chosen`(정책이 고른 스텝만)과 `reward_forced`(강제 스텝의 보상 몫)로 분해해 본다. 09-16 11:10 이전 런에는 이 필드가 없어 "—" 로 나온다.',
      '- 사전 등록 결정 규칙: per5_D1 에서 SW − Adam3e-4(AMSGrad on) 보상 30–149 격차가 ≥ +0.5 이고 부호 ≥4/5 → 확약 제거(D=1) 채택. 격차 < +0.47 이거나 부호 ≤3/5 → 확약 D5 유지 + 회계 수정.', '']
for _c, _lab in (('per5', 'per5 (확약 5·n3, 기존)'), ('per5_D1', 'per5_D1 (확약 없음)'), ('per5_n1', 'per5_n1 (n-step 끔)'), ('per5_D10', 'per5_D10 (확약 10)'), ('per5_D10n1', 'per5_D10n1 (확약 10 + n-step 끔)')):
    _sd = sorted({s_ for (c_, l_, s_) in H if c_ == _c})
    if not _sd: continue
    L.append(f'### {_lab} — 시드 {_sd}'); L.append('')
    table(_c, [('보상 30–149', lambda h: wm(h, 'reward', 30, 150), True), ('오탐 30–149 (원시)', lambda h: wm(h, 'fpr', 30, 150), False),
               ('오탐 30–149 (고른 스텝만)', lambda h: wm(h, 'fpr_chosen', 30, 150), False), ('강제 스텝 보상 몫 30–149', lambda h: wm(h, 'reward_forced', 30, 150), True),
               ('후기 F1 110–149 (공격 에피)', lambda h: wm(h, 'f1', 110, 150, True), True), ('탐지 지연 30–149', lambda h: wm(h, 'delay', 30, 150), False)], P)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for i, (cell, n_ep) in enumerate((('gblk', 150), ('dual300', 300))):
    for l in LS:
        hs = [H[k] for k in H if k[0] == cell and k[1] == l and len(H[k]) >= n_ep]
        if not hs: continue
        r = np.nanmean([[x['reward'] for x in h[:n_ep]] for h in hs], 0); k = 5
        ax[i].plot(np.arange(n_ep - k + 1) + k - 1, np.convolve(r, np.ones(k) / k, 'valid'), color=COL[l], lw=2, label=f'{NM[l]} (n={len(hs)})')
    ax[i].axvspan(60, 110, color='#888', alpha=0.12, lw=0); ax[i].set_xlabel('에피소드'); ax[i].set_ylabel('보상 (5 ep 이동평균)')
    ax[i].set_title({'gblk': '돌풍 + 첫 노출 급변 (회색 = 강풍 블록 60–109)', 'dual300': '같은 무대 300 ep'}[cell]); ax[i].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f'{N}/ext_gblk_curves.png', dpi=130); plt.close(fig)
L += ['그림: `night/ext_gblk_curves.png` · 완료 ext_G 런 ' + str(len(glob.glob(f'{Q}/ext_G_*.json')))]
open(f'{N}/EXT_RESULT.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
