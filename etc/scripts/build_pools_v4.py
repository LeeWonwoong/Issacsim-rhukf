#!/usr/bin/env python3
# build_pools_v4 — FROZEN-ENV v2 실측 CSV에서 surrogate v4 용 클래스별 (gyro,vel) obs 풀 추출.
#   소스: results_weak_overlap(δ0.1/0.2·ws8·5단계) + results_boundary_ws(δ0.6/0.7·ws7/8) + results_band_v2
import csv, numpy as np
from collections import defaultdict

pools=defaultdict(list)   # key -> [(g,v),...]

def add(path, dmap):
    for r in csv.DictReader(open(path)):
        try:
            s=int(r['step']); g=float(r['nis_g_scaled']); v=float(r['nis_v_scaled'])
        except: continue
        atk=r['attack_active']=='1'; pol=r['policy']; dl=dmap.get(r['bias'])
        if pol=='hover':
            if atk and dl: pools[f'hover_atk_{dl}'].append((g,v))
            elif s>=40: pools['hover_wind'].append((g,v))   # hover+바람(윈도우 후)
            else: pools['hover_clean'].append((g,v))
            continue
        # track
        if atk and dl:
            # 온셋(버스트 시작 3스텝)과 정상상태 분리 위해 delay 필요하나 CSV엔 없음 → 통합 풀(정상상태 지배)
            pools[f'atk_{dl}'].append((g,v))
        elif s<40 or (s>=240):     # weak_overlap 은 wind 60-180 창, boundary 는 40+ — 보수적으로 <40 만 clean
            pools['clean'].append((g,v))
        elif s>=60 and s<120:      # weak_overlap 의 wind-only 창
            pools['wind'].append((g,v))

# weak_overlap: WIND 60-180, ATK 120-240, δ0.1/0.2, ws8
add('results_weak_overlap/sweep_detail.csv', {'0.436':'d01','0.872':'d02'})
# boundary: WIND 40+, ATK 80+, δ0.6/0.7, ws7/8 — wind-only 창 = 40..80 (위 로직상 60-120 만 잡히므로 보정)
for r in csv.DictReader(open('results_boundary_ws/sweep_detail.csv')):
    try: s=int(r['step']); g=float(r['nis_g_scaled']); v=float(r['nis_v_scaled'])
    except: continue
    atk=r['attack_active']=='1'; dl={'2.616':'d06','3.052':'d07'}.get(r['bias'])
    if r['policy']=='hover':
        if atk and dl: pools[f'hover_atk_{dl}'].append((g,v))
        elif s>=40: pools['hover_wind'].append((g,v))
        else: pools['hover_clean'].append((g,v))
    else:
        if atk and dl: pools[f'atk_{dl}'].append((g,v))
        elif s<40: pools['clean'].append((g,v))
        elif 40<=s<80: pools['wind'].append((g,v))

out={k:np.array(vs,dtype=np.float32) for k,vs in pools.items()}
np.savez('etc/scripts/surr_pools_v4.npz', **out)
print('pool 크기:')
for k in sorted(out): print(f'  {k:16}: {len(out[k]):>6}  g_med={np.median(out[k][:,0]):.2f} v_med={np.median(out[k][:,1]):.2f}')
