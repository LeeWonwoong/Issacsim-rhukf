#!/usr/bin/env python3
"""train_pool_v3.npz — v6 풀 (FROZEN-v3 실데이터, 압축 NIS 스케일 0~3).
소스: ① results_v3_calwin_s42/train.log (TRAIN, 버스트 δ0.1~0.7, 행동 라벨)
     ② results_v3_band{_try2,}/sweep_detail.csv (지속 δ0.6~0.9, track/호버 정책)
풀 키: track_clean / hover_entry(dwell0~2) / hover_settled / post_track / post_hover
      / {track,hover}_atk_b0..7 (δ 0.1~0.9, 폭 0.1)"""
import re, csv, numpy as np, os
os.chdir('/home/acsl/projects/Issacsim-rhukf')
P = {k: [] for k in ['track_clean','hover_entry','hover_settled','post_track','post_hover']}
for i in range(8): P[f'track_atk_b{i}']=[]; P[f'hover_atk_b{i}']=[]
def binof(d): return min(7, max(0, int((d-0.1)/0.1)))
# ── ① v3 학습 로그
PAT=re.compile(r'\[\s*(\d+)\]\s+TRAIN\s+.(ATK|NRM)(?:\(τx([+-][\d.]+) τy([+-][\d.]+)\))?[^|]*?(TRACK|HOVER)\s+\|\s+ε=[\d.]+ \| NIS v=([\d.]+) g=([\d.]+)')
prev=-1; dwell=-1; since_end=99; prevh=False; preva=False
n1=0
for l in open('results_v3_calwin_s42/train.log',errors='replace'):
    m=PAT.search(l)
    if not m: continue
    stp=int(m.group(1)); a=m.group(2)=='ATK'; h=m.group(5)=='HOVER'
    v=float(m.group(6)); g=float(m.group(7))
    d=max(abs(float(m.group(3) or 0)),abs(float(m.group(4) or 0)))/4.36
    if stp<prev: dwell=-1; since_end=99; prevh=False; preva=False
    dwell=(dwell+1 if prevh else 0) if h else -1
    since_end=0 if a else (since_end+1 if preva or since_end<99 else 99)
    if a: since_end=0
    if a and d>=0.08:
        P[f'{"hover" if h else "track"}_atk_b{binof(d)}'].append((g,v))
    elif not a:
        if 1<=since_end<=2:
            P['post_hover' if h else 'post_track'].append((g,v))
        elif h and dwell<=2: P['hover_entry'].append((g,v))
        elif h: P['hover_settled'].append((g,v))
        else: P['track_clean'].append((g,v))
    prevh=h; preva=a; prev=stp; n1+=1
print(f'① v3 로그 {n1} 프레임')
# ── ② 밴드 스윕 디테일 (고δ 지속)
n2=0
for f in ['results_v3_band_try2/sweep_detail.csv','results_v3_band/sweep_detail.csv']:
    if not os.path.exists(f): continue
    for r in csv.DictReader(open(f)):
        if r['attack_active'] not in ('1','True'): continue
        d=float(r['bias'])/4.36
        if d<0.1 or d>0.92: continue
        h=r['action']=='1'
        g=float(r['nis_g_scaled']); v=float(r['nis_v_scaled'])
        P[f'{"hover" if h else "track"}_atk_b{binof(d)}'].append((g,v)); n2+=1
print(f'② 밴드 디테일 {n2} 공격프레임')
out={}
for k,rows in P.items():
    arr=np.array(rows) if rows else np.zeros((0,2))
    out[f'g_{k}']=np.sort(arr[:,0]) if len(arr) else np.array([0.3])
    out[f'v_{k}']=np.sort(arr[:,1]) if len(arr) else np.array([0.3])
    print(f'  {k:16s} n={len(arr):6d}  g중앙={np.median(arr[:,0]) if len(arr) else 0:.3f}')
np.savez('etc/scripts/train_pool_v3.npz', **out)
print('saved train_pool_v3.npz')
