#!/usr/bin/env python3
"""pick_best — 튜닝 로그에서 최고 보상 조합의 env 문자열을 뽑는다."""
import re, sys
log, p0 = sys.argv[1], sys.argv[2]
best = (-1e9, f'SW_P={p0}')
for line in open(log, errors='ignore'):
    m = re.match(r'\s+(\w[\w.]*)\s+:\s+SWIRL\s+후반100 보상\s+(-?[\d.]+)', line)
    if not m: continue
    tag, val = m.group(1), float(m.group(2))
    env = f'SW_P={p0}'
    if tag.startswith('R'):   env += f' SW_R={tag[1:]}'
    elif tag.startswith('N'): env += f' SW_N_OVR={tag[1:]}'
    elif tag.startswith('A'): env += f' SW_A={tag[1:]}'
    if val > best[0]: best = (val, env)
print(best[1])
