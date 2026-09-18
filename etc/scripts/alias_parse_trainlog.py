"""tune 배치 train.log 파싱 → 확정 env 하에서 에이전트가 실제로 본 관측 분포.
   per-step: nis_v, nis_g(압축), raw, ATK/NRM, TRACK/HOVER, δ, pattern, wind, phase(TRAIN/EVAL)"""
import re, numpy as np
RE_EP  = re.compile(r'^\s+(TRAIN|EVAL) Ep (\d+)/')
RE_HDR = re.compile(r'^\s+Pattern: (\w+) \| Attack: (\w+) \(int=([\d.]+), start=(\d+)\) \| Wind: (\S+) \(([\d.]+) m/s\)')
RE_STP = re.compile(r'\[\s*(\d+)\] (TRAIN|EVAL) (?:🔴ATK\(τx([+-][\d.]+) τy([+-][\d.]+)\)|⚪NRM) (TRACK|HOVER) '
                    r'\| ε=([\d.]+) \| NIS v=([\d.]+) g=([\d.]+) \(raw v=([\d.]+) g=([\d.]+)\)')
PATS={'circle':0,'figure8':1,'waypoint':2,'aggressive':3}
AUTH=4.36   # roll 제어권한 [N·m] — δ = |τ|/AUTH

def parse(path, max_lines=None):
    ep=-1; pat=-1; ws=0.0; rows=[]
    with open(path, errors='ignore') as f:
        for i,ln in enumerate(f):
            if max_lines and i>max_lines: break
            m=RE_EP.match(ln)
            if m: ep=int(m.group(2)); continue
            m=RE_HDR.match(ln)
            if m: pat=PATS.get(m.group(1),-1); ws=float(m.group(6)); continue
            m=RE_STP.search(ln)
            if m:
                tx,ty=m.group(3),m.group(4)
                atk = 1 if tx is not None else 0
                d = (np.hypot(float(tx),float(ty))/AUTH) if atk else 0.0
                rows.append((ep, int(m.group(1)), 1 if m.group(2)=='EVAL' else 0, atk, d,
                             1 if m.group(5)=='TRACK' else 0, float(m.group(6)),
                             float(m.group(7)), float(m.group(8)),
                             float(m.group(9)), float(m.group(10)), pat, ws))
    A=np.array(rows)
    return dict(ep=A[:,0].astype(int), step=A[:,1].astype(int), is_eval=A[:,2].astype(bool),
                atk=A[:,3].astype(int), delta=A[:,4], track=A[:,5].astype(bool), eps=A[:,6],
                vel=A[:,7], gyro=A[:,8], vel_raw=A[:,9], gyro_raw=A[:,10],
                pat=A[:,11].astype(int), ws=A[:,12])
