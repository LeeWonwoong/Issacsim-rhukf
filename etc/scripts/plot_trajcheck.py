#!/usr/bin/env python3
"""plot_trajcheck — sweep_detail.csv(ref_x/ref_y/ref_alt 포함) 로 패턴별 reference vs 실제 궤적 + 추종 RMSE.
사용: python3 etc/scripts/plot_trajcheck.py results_trajcheck [out.png]"""
import sys, os, csv, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
d=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else os.path.join(d,'trajcheck.png')
ep=collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(d,'sweep_detail.csv'))):
    if r['policy']!='track' or 'ref_x' not in r: continue
    ep[(r['pattern'],r['disturbance_type'],r['wind_speed'],r['cell_idx'],r['episode'])].append(r)
pats=['waypoint','circle','figure8','aggressive','scurve']; conds=sorted(set((k[1],k[2]) for k in ep))
fig,ax=plt.subplots(len(conds),len(pats),figsize=(4*len(pats),4*len(conds)),squeeze=False)
rows=[]
for ci,(dt,ws) in enumerate(conds):
    for pi,p in enumerate(pats):
        a=ax[ci][pi]; first=True
        for k,rs in ep.items():
            if k[0]!=p or (k[1],k[2])!=(dt,ws): continue
            rs.sort(key=lambda r:int(r['step']))
            st=np.array([int(r['step']) for r in rs]); m=st>=20
            px=np.array([float(r['pos_x']) for r in rs])[m]; py=np.array([float(r['pos_y']) for r in rs])[m]; pz=np.array([float(r['alt']) for r in rs])[m]
            rx=np.array([float(r['ref_x']) for r in rs])[m]; ry=np.array([float(r['ref_y']) for r in rs])[m]; rz=np.array([float(r['ref_alt']) for r in rs])[m]
            ok=~np.isnan(rx)
            a.plot(rx[ok],ry[ok],'k--',lw=1,label='reference' if first else None); a.plot(px,py,'-',lw=1.2,label=f'actual ep{k[4]}')
            e=np.hypot(px[ok]-rx[ok],py[ok]-ry[ok]); ez=np.abs(pz[ok]-rz[ok])
            rows.append((p,dt,ws,k[4],np.sqrt(np.mean(e**2)),np.max(e),np.sqrt(np.mean(ez**2)),np.max(ez),np.mean(np.hypot(*(np.diff(np.c_[px,py],axis=0).T))*10)))
            first=False
        a.set_title(f'{p} · {dt} {ws}'); a.set_aspect('equal'); a.grid(alpha=.3); a.legend(fontsize=7)
fig.suptitle('무공격 reference(점선) vs 실제 궤적 (NED x/y)'); fig.tight_layout(); fig.savefig(out,dpi=110)
print(f"{'pattern':11s} {'dist':16s} {'ws':>4s} ep  {'XY RMSE':>8s} {'XY max':>7s} {'Z RMSE':>7s} {'Z max':>6s} {'mean|v|':>8s}")
for r in sorted(rows): print(f"{r[0]:11s} {r[1]:16s} {r[2]:>4s} {r[3]:>2s}  {r[4]:8.2f} {r[5]:7.2f} {r[6]:7.2f} {r[7]:6.2f} {r[8]:8.2f}")
print('그림:',out)
