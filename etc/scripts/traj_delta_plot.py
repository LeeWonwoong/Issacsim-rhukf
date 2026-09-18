# -*- coding: utf-8 -*-
"""δ 스윕 공격대응 궤적 — 패턴별 5(δ)×2(track/whover) 그리드 top-down."""
import os,csv,sys,numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
AUTH=4.36
D=np.load('results_traj_delta/zu_log.npz',allow_pickle=True)['data']
rs=list(np.where(D[:,1]==1)[0])+[len(D)]
segs=[np.arange(rs[i],rs[i+1]) for i in range(len(rs)-1)]
S=list(csv.DictReader(open('results_traj_delta/sweep_detail.csv')))
cells=[]
for c in sorted(set(int(r['cell_idx']) for r in S)):
    rows=[r for r in S if int(r['cell_idx'])==c]
    last=max(float(r['step']) for r in rows)
    cr=[r['crash_reason'] for r in rows if r['crash_reason'] not in ('','timeout')]
    cells.append(dict(b=float(rows[0]['bias']),pol=rows[0]['policy'],pat=rows[0]['pattern'],
                      last=last,crash=cr[0] if cr else None))
print(f"segs {len(segs)} / cells {len(cells)}")
# 명령 경로
kk=3.2/5.0; wps=np.array([[0.,0.],[5.,0.],[5.,5.],[-5.,5.],[-5.,0.],[0.,0.]])*kk
Rs,Na=1.1,4; L1=np.pi*Rs*Na
ss=np.linspace(0,L1,600); sc=[]
for s_ in ss:
    i=min(int(s_/(np.pi*Rs)),Na-1); ph=s_/Rs-i*np.pi; Cx=(2*i+1)*Rs
    th=(np.pi-ph) if i%2==0 else (np.pi+ph)
    sc.append((-(Cx+Rs*np.cos(th)),Rs*np.sin(th)))
sc=np.array(sc)
DELTAS=[0.436,0.872,1.744,2.616,3.488]
for pat,cmdpath,pat_lbl,fname in [('waypoint',wps,'waypoint (코너감속)','traj_delta_waypoint.png'),
                                  ('scurve',sc,'aggressive 신형 (3D S-curve, 수평투영)','traj_delta_aggressive.png')]:
    fig,axes=plt.subplots(2,5,figsize=(20,8.2),sharex=False,sharey=False)
    for col,b in enumerate(DELTAS):
        for row,pol in enumerate(['track','whover2']):
            A=axes[row,col]
            ci=[i for i,c in enumerate(cells) if c['pat']==pat and c['pol']==pol and abs(c['b']-b)<1e-6]
            if not ci or ci[0]>=len(segs): A.axis('off'); continue
            ci=ci[0]; meta=cells[ci]; sg=segs[ci]
            x=D[sg,4]; y=D[sg,5]; atk=D[sg,2]==1; act=D[sg,3]==1
            a=np.where(atk)[0]; s0,s1=(a[0],a[-1]) if len(a) else (len(x),len(x))
            A.plot(cmdpath[:,0],cmdpath[:,1],'k--',lw=1.0,alpha=.55)
            A.plot(x[:s0],y[:s0],c='#2ca02c',lw=1.2)
            A.plot(x[s0:s1+1],y[s0:s1+1],c='#d62728',lw=1.6)
            A.plot(x[s1+1:],y[s1+1:],c='#1f77b4',lw=1.2)
            if act.sum()>3: A.scatter(x[act][::3],y[act][::3],s=7,c='#9467bd',alpha=.5,zorder=5)
            r=np.hypot(x,y); rmax=r.max()
            ok = meta['last']>=399
            if not ok: A.scatter(x[-1],y[-1],marker='*',s=220,c='k',zorder=7)
            A.set_title(f"δ={b/AUTH:.1f} · {'생존' if ok else meta['crash'].replace('crash_','추락(')+')'} · 최대이탈 {rmax:.1f}m",
                        fontsize=9.5, color=('#1a5c37' if ok else '#8a1f14'))
            A.axis('equal'); A.grid(alpha=.25); A.tick_params(labelsize=7)
            if col==0: A.set_ylabel('track (무대응)' if row==0 else 'whover2 (호버대응)',fontsize=11,fontweight='bold')
    fig.suptitle(f'{pat_lbl} — 공격 δ 0.1~0.8 × 대응 유무 (공격창 15~25s · 무풍 · speed1)\n'
                 '초록=공격 전 · 빨강=공격 중 · 파랑=공격 후 · 보라점=호버 · ★=종료지점',fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig(fname,dpi=110)
    print('saved',fname)
# 요약표
print(f"\n{'pattern':>9s} {'δ':>4s} | {'track':>26s} | {'whover2':>26s}")
for pat in ['waypoint','scurve']:
    for b in DELTAS:
        line=f"{pat:>9s} {b/AUTH:4.1f}"
        for pol in ['track','whover2']:
            m=[c for c in cells if c['pat']==pat and c['pol']==pol and abs(c['b']-b)<1e-6]
            if m:
                c=m[0]; ci=cells.index(c); sg=segs[ci] if ci<len(segs) else None
                rmax=np.hypot(D[sg,4],D[sg,5]).max() if sg is not None else float('nan')
                st='생존' if c['last']>=399 else c['crash']
                line+=f" | {st:>14s} 이탈 {rmax:5.1f}m"
        print(line)
