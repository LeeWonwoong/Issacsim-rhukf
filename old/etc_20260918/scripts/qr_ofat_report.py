# -*- coding: utf-8 -*-
"""qr_ofat_sweep.py 결과 판독 + Pareto 플롯.
목표 4개: ①고집(sustain≈1) ②온셋 빠름(τon↓) ③오프셋 빠름(τoff↓) ④비자명(d′ 과도하지 않음)"""
import json,sys,os,numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
R=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'scratchpad/qr_ofat.json'))
base=[r for r in R if r['knob']=='BASE'][0]['m']
KN=['q_pos','q_eul','q_vel','q_gyr','r_pos','r_vel','r_gyr']
LBL={'q_pos':'Q_pos','q_eul':'Q_euler','q_vel':'Q_vel','q_gyr':'Q_gyro','r_pos':'R_pos','r_vel':'R_vel','r_gyr':'R_gyro'}
BASEV={'q_pos':1e-3,'q_eul':5e-4,'q_vel':5e-3,'q_gyr':5e-3,'r_pos':0.5,'r_vel':0.1,'r_gyr':0.2}
def rows(k): return sorted([r for r in R if r['knob']==k], key=lambda r: float(r['val']))
print('='*118)
print(' Q·R 전 노브 OFAT — 기준(BASE): ' + ' '.join(f'{LBL[k]}={BASEV[k]:g}' for k in KN))
b=base['gyro']; bv=base['vel']
print(f"  BASE  gyro: 바닥 {b['base']:.2f} 고원 {b['plateau']:.2f} τon {b['tau_on']:.0f} τoff {b['tau_off']:.0f} 고집 {b['sustain']:.2f} d′ {b['dprime']:.2f} OVL {b['ovl']:.3f}")
print(f"        vel : 바닥 {bv['base']:.2f} 고원 {bv['plateau']:.2f} τon {bv['tau_on']:.0f} τoff {bv['tau_off']:.0f} 고집 {bv['sustain']:.2f} d′ {bv['dprime']:.2f} OVL {bv['ovl']:.3f}")
print('='*118)
for k in KN:
    rr=rows(k)
    if not rr: continue
    print(f"\n── {LBL[k]}  (기준 {BASEV[k]:g}) ──")
    print(f"  {'값':>9s} | {'g바닥':>6s} {'g고원':>6s} {'gτon':>5s} {'gτoff':>6s} {'g고집':>5s} {'g d′':>6s} {'gOVL':>6s} | {'v바닥':>6s} {'vτoff':>6s} {'v d′':>6s} {'vOVL':>6s}")
    for r in rr:
        g=r['m']['gyro']; v=r['m']['vel']
        print(f"  {float(r['val']):9g} | {g['base']:6.2f} {g['plateau']:6.2f} {g['tau_on']:5.0f} {g['tau_off']:6.0f} {g['sustain']:5.2f} {g['dprime']:6.2f} {g['ovl']:6.3f} "
              f"| {v['base']:6.2f} {v['tau_off']:6.0f} {v['dprime']:6.2f} {v['ovl']:6.3f}")
# ── 플롯 ──
fig,ax=plt.subplots(2,3,figsize=(17,9))
metrics=[('dprime','d′ (분리도) — 낮을수록 비자명',0),('tau_off','τ_off [스텝] — 낮을수록 좋음',1),
         ('sustain','고집 sustain — 1에 가까울수록 좋음',2),('base','평시 바닥 NIS',3),
         ('tau_on','τ_on [스텝] — 낮을수록 좋음',4),('ovl','OVL 겹침 — 높을수록 비자명',5)]
cols=dict(zip(KN,['#999','#8c564b','#1f77b4','#d62728','#bbb','#2ca02c','#ff7f0e']))
for mk,ttl,i in metrics:
    A=ax[i//3,i%3]
    for k in KN:
        rr=rows(k)
        if len(rr)<2: continue
        x=[float(r['val'])/BASEV[k] for r in rr]; y=[r['m']['gyro'][mk] for r in rr]
        A.plot(x,y,'o-',c=cols[k],lw=1.8,ms=4,label=LBL[k])
    A.axhline(base['gyro'][mk],c='k',ls=':',lw=1.2)
    A.axvline(1.0,c='k',ls=':',lw=.8,alpha=.5)
    A.set_xscale('log'); A.set_title(ttl,fontsize=11); A.set_xlabel('기준 대비 배율'); A.grid(alpha=.25)
    if i==0: A.legend(fontsize=8,ncol=2)
fig.suptitle('Q·R 전 노브 OFAT — gyro 채널 (기준선=현행 config)',fontsize=13)
fig.tight_layout(); fig.savefig('qr_ofat_knobs.png',dpi=110); plt.close(fig)
# Pareto: d′ vs τoff
fig,ax=plt.subplots(1,2,figsize=(14,5.6))
for ch,A,ttl in [('gyro',ax[0],'gyro'),('vel',ax[1],'vel')]:
    for k in KN:
        rr=rows(k)
        if len(rr)<2: continue
        x=[r['m'][ch]['dprime'] for r in rr]; y=[r['m'][ch]['tau_off'] for r in rr]
        A.plot(x,y,'o-',c=cols[k],lw=1.5,ms=5,alpha=.85,label=LBL[k])
        for r in rr:
            A.annotate(f"{float(r['val']):g}",(r['m'][ch]['dprime'],r['m'][ch]['tau_off']),fontsize=6.5,alpha=.65,
                       xytext=(3,3),textcoords='offset points')
    A.scatter([base[ch]['dprime']],[base[ch]['tau_off']],s=180,marker='*',c='k',zorder=5,label='현행')
    A.set_xlabel("d′ (낮을수록 비자명 ←)"); A.set_ylabel('τ_off [스텝] (낮을수록 빠른 복귀 ↓)')
    A.set_title(f'{ttl} — 비자명성 vs 복귀속도 tradeoff',fontsize=11); A.grid(alpha=.25); A.legend(fontsize=8,ncol=2)
fig.suptitle('Q·R Pareto — 좌하단이 목표(비자명 + 빠른복귀)',fontsize=13)
fig.tight_layout(); fig.savefig('qr_ofat_pareto.png',dpi=110); plt.close(fig)
print('\nsaved qr_ofat_knobs.png, qr_ofat_pareto.png')
