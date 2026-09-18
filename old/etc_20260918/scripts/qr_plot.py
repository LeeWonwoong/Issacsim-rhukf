# -*- coding: utf-8 -*-
import os,json,numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
G=json.load(open('scratchpad/qr_grid2d.json')); O=json.load(open('scratchpad/qr_ofat.json'))
C=json.load(open('scratchpad/qr_cand.json'))
QG=sorted(set(r['qg'] for r in G)); RG=sorted(set(r['rg'] for r in G))
def M(key,ch='gyro'):
    A=np.full((len(QG),len(RG)),np.nan)
    for r in G: A[QG.index(r['qg']),RG.index(r['rg'])]=r['m'][ch][key]
    return A
fig=plt.figure(figsize=(17,10)); gs=fig.add_gridspec(2,3,hspace=.33,wspace=.28)
for i,(key,ttl,cm) in enumerate([('dprime',"gyro d′ — 낮을수록 비자명",'viridis_r'),
                                 ('tau_off','gyro τ_off [스텝] — 낮을수록 빠른복귀','magma_r'),
                                 ('plateau','gyro 공격 고원 — 높아야 탐지가능','viridis')]):
    A=fig.add_subplot(gs[0,i]); Z=M(key)
    im=A.imshow(Z,cmap=cm,aspect='auto',origin='lower')
    A.set_xticks(range(len(RG))); A.set_xticklabels([f'{v:g}' for v in RG],fontsize=8,rotation=45)
    A.set_yticks(range(len(QG))); A.set_yticklabels([f'{v:g}' for v in QG],fontsize=8)
    A.set_xlabel('R_gyro'); A.set_ylabel('Q_gyro'); A.set_title(ttl,fontsize=11)
    for a in range(len(QG)):
        for b in range(len(RG)): A.text(b,a,f'{Z[a,b]:.2f}',ha='center',va='center',fontsize=7,
                                        color='w' if (Z[a,b]-np.nanmin(Z))/(np.nanmax(Z)-np.nanmin(Z))>.55 else 'k')
    A.scatter([RG.index(0.2)],[QG.index(5e-3)],s=200,marker='*',c='red',zorder=5)
    plt.colorbar(im,ax=A,fraction=.046)
# Pareto
A=fig.add_subplot(gs[1,0])
sc=A.scatter([r['m']['gyro']['dprime'] for r in G],[r['m']['gyro']['tau_off'] for r in G],
             c=[r['m']['gyro']['plateau'] for r in G],cmap='viridis',s=55,alpha=.85)
plt.colorbar(sc,ax=A,label='공격 고원(탐지가능성)')
for nm,cfg,m,e in C:
    A.annotate(nm.split('_')[1] if '_' in nm else nm,(m['gyro']['dprime'],m['gyro']['tau_off']),
               fontsize=8,fontweight='bold',xytext=(4,4),textcoords='offset points')
    A.scatter([m['gyro']['dprime']],[m['gyro']['tau_off']],s=140,marker='D',
              edgecolor='r',facecolor='none',lw=1.8,zorder=6)
A.set_xlabel("d′ (← 비자명)"); A.set_ylabel('τ_off [스텝] (↓ 빠른복귀)')
A.set_title('gyro Pareto — 좌하단이 목표',fontsize=11); A.grid(alpha=.25)
# R_vel → vel
A=fig.add_subplot(gs[1,1])
rv=sorted([r for r in O if r['knob']=='r_vel'],key=lambda r: float(r['val']))
x=[float(r['val']) for r in rv]
A.semilogx(x,[r['m']['vel']['tau_off'] for r in rv],'o-',c='#1f77b4',lw=2,label='vel τ_off [스텝]')
A2=A.twinx(); A2.semilogx(x,[r['m']['vel']['dprime'] for r in rv],'s--',c='#d62728',lw=2,label="vel d′")
A2.set_ylabel("vel d′",color='#d62728')
A.axvline(0.1,c='k',ls=':',lw=1.5); A.text(0.105,38,'현행 0.1',fontsize=9)
A.set_xlabel('R_vel'); A.set_ylabel('vel τ_off [스텝]',color='#1f77b4')
A.set_title('R_vel — vel 잔향(복귀 tail)의 지배 노브',fontsize=11); A.grid(alpha=.25); A.legend(fontsize=8,loc='center right')
# 후보 비교
A=fig.add_subplot(gs[1,2])
nms=[c[0].split('_')[1] if '_' in c[0] else c[0] for c in C]
snr=[c[3]['gyro']['steady_snr'] for c in C]; berr=[c[3]['gyro']['steady_berr'] for c in C]
vt=[c[2]['vel']['tau_off'] for c in C]
xx=np.arange(len(C)); w=.27
A.bar(xx-w,snr,w,label='gyro SNR(정상) ↓좋음',color='#d62728')
A.bar(xx,[b*40 for b in berr],w,label='gyro Bayes오류 ×40 ↑좋음',color='#2ca02c')
A.bar(xx+w,[t/5 for t in vt],w,label='vel τ_off ÷5 ↓좋음',color='#1f77b4')
A.set_xticks(xx); A.set_xticklabels(nms,fontsize=8.5,rotation=15)
A.set_title('후보 config 비교 (A=현행)',fontsize=11); A.legend(fontsize=8); A.grid(alpha=.25,axis='y')
fig.suptitle('Q·R 전 노브 스윕 — 고집·온셋·오프셋·비자명성 동시 최적화 (오프라인 zu_v2 재구동)',fontsize=13.5)
fig.savefig('qr_sweep_summary.png',dpi=110,bbox_inches='tight'); plt.close(fig)
print('saved qr_sweep_summary.png')
