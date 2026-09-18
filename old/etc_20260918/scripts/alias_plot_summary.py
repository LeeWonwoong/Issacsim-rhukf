import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'etc/scripts'); sys.path.insert(0,'.')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
from alias_core import overlap, dprime, bayes_err, auc, PATS
# ── A ──
R=np.load('scratchpad/A_cache.npy',allow_pickle=True).item()
D=np.load('results_zu_v2/zu_log.npz',allow_pickle=True)['data']
delay=D[:,21]; reset=D[:,1]; n=len(delay)
so=np.full(n,9999.); c=9999
for i in range(n):
    if reset[i]==1: c=9999
    if delay[i]>=0: c=0
    else: c+=1
    so[i]=c
f=R['fresh']; atk=R['atk']==1; idx=np.where(f)[0]
E=dict(g=R['gyro'][idx],v=R['vel'][idx],a=atk[idx],w=R['w'][idx],dl=delay[idx],so=so[idx])
BEN=(~E['a'])&(E['so']>15); ST=E['a']&(E['dl']>=3); ON=E['a']&(E['dl']<3); TL=(~E['a'])&(E['so']<=15)
THR=np.percentile(E['g'][ST],10)
# ── B ──
B=np.load('scratchpad/B_cache.npy',allow_pickle=True).item()
ep,st,a,tr=B['ep'],B['step'],B['atk'],B['track']
cont=np.r_[False,(ep[1:]==ep[:-1])&(st[1:]>st[:-1])]
clean=np.ones(len(a),bool)
for k in range(1,5):
    prev=np.r_[np.zeros(k,int),a[:-k]]; same=np.r_[np.zeros(k,bool),(ep[k:]==ep[:-k])]
    clean &= ~((prev==1)&same)
BENb=(a==0)&cont&clean&tr; SUSb=(a==1)&cont&(np.r_[False,a[:-1]==1])&tr

fig=plt.figure(figsize=(16,9.5))
gs=fig.add_gridspec(2,3,hspace=.34,wspace=.26)
# (1) 분포 — 상태별
A=fig.add_subplot(gs[0,0]); bins=np.linspace(0,3.05,50)
for m,lab,cc in [(BEN,'평시(공격後 1.5s 배제)','#2ca02c'),(ON,'공격 온셋 0-2스텝','#ff7f0e'),
                 (ST,'공격 정상상태 (≥3스텝)','#d62728'),(TL,'공격 종료後 tail','#9467bd')]:
    A.hist(E['g'][m],bins,density=True,histtype='step',lw=2,color=cc,label=f'{lab} (n={m.sum()})')
A.axvline(THR,c='k',ls=':',lw=1.4); A.set_title('gyro NIS — 상태별 분포',fontsize=11)
A.set_xlabel('압축 gyro NIS'); A.set_ylabel('밀도'); A.legend(fontsize=8); A.grid(alpha=.25)
A=fig.add_subplot(gs[0,1])
for m,lab,cc in [(BEN,'평시','#2ca02c'),(ON,'온셋 0-2','#ff7f0e'),(ST,'정상상태','#d62728'),(TL,'종료後 tail','#9467bd')]:
    A.hist(E['v'][m],bins,density=True,histtype='step',lw=2,color=cc,label=lab)
A.set_title('vel NIS — 상태별 분포 (전 구간 겹침)',fontsize=11); A.set_xlabel('압축 vel NIS'); A.legend(fontsize=8); A.grid(alpha=.25)
# (2) δ 의존
A=fig.add_subplot(gs[0,2])
ds=[(0.10,0.20),(0.20,0.30),(0.30,0.45),(0.45,0.60),(0.60,0.80)]
xs=[(l+h)/2 for l,h in ds]
g0=B['gyro'][BENb]; v0=B['vel'][BENb]
bg=[bayes_err(g0,B['gyro'][SUSb&(B['delta']>=l)&(B['delta']<h)]) for l,h in ds]
bv=[bayes_err(v0,B['vel'][SUSb&(B['delta']>=l)&(B['delta']<h)]) for l,h in ds]
A.plot(xs,bg,'o-',c='#d62728',lw=2,label='gyro'); A.plot(xs,bv,'s-',c='#1f77b4',lw=2,label='vel')
A.axhline(0.5,c='k',ls='--',lw=1,alpha=.5); A.text(0.45,0.47,'0.5 = 완전 구분불가',fontsize=8)
A.set_xlabel('공격 세기 δ'); A.set_ylabel('1스텝 Bayes 오류 (균형)'); A.set_ylim(0,0.55)
A.set_title('공격 세기별 aliasing (정상상태)',fontsize=11); A.legend(fontsize=9); A.grid(alpha=.25)
# (3) 조건별 평시 바닥
A=fig.add_subplot(gs[1,0])
conds=[('무풍\n0-0.5',(B['ws']<0.5)),('약풍\n0.5-2',(B['ws']>=0.5)&(B['ws']<2)),('강풍\n4-7',(B['ws']>=4)&(B['ws']<7)),
       ('최강풍\n7-10',(B['ws']>=7)),('최강풍×\naggressive',(B['ws']>=7)&(B['pat']==3))]
med=[np.median(B['gyro'][BENb&c]) for _,c in conds]; p90=[np.percentile(B['gyro'][BENb&c],90) for _,c in conds]
p99=[np.percentile(B['gyro'][BENb&c],99) for _,c in conds]
x=np.arange(len(conds)); A.bar(x-.25,med,.25,label='중앙',color='#2ca02c')
A.bar(x,p90,.25,label='90p',color='#7fbf7f'); A.bar(x+.25,p99,.25,label='99p',color='#c7e9c0',edgecolor='#2ca02c')
A.axhline(np.percentile(B['gyro'][SUSb],10),c='r',ls='--',lw=1.5,label='공격 하위10%')
A.set_xticks(x); A.set_xticklabels([c[0] for c in conds],fontsize=8.5); A.set_ylabel('평시 gyro NIS')
A.set_title('회전바람·궤적이 올리는 평시 gyro 바닥',fontsize=11); A.legend(fontsize=8); A.grid(alpha=.25,axis='y')
# (4) Q_gyro A/B
A=fig.add_subplot(gs[1,1])
qs=[1e-3,2e-3,5e-3,1e-2]; alias=[4.66,2.58,0.83,1.16]; p99q=[2.42,1.90,1.26,1.14]; berr=[0.060,0.043,0.034,0.035]
A.semilogx(qs,alias,'o-',c='#d62728',lw=2,label='평시 aliasing율 [%]')
A.semilogx(qs,p99q,'s-',c='#ff7f0e',lw=2,label='평시 gyro 99p')
A2=A.twinx(); A2.semilogx(qs,berr,'^--',c='#1f77b4',lw=1.6,label='Bayes 오류'); A2.set_ylabel('Bayes 오류',color='#1f77b4')
A.axvline(5e-3,c='k',ls=':',lw=1.5); A.text(5.3e-3,4.0,'현행\n5e-3',fontsize=8.5)
A.axvline(2e-3,c='gray',ls=':',lw=1.2); A.text(1.15e-3,4.0,'문서 Ⅵ\n2e-3',fontsize=8.5,color='gray')
A.set_xlabel('UKF Q_gyro'); A.set_ylabel('aliasing율 [%] / 99p')
A.set_title('Q_gyro = gyro aliasing 의 지배 노브',fontsize=11); A.legend(fontsize=8,loc='upper right'); A.grid(alpha=.25)
# (5) tail 오염 데모
A=fig.add_subplot(gs[1,2])
lbl=['tail 미배제\n(구 분석)','tail 15스텝\n배제','tail 30스텝\n배제']
v15=[2.43,1.09,0.90]; v90=[3.00,1.97,1.49]
x=np.arange(3); A.bar(x-.19,v15,.38,label='중앙',color='#9467bd'); A.bar(x+.19,v90,.38,label='90p',color='#c5b0d5')
A.axhline(2.52,c='r',ls='--',lw=1.5,label='공격 정상상태 중앙')
A.set_xticks(x); A.set_xticklabels(lbl,fontsize=8.5); A.set_ylabel('평시 급기동(|ω|>1.5) gyro NIS')
A.set_title('"기동 aliasing" 의 정체 = 공격 종료後 잔향',fontsize=11); A.legend(fontsize=8); A.grid(alpha=.25,axis='y')
fig.suptitle('Fig 3. Perceptual aliasing 정밀 측정 — 진폭·세기·조건·필터 Q·평시 정의',fontsize=13.5)
fig.savefig('alias_fig3_summary.png',dpi=110,bbox_inches='tight'); plt.close(fig)
print('fig3 saved')
