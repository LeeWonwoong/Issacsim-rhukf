import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

O=pd.read_csv('results_deadline/sweep_summary.csv')          # 원본 position-hold
N=pd.read_csv('results_deadline_softhold/sweep_summary.csv') # soft hold
for df in (O,N): df['delay']=df.policy.map(lambda p:99 if p=='track' else int(p.replace('dhover','')))
okey=[0,3,5,7,99]; xl=['0','3','5','7','track']
CELLS=[('hover',1.37),('hover',1.40),('waypoint',1.37),('waypoint',1.40)]

def surv(df,pat,b): return [df[(df.pattern==pat)&(df.bias==b)&(df.delay==d)].survived.mean() for d in okey]

print('='*70); print('position-hold(원본) → soft-hold  생존율 비교'); print('='*70)
for pat,b in CELLS:
    o=surv(O,pat,b); n=surv(N,pat,b)
    print(f'\n {pat} b={b}   delay: '+' '.join(f'{x:>6}' for x in xl))
    print('   원본 pos-hold : '+' '.join(f'{v:6.2f}' for v in o))
    print('   soft-hold     : '+' '.join(f'{v:6.2f}' for v in n))

fig,ax=plt.subplots(1,4,figsize=(19,4.6),sharey=True)
for j,(pat,b) in enumerate(CELLS):
    a=ax[j]; xs=[0,3,5,7]
    o=surv(O,pat,b); n=surv(N,pat,b)
    a.plot(xs,o[:4],'o--',color='gray',label='pos-hold (old)')
    a.plot(xs,n[:4],'s-',color='tab:blue',label='soft-hold (new)')
    a.axhline(o[4],ls=':',color='gray',alpha=.6); a.axhline(n[4],ls=':',color='tab:blue',alpha=.6)
    a.scatter([8],[o[4]],color='gray',marker='o'); a.scatter([8],[n[4]],color='tab:blue',marker='s')
    a.axhline(0.95,color='r',ls='--',lw=.8)
    a.set_title(f'{pat} b={b}'); a.set_xlabel('delay (steps); 8=track'); a.set_xticks([0,3,5,7,8]); a.set_xticklabels(['0','3','5','7','trk'])
    a.set_ylim(-.05,1.08); a.grid(alpha=.3)
    if j==0: a.set_ylabel('survival'); a.legend(fontsize=8)
plt.suptitle('Soft-hold vs position-hold: survival vs response delay (N=12)',fontsize=13)
plt.tight_layout(); plt.savefig('results_deadline_softhold/softhold_compare.png',dpi=110)
print('\n[plot] results_deadline_softhold/softhold_compare.png')
