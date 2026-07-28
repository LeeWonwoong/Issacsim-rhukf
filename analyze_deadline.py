import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = pd.read_csv('results_deadline/sweep_detail.csv')
S = pd.read_csv('results_deadline/sweep_summary.csv')
ATK = 100
def delaycond(p): return 'no_resp' if p=='track' else int(p.replace('dhover',''))
S['delay'] = S.policy.map(delaycond)
D['delay'] = D.policy.map(delaycond)
DELAYS = [0,3,5,7,'no_resp']
PATS = ['hover','waypoint','figure8']
BIAS = sorted(S.bias.unique())

print('='*72); print('[P1-1] 패턴별 지연-생존율 (survival mean; N=12/cell)'); print('='*72)
for pat in PATS:
    print(f'\n  {pat}:')
    print('    delay:   ' + '  '.join(f'{str(d):>7}' for d in DELAYS))
    for b in BIAS:
        row=[]
        for d in DELAYS:
            v=S[(S.pattern==pat)&(S.bias==b)&(S.delay==d)].survived.mean()
            row.append(f'{v:7.2f}')
        print(f'    b={b}: '+'  '.join(row))

print('\n'+'='*72); print('[P1-2] 패턴별 마감 = 생존율 ≥0.95 유지하는 최대 delay (bias별)'); print('='*72)
for pat in PATS:
    for b in BIAS:
        ok=[d for d in [0,3,5,7] if S[(S.pattern==pat)&(S.bias==b)&(S.delay==d)].survived.mean()>=0.95]
        dl = max(ok) if ok else '<0(즉각도 미달)'
        print(f'  {pat:8} b={b}: 마감(최대허용지연)= {dl}')
print('  → RL 기준 = 가장 빡빡한 패턴/bias의 마감')

print('\n'+'='*72); print('[P1-3] 추락원인 분해 (delay별, 전 패턴/bias 합산) flip vs altitude vs drift'); print('='*72)
print('  delay | timeout flip altitude drift')
for d in DELAYS:
    s=S[S.delay==d]; n=len(s)
    def c(r): return (s.crash_reason==r).sum()
    print(f'  {str(d):>7} | {(s.crash_reason=="timeout").sum():3d}   {c("crash_flip"):3d}  {c("crash_altitude"):4d}    {c("crash_drift"):3d}   (N={n})')

print('\n'+'='*72); print('[즉각대응] delay별 전환후 고도/자세 (dhover 셀만; 전환=step 100+d)'); print('='*72)
print('  pattern bias delay | switch|roll| switch|pitch| min_alt(전환후) alt@+60')
for pat in PATS:
    for b in [1.40]:  # 최악 bias 대표
        for d in [0,3,5,7]:
            sw=ATK+d
            sub=D[(D.pattern==pat)&(D.bias==b)&(D.delay==d)]
            if len(sub)==0: continue
            at_sw=sub[sub.step==sw]
            post=sub[sub.step>=sw]
            at60=sub[sub.step==sw+60]
            def m(x,c): return x[c].abs().mean() if len(x) else float('nan')
            minalt=post.groupby('episode').alt.min().mean() if len(post) else float('nan')
            a60=at60.alt.mean() if len(at60) else float('nan')
            print(f'  {pat:8} {b} d={d} | {m(at_sw,"roll"):.4f}      {m(at_sw,"pitch"):.4f}       {minalt:.3f}          {a60:.3f}')

print('\n'+'='*72); print('[P2-1] 정상 pos NIS 바닥 (공격 전 step<100, 패턴별 raw)'); print('='*72)
pre=D[D.step<ATK]
for pat in PATS:
    s=pre[pre.pattern==pat]
    print(f'  {pat:8}: pos median={s.nis_p_raw.median():.3f} p99={s.nis_p_raw.quantile(.99):.3f} max={s.nis_p_raw.max():.2f} '
          f'| vel med={s.nis_v_raw.median():.3f} gyr med={s.nis_g_raw.median():.3f}')

print('\n'+'='*72); print('[P2-2] 침묵구간 pos NIS 단조증가? (no_resp track, 온셋 100 기준 상대스텝)'); print('='*72)
tr=D[D.delay=='no_resp'].copy(); tr['rel']=tr.step-ATK
print('  rel(온셋후):  ' + '  '.join(f'{r:>6}' for r in [0,2,4,6,8,10,12]))
for chan in ['nis_p_raw','nis_v_raw','nis_g_raw','gt_err']:
    vals=[tr[tr.rel==r][chan].mean() for r in [0,2,4,6,8,10,12]]
    print(f'  {chan:10}: ' + '  '.join(f'{v:6.3f}' for v in vals))

# ---- 플롯 ----
fig,ax=plt.subplots(2,3,figsize=(16,9))
xt=[0,3,5,7]; xl=['0','3','5','7']
# row0: deadline curves per pattern
for j,pat in enumerate(PATS):
    a=ax[0][j]
    for b in BIAS:
        y=[S[(S.pattern==pat)&(S.bias==b)&(S.delay==d)].survived.mean() for d in xt]
        yn=S[(S.pattern==pat)&(S.bias==b)&(S.delay=='no_resp')].survived.mean()
        a.plot(xt,y,'o-',label=f'b={b}')
        a.axhline(yn,ls=':',alpha=.4)
    a.axhline(0.95,color='r',ls='--',lw=1); a.set_ylim(-.05,1.05)
    a.set_title(f'Deadline: {pat}'); a.set_xlabel('response delay (steps)'); a.set_ylabel('survival'); a.set_xticks(xt); a.grid(alpha=.3); a.legend(fontsize=8)
# row1-0: crash decomposition by delay
a=ax[1][0]
reasons=['crash_flip','crash_altitude','crash_drift']
bottom=np.zeros(len(DELAYS))
for r in reasons:
    vals=[(S[S.delay==d].crash_reason==r).sum() for d in DELAYS]
    a.bar([str(x) for x in DELAYS],vals,bottom=bottom,label=r.replace('crash_',''))
    bottom+=vals
a.set_title('Crash cause by delay (all pat/bias)'); a.set_xlabel('delay'); a.set_ylabel('# crashes'); a.legend(fontsize=8)
# row1-1: switch attitude vs delay (b=1.40)
a=ax[1][1]
for pat in PATS:
    sw_roll=[D[(D.pattern==pat)&(D.bias==1.40)&(D.delay==d)&(D.step==ATK+d)].roll.abs().mean() for d in [0,3,5,7]]
    a.plot([0,3,5,7],sw_roll,'s-',label=pat)
a.set_title('Switch-time |roll| vs delay (b=1.40)'); a.set_xlabel('delay'); a.set_ylabel('|roll| at switch (rad)'); a.grid(alpha=.3); a.legend(fontsize=8)
# row1-2: pos/vel/gyr NIS in silence window (no_resp, b avg)
a=ax[1][2]
tr=D[D.delay=='no_resp'].copy(); tr['rel']=tr.step-ATK
rr=list(range(-5,16))
for chan,c in [('nis_p_raw','tab:blue'),('nis_v_raw','tab:green'),('nis_g_raw','tab:red')]:
    v=[tr[tr.rel==r][chan].mean() for r in rr]
    a.plot(rr,v,'-',color=c,label=chan.replace('nis_','').replace('_raw',''))
a.axvline(0,ls='--',c='gray'); a.set_title('NIS onset/silence (no_response)'); a.set_xlabel('steps rel. onset'); a.set_ylabel('NIS raw'); a.grid(alpha=.3); a.legend(fontsize=8)
plt.suptitle('Deadline re-measurement (540ep) — survival deadlines, crash cause, switch attitude, pos-NIS',fontsize=13)
plt.tight_layout(); plt.savefig('results_deadline/deadline_analysis.png',dpi=110)
print('\n[plot] results_deadline/deadline_analysis.png')
