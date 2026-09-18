"""현 프레임워크 Perceptual Aliasing 정밀 측정 (온셋 정렬 보정판)."""
import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'etc/scripts'); sys.path.insert(0,'.')
import numpy as np
from alias_core import replay, dprime, auc, overlap, bayes_err, ambiguous_frac, runs, PATS

OUT=[]
def P(s=''): OUT.append(str(s)); print(s)

# ── Dataset A: 오프라인 재구동 + 원자료 라벨 ──
R=np.load('scratchpad/A_cache.npy',allow_pickle=True).item()
D=np.load('results_zu_v2/zu_log.npz',allow_pickle=True)['data']
delay=D[:,21]; action=D[:,3]; reset=D[:,1]
n=len(delay)
# 버스트 종료 후 경과 (복귀 tail)
since_off=np.full(n,9999.0); c=9999
for i in range(n):
    if reset[i]==1: c=9999
    if delay[i]>=0: c=0
    else: c+=1
    since_off[i]=c
f=R['fresh']; atk=R['atk']==1
BEN   = f&(~atk)&(since_off>15)      # 평시(복귀 tail 15스텝 배제 = 순수 평시)
TAIL  = f&(~atk)&(since_off<=15)     # 공격 직후 복귀구간
ONSET = f&atk&(delay<3)              # 온셋 과도 (관측 아직 안 오름)
STEADY= f&atk&(delay>=3)             # 공격 정상상태 (관측 완전 반영)

P('='*106)
P(' Ⅰ. 관측의 시간구조 — 공격은 "즉시" 보이지 않는다 (aliasing 의 1차 원천)')
P('='*106)
P(' 버스트 온셋 이후 경과(RL스텝 @10Hz)별 압축 NIS 중앙값. δ 0.35~0.51(중간세기) 구간 캡처.')
P(f" {'경과':>12s} {'시간':>6s} {'n':>5s} | {'gyro':>6s} {'p90':>5s} | {'vel':>6s} {'p90':>5s}")
for lo,hi,nm in [(0,1,'온셋 0'),(1,3,'1-2'),(3,5,'3-4'),(5,10,'5-9'),(10,20,'10-19'),(20,99,'20+')]:
    m=f&atk&(delay>=lo)&(delay<hi)
    if m.sum()<10: continue
    P(f" {nm:>12s} {lo*0.1:5.1f}s {m.sum():5d} | {np.median(R['gyro'][m]):6.2f} {np.percentile(R['gyro'][m],90):5.2f} "
      f"| {np.median(R['vel'][m]):6.2f} {np.percentile(R['vel'][m],90):5.2f}")
P(f" {'[순수평시]':>12s} {'':>6s} {BEN.sum():5d} | {np.median(R['gyro'][BEN]):6.2f} {np.percentile(R['gyro'][BEN],90):5.2f} "
  f"| {np.median(R['vel'][BEN]):6.2f} {np.percentile(R['vel'][BEN],90):5.2f}")
P(' → gyro 는 온셋 후 3스텝(0.3s) 지나야 평시(0.46)에서 2.4 수준으로 오른다.')
P('   즉 **온셋 0-2스텝은 원리적으로 평시와 구분 불가**(관측에 아직 정보 없음) = 탐지지연 하한.')
P('   프레임워크의 대응 데드라인 d≤3 은 이 물리적 하한과 정확히 맞닿아 있다.')
P()
P(' 버스트 종료 후 복귀 (관측이 언제 평시로 돌아오나):')
P(f" {'경과':>12s} {'n':>5s} | {'gyro':>6s} | {'vel':>6s}")
for lo,hi,nm in [(1,3,'1-2'),(3,6,'3-5'),(6,11,'6-10'),(11,21,'11-20'),(21,41,'21-40'),(41,9998,'41+')]:
    m=f&(~atk)&(since_off>=lo)&(since_off<hi)
    if m.sum()<10: continue
    P(f" {nm:>12s} {m.sum():5d} | {np.median(R['gyro'][m]):6.2f} | {np.median(R['vel'][m]):6.2f}")
P(' → gyro 는 2-5스텝에 복귀, vel 은 20-40스텝(2-4s) 끌린다 = **OFF 구간이 "공격중"처럼 보이는 비대칭 aliasing**.')

# ── Ⅱ. 채널별 aliasing (온셋 보정) ──
def blk(neg,pos,tag):
    return (f'{tag:24s} n0={len(neg):5d} n1={len(pos):4d} | '
            f"d'={dprime(neg,pos):5.2f} AUC={auc(neg,pos):5.3f} OVL={overlap(neg,pos,0,3):5.3f} "
            f'BayesErr={bayes_err(neg,pos):5.3f} 애매질량={ambiguous_frac(neg,pos):5.3f}')
P(); P('='*106); P(' Ⅱ. 채널별 aliasing — 온셋 과도 제외(정상상태 공격 vs 순수 평시)'); P('='*106)
P(' '+blk(R['gyro'][BEN],R['gyro'][STEADY],'gyro NIS  정상상태'))
P(' '+blk(R['vel'][BEN], R['vel'][STEADY], 'vel  NIS  정상상태'))
P(' '+blk(R['gyro'][BEN],R['gyro'][ONSET], 'gyro NIS  온셋0-2'))
P(' '+blk(R['vel'][BEN], R['vel'][ONSET],  'vel  NIS  온셋0-2'))
P(' '+blk(R['gyro'][BEN],R['gyro'][TAIL],  'gyro NIS  복귀tail'))
P(' '+blk(R['vel'][BEN], R['vel'][TAIL],   'vel  NIS  복귀tail'))
P(' → **gyro 는 정상상태에서 거의 완전 분리(easy anchor), 온셋/복귀에서만 aliasing.**')
P('   vel 은 정상상태에서도 크게 겹친다 = 지속적 POMDP 채널.')
open('scratchpad/alias_p1.txt','w').write('\n'.join(OUT))
