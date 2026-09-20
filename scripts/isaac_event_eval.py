#!/usr/bin/env python3
# Isaac 학습 steps/ → 사건 단위 클래스별 탐지율·지연 + 무공격 에피 오경보율 (ep>EP0). 라벨 규약: δ 활성 [start,end) → 관측·행동 한 스텝 뒤. 행동 a_t = 다음 행 prev_action.
import numpy as np,glob,json,sys
K=['strong_persist','trans_persist','weak_persist','strong_burst','trans_burst','weak_burst']
R='/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/isaac_v2/'; EP0=int(sys.argv[1]) if len(sys.argv)>1 else 100
for d in sorted(glob.glob(R+'*_s*')):
    res={}; nfa=0; ncl=0; fp=0; tn=0
    for f in sorted(glob.glob(d+'/steps/ep*.npz')):
        ep=int(f[-8:-4])
        if ep<=EP0: continue
        z=np.load(f,allow_pickle=True); ev=json.loads(str(z['events'])) if z['events'].shape==() else []
        Rw=z['rows']; cols=[str(c) for c in z['cols']]; C={k:Rw[:,i] for i,k in enumerate(cols)}
        st=C['step'].astype(int); pa=C['prev_action'].astype(int)
        a=np.full(st.max()+2,-1); a[st[:-1]]=pa[1:]
        if not ev: ncl+=1; nfa+=int((a[a>=0]>0).any())
        for e in ev:
            s0,e0,c=e['start'],e['end'],e['cls']; lab=a[s0+1:min(e0+1,len(a))]; lab=lab[lab>=0]
            det=int((lab>0).any()) if len(lab) else 0; dl=int(np.argmax(lab>0)) if det else None
            r=res.setdefault(c,dict(n=0,det=0,dl=[])); r['n']+=1; r['det']+=det
            if det: r['dl'].append(dl)
    if not res: continue
    tot=sum(r['n'] for r in res.values()); det=sum(r['det'] for r in res.values()); dl=np.mean(sum((r['dl'] for r in res.values()),[]))
    print(f"{d.split('/')[-1]:10s} ep>{EP0}: 사건 {tot} 탐지 {det/tot:.3f} 지연 {dl:.2f} | 무공격 {ncl} 중 오경보에피 {nfa} ({nfa/max(ncl,1):.2f}) | " + ' '.join(f"{c.split('_')[0][0]}{c.split('_')[1][0]} {res[c]['det']/res[c]['n']:.2f}" for c in K if c in res))
