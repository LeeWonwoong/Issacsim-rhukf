import sys,os,time,json,numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0,'.'); sys.path.insert(0,'etc/scripts')
os.environ['SURR_V7']='1'
WID=int(sys.argv[1])
CLR=('RHUKF_PD','RHUKF_PINIT','RHUKF_FORM','RHUKF_ALPHA','RHUKF_R','RHUKF_N','RHUKF_HUBER_C','RHUKF_SPAS','RHUKF_Q','RHUKF_TAU','RHUKF_UI','NET_HIDDEN')
def go(name,env,seed=42):
    for k in CLR: os.environ.pop(k,None)
    import surrogate_run as SR
    t0=time.time(); h=SR.run_config(env,n_ep=160,seed=seed,ep_steps=400,agent_type='rhukf')
    s=SR.summarize(h); s['crash']=int(sum(r['crashed'] for r in h))
    s['f1_early']=float(np.mean([r['f1'] for r in h[20:60] if r['has_atk']]))
    print(f"  {name:26s} F1={s['F1']:.3f} e={s['f1_early']:.3f} fpr={s['fpr']:.3f} cr={s['crash']} rwd={s['reward']:.0f} ({time.time()-t0:.0f}s)",flush=True)
    return s
B=dict(NET_HIDDEN=16,RHUKF_Q='1e-3',RHUKF_TAU='0.005',RHUKF_UI=1,RHUKF_FORM='absolute',RHUKF_SPAS=1)
# 상위3 config (S2 기준): a0.1_R1.5_P0.01, a0.1_R1_P0.03, a0.1_R2_P0.05
cfgs=[('a0.1_R1.5_P0.01',dict(B,RHUKF_ALPHA='0.1',RHUKF_R='1.5',RHUKF_PINIT='0.01')),
      ('a0.1_R1_P0.03',dict(B,RHUKF_ALPHA='0.1',RHUKF_R='1',RHUKF_PINIT='0.03')),
      ('a0.1_R2_P0.05',dict(B,RHUKF_ALPHA='0.1',RHUKF_R='2',RHUKF_PINIT='0.05'))]
jobs=[]
for nm,e in cfgs:
    jobs.append((f'{nm}_hON_N6',dict(e,RHUKF_N=6,RHUKF_HUBER_C='3.0')))
    jobs.append((f'{nm}_hOFF_N6',dict(e,RHUKF_N=6,RHUKF_HUBER_C='1e9')))
    jobs.append((f'{nm}_hON_N7',dict(e,RHUKF_N=7,RHUKF_HUBER_C='3.0')))
mine=jobs[WID::2]; out={}
print(f'[w{WID}] {len(mine)} runs',flush=True)
for n,e in mine: out[n]=go(n,e)
json.dump(out,open(f'/tmp/v7hn_w{WID}.json','w'),default=float); open(f'V7HN_W{WID}_DONE','w').close()
