import sys,os,time,json,numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0,'.'); sys.path.insert(0,'etc/scripts')
os.environ['SURR_V7']='1'
WID=int(sys.argv[1])
CLR=('RHUKF_PD','RHUKF_PINIT','RHUKF_FORM','RHUKF_ALPHA','RHUKF_R','RHUKF_N','RHUKF_HUBER_C','RHUKF_SPAS','RHUKF_Q','RHUKF_TAU','RHUKF_UI','NET_HIDDEN','ADAM_LR','OPT','ADAM_LOSS')
def go(name,env,ag,seed=42):
    for k in CLR: os.environ.pop(k,None)
    import surrogate_run as SR
    t0=time.time(); h=SR.run_config(env,n_ep=160,seed=seed,ep_steps=400,agent_type=ag)
    s=SR.summarize(h); s['crash']=int(sum(r['crashed'] for r in h))
    s['f1_early']=float(np.mean([r['f1'] for r in h[20:60] if r['has_atk']]))
    print(f"  {name:26s} F1={s['F1']:.3f} e={s['f1_early']:.3f} fpr={s['fpr']:.3f} cr={s['crash']} rwd={s['reward']:.0f} ({time.time()-t0:.0f}s)",flush=True)
    return s
# top2 SWIRL: P0.03, P0.01 (huber off) · Adam MSE · SGD MSE · lethal 30/40/50/60
SW1=dict(NET_HIDDEN=16,RHUKF_N=6,RHUKF_Q='1e-3',RHUKF_TAU='0.005',RHUKF_UI=1,RHUKF_FORM='absolute',RHUKF_SPAS=1,RHUKF_ALPHA='0.1',RHUKF_R='1',RHUKF_PINIT='0.03',RHUKF_HUBER_C='1e9')
SW2=dict(SW1,RHUKF_PINIT='0.01')
jobs=[]
for L in ('0.4','0.5'):
    jobs.append((f'SWIRL_P03_L{L}',dict(SW1),'rhukf',L))
    jobs.append((f'SWIRL_P01_L{L}',dict(SW2),'rhukf',L))
    jobs.append((f'Adam_L{L}',dict(NET_HIDDEN=16,ADAM_LR='3e-4',ADAM_LOSS='mse'),'adam',L))
    jobs.append((f'SGD_L{L}',dict(NET_HIDDEN=16,ADAM_LR='3e-4',OPT='sgd',ADAM_LOSS='mse'),'adam',L))
mine=jobs[WID::2]; out={}
print(f'[w{WID}] {len(mine)} runs',flush=True)
for n,e,ag,L in mine:
    os.environ['SURR_P_LETHAL']=L
    out[n]=go(n,e,ag)
json.dump(out,open(f'/tmp/v7lth_w{WID}.json','w'),default=float); open(f'V7LTH_W{WID}_DONE','w').close()
