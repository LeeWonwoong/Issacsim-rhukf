#!/usr/bin/env python3
"""scripts/eval_model.py — 저장된 final_model.pt 를 불러 train.evaluate(greedy, 별도 시드) 로 다시 평가 → eval.json 덮어씀.

    python3 scripts/eval_model.py <런 폴더> [<런 폴더> ...] [--n 100] [--device cpu|cuda]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfgload import load_experiment
from env.observation import ObsBuilder
import train


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 100
    dev = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'cpu'
    for d in args:
        exp = load_experiment([os.path.join(d, 'config.yaml')], [f'run.device={dev}', f'run.outdir={d}'])
        agent = train.make_agent(exp); agent.load(os.path.join(d, 'final_model.pt'))
        ev = train.evaluate(agent, exp, ObsBuilder(exp.obs), n)
        with open(os.path.join(d, 'eval.json'), 'w') as f: json.dump(ev, f, default=float, ensure_ascii=False)
        print(f"{os.path.basename(d)}: F1 {ev['f1']:.3f} P {ev['prec']:.3f} R {ev['rec']:.3f} FPR {ev['fpr']:.4f} 사건탐지 {ev['event_det']:.2f}/{ev['n_events']} 지연 {ev['delay']:.2f}(중앙 {ev['delay_med']:.1f}) 오경보에피 {ev['fa_episode_rate']:.2f} | "
              + ' '.join(f"{k.replace('_persist','P').replace('_burst','B')} {v:.2f}" for k, v in ev['rec_by_cls'].items() if v is not None))


if __name__ == '__main__':
    main()
