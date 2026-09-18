#!/usr/bin/env python3
"""scripts/grid.py — 격자 실험 실행기. 격자 정의는 YAML 한 장, 각 런은 train.py 서브프로세스.

    python3 scripts/grid.py list configs/grids/cp_regime.yaml
    setsid nohup python3 scripts/grid.py run configs/grids/cp_regime.yaml --gpu 10 --cpu 4 > grid.out 2>&1 < /dev/null &

격자 YAML
    base: [configs/cp_regime.yaml]        # 공통 설정(여러 개면 뒤가 덮어씀)
    outdir: results/claudecodefortest/X   # 런별 하위 폴더 <stage>_<learner>_<축값>/
    seeds: [42]
    stages:                               # 순서대로 실행(앞 단계가 끝나야 다음 단계)
      lin: {}
      const: {reward.fn_per_step: 0.0}
    learners:
      swirl:
        device: gpu                       # gpu | cpu (워커 풀)
        set: {agent.type: swirl}
        axes:                             # 데카르트 곱. 값이 dict 면 여러 키를 한 번에(이름은 _name 또는 인덱스)
          reward.scale: [0.2, 0.4]
          agent.swirl.R: [0.5, 1]
          tu: [{_name: cp, agent.tau: 0.005, agent.update_interval: 1}, {_name: ll, agent.tau: 0.02, agent.update_interval: 4}]
이미 hist.json 이 있는 런은 건너뛴다(재실행 안전). 로그: <outdir>/grid.log, 런별 <run>/train.log.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _short(key: str) -> str:
    return key.split('.')[-1]


def _fmt(v) -> str:
    return f'{v:g}' if isinstance(v, float) else str(v)


def jobs(spec: dict):
    """[(stage, learner, device, run_name, [--set ...])] — 단계 순서 유지."""
    out = []
    for stage, st_set in (spec.get('stages') or {'main': {}}).items():
        for ln, L in spec['learners'].items():
            axes = L.get('axes') or {}
            names = list(axes)
            for combo in itertools.product(*[axes[k] for k in names]):
                sets = dict(st_set or {}); sets.update(L.get('set') or {}); tag = []
                for k, v in zip(names, combo):
                    if isinstance(v, dict):
                        vv = dict(v); nm = vv.pop('_name', None)
                        sets.update(vv); tag.append(f'{_short(k)}{nm if nm is not None else axes[k].index(v)}')
                    else:
                        sets[k] = v; tag.append(f'{_short(k)}{_fmt(v)}')
                for sd in spec.get('seeds', [42]):
                    name = '_'.join([stage, ln] + tag + [f's{sd}'])
                    s2 = dict(sets); s2['run.seed'] = sd
                    out.append((stage, ln, L.get('device', 'gpu'), name, s2))
    names = [j[3] for j in out]
    assert len(names) == len(set(names)), '런 이름 중복 — 축 값을 확인'
    return out


def _cmd(spec, name, sets):
    outdir = os.path.join(spec['outdir'], name)
    cmd = [sys.executable, os.path.join(ROOT, 'train.py')]
    for b in (spec['base'] if isinstance(spec['base'], list) else [spec['base']]):
        cmd += ['--config', os.path.join(ROOT, b) if not os.path.isabs(b) else b]
    for k, v in sets.items():
        cmd += ['--set', f'{k}={v}' if isinstance(v, str) else f'{k}={json.dumps(v)}']   # JSON = 유효한 YAML
    cmd += ['--set', f'run.outdir={outdir}', '--set', f'run.name={name}']
    return cmd, outdir


def worker(spec, stage, device, wid, nw):
    mine = [j for j in jobs(spec) if j[0] == stage and j[2] == device]
    for i, (_, _, _, name, sets) in enumerate(mine):
        if i % nw != wid:
            continue
        cmd, outdir = _cmd(spec, name, sets)
        if os.path.exists(os.path.join(outdir, 'hist.json')):
            continue
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        with open(os.path.join(outdir, 'stdout.log'), 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print(f'{time.strftime("%m-%d %H:%M")} {name} rc={rc} {(time.time() - t0) / 60:.1f}분', flush=True)


def run(spec, gyaml, n_gpu, n_cpu, omp_gpu, omp_cpu):
    os.makedirs(spec['outdir'], exist_ok=True)
    log = open(os.path.join(spec['outdir'], 'grid.log'), 'a')

    def L(m):
        line = f'[{time.strftime("%m-%d %H:%M")}] {m}'; print(line, flush=True); log.write(line + '\n'); log.flush()
    J = jobs(spec)
    L(f'격자 {gyaml}: 런 {len(J)}개, 단계 {list(dict.fromkeys(j[0] for j in J))}, GPU 워커 {n_gpu}·CPU 워커 {n_cpu}')
    for stage in dict.fromkeys(j[0] for j in J):
        procs = []
        for dev, nw, omp in (('gpu', n_gpu, omp_gpu), ('cpu', n_cpu, omp_cpu)):
            n = sum(1 for j in J if j[0] == stage and j[2] == dev)
            nw = min(nw, n)
            env = dict(os.environ, OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp))
            if dev == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
            for w in range(nw):
                lf = open(os.path.join(spec['outdir'], f'worker_{stage}_{dev}{w}.log'), 'a')
                procs.append(subprocess.Popen([sys.executable, __file__, 'worker', gyaml, stage, dev, str(w), str(nw)],
                                              stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT))
        L(f'단계 {stage} 시작 (워커 {len(procs)})')
        for p in procs:
            p.wait()
        done = sum(os.path.exists(os.path.join(spec['outdir'], j[3], 'hist.json')) for j in J if j[0] == stage)
        L(f'단계 {stage} 종료 — 완료 {done}/{sum(1 for j in J if j[0] == stage)}')
    open(os.path.join(spec['outdir'], 'GRID_DONE'), 'w').close()
    L('전체 종료')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['list', 'run', 'worker'])
    ap.add_argument('grid')
    ap.add_argument('rest', nargs='*')
    ap.add_argument('--gpu', type=int, default=10)
    ap.add_argument('--cpu', type=int, default=4)
    ap.add_argument('--omp-gpu', type=int, default=4)
    ap.add_argument('--omp-cpu', type=int, default=1)
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.grid))
    if a.mode == 'list':
        for j in jobs(spec):
            print(j[2], j[3])
        print(len(jobs(spec)), '런')
    elif a.mode == 'run':
        run(spec, a.grid, a.gpu, a.cpu, a.omp_gpu, a.omp_cpu)
    else:
        stage, dev, wid, nw = a.rest
        worker(spec, stage, dev, int(wid), int(nw))


if __name__ == '__main__':
    main()
