#!/usr/bin/env python3
"""export_policy.py — 학습 결과(final_model.pt) → 실기 추론용 npz (torch 불요, numpy 만).

    python3 px4_field/export_policy.py results/claudecodefortest/isaac_v2/swirl_s42/final_model.pt px4_field/models/swirl_v2_s42.npz [--net active]

npz 내용: theta(flat), 레이어별 (W_start, W_len, W_shape, b_start, b_len), act_name, dimS, nA, obs spec(compress/clip/div/window/features).
행동망은 학습 규칙과 같게 기본 theta_target(SWIRL act=target). Adam 도 target 망(평가 규칙 동일).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pt'); ap.add_argument('out')
    ap.add_argument('--net', choices=['target', 'active'], default='target', help='내보낼 망 (기본 target = 학습 행동 규칙)')
    a = ap.parse_args()
    import torch
    sd = torch.load(a.pt, map_location='cpu', weights_only=False)
    theta = sd['theta_target' if a.net == 'target' else 'theta']
    theta = np.asarray(theta.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    info = sd['info']; cfg = sd['config']
    layers = info['layers']
    W_start = np.array([l['W_start'] for l in layers]); W_len = np.array([l['W_len'] for l in layers])
    W_rows = np.array([l['W_shape'][0] for l in layers]); W_cols = np.array([l['W_shape'][1] for l in layers])
    b_start = np.array([l['b_start'] for l in layers]); b_len = np.array([l['b_len'] for l in layers])
    obs = dict(compress=getattr(cfg, 'obs_compress', 'log1p_sqrt'), clip=float(getattr(cfg, 'obs_clip', 4.0)),
               div=float(getattr(cfg, 'obs_div', 4.0)), window=int(getattr(cfg, 'window_size', 4)),
               features=list(getattr(cfg, 'obs_features', ['vel', 'gyro', 'action'])))
    # ★10-01 재구성 뒤 Config 에 obs_compress·obs_clip·obs_div·obs_features 가 없다(위 기본값이 우연히 맞았을 뿐).
    #   런 폴더의 config.yaml(obs 절)을 정답으로 읽고, 체크포인트 쪽과 다르면 실기 관측이 학습과 달라지므로 멈춘다.
    cy = os.path.join(os.path.dirname(os.path.abspath(a.pt)), 'config.yaml')
    if os.path.exists(cy):
        import yaml
        o = (yaml.safe_load(open(cy)) or {}).get('obs') or {}
        ref = dict(compress=o.get('compress', obs['compress']), clip=float(o.get('clip', obs['clip'])), div=float(o.get('div', obs['div'])),
                   window=int(o.get('window', obs['window'])), features=list(o.get('features', obs['features'])))
        if ref != obs:
            sys.exit(f'관측 규약 불일치: 체크포인트 {obs} vs config.yaml {ref} — config.yaml 을 따르려면 확인 후 수정')
    else:
        print(f'⚠ {cy} 없음 — 체크포인트 기본값으로 관측 규약을 정함(확인 필요): {obs}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, theta=theta, W_start=W_start, W_len=W_len, W_rows=W_rows, W_cols=W_cols, b_start=b_start, b_len=b_len,
             act_name=str(info.get('act_name', getattr(cfg, 'activation_fn', 'silu'))), dimS=int(info['dimS']), nA=int(info['nA']),
             use_residual=bool(info.get('use_residual', False)), obs_json=json.dumps(obs), src=os.path.abspath(a.pt), net=a.net,
             agent=str(getattr(cfg, 'agent_type', '')), seed=int(getattr(cfg, 'seed', -1)), steps_done=int(sd.get('steps_done', -1)))
    print(f'저장 {a.out}: theta {theta.size} · 층 {len(layers)} · act {info.get("act_name")} · dimS {info["dimS"]} nA {info["nA"]} · obs {obs} · net {a.net}')


if __name__ == '__main__':
    main()
