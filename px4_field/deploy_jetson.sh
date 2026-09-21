#!/usr/bin/env bash
# deploy_jetson.sh — 실시간 정책 노드(f13_policy) 실행에 필요한 파일만 젯슨으로 복사 (2026-09-21)
#   사용:  JETSON=quad@192.168.x.x bash px4_field/deploy_jetson.sh          # 대상 ~/swirl_field/
#   젯슨 의존성: python3-numpy, python3-yaml, pymavlink, rclpy + px4_msgs(ActuatorAttack 포함 재빌드본). torch 불필요.
#   젯슨에서:  cd ~/swirl_field/px4_field
#              python3 f13_policy.py --model models/swirl_v4_s42.npz --pattern circle --attack rc --fs-url /dev/ttyACM0 --shadow   # 첫 비행은 shadow
#              python3 plot_policy_log.py --latest field_logs --live                                                                # 실시간 그림
set -eu
JETSON="${JETSON:?JETSON=user@host 필요}"
DEST="${DEST:-swirl_field}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FILES=(
  env/__init__.py env/knobs.py env/ukf_filter.py env/observation.py env/attack.py env/scenario.py env/reward.py env/failsafe_params.py
  calibration/calibration.json
  configs/base.yaml configs/newenv.yaml configs/newenv_v2.yaml configs/newenv_v3.yaml configs/newenv_v4.yaml configs/overlays/isaac.yaml
  px4_field/f13_policy.py px4_field/policy_np.py px4_field/export_policy.py px4_field/plot_policy_log.py
  px4_field/offboard_common.py px4_field/f5_pattern.py px4_field/traj_common.py px4_field/nis_from_ulog.py px4_field/fetch_ulog.py px4_field/check_ulog.py
  px4_field/models
  etc/analysis/rc_attack_trigger.py
)
for f in "${FILES[@]}"; do [ -e "$f" ] || echo "⚠ 없음: $f"; done
rsync -avR --exclude '__pycache__' "${FILES[@]}" "$JETSON:~/$DEST/"
echo "완료 → $JETSON:~/$DEST/   (f13 은 cfgload/torch 를 쓰지 않는다 — YAML 병합으로 공격 설정을 읽는다)"
